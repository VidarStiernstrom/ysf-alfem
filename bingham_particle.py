"""
Script for simulating a particle sedimenting in a Bingham fluid, or equivalently
Bingham fluid flow around a cylinder.

The problem is formulated as a variational inequality using the augmented Lagrange method (ALM).
The ALM is then solved using Uzawa iterations, where the variational form is discretized with Taylor-Hood
finite elements. A few different ALM formulations are implemented (with varying sucess!), based on
- Roquet, Samarito 2003, https://doi.org/10.1016/S0045-7825(03)00262-7
- Chaparian, Tammisola 2019, https://doi.org/10.1016/j.jnnfm.2019.104148
Currently, the best working formulation appears to be that of Roquet, Samarito 2003.

Run with
    `mpirun -np N python bingham_particle.py`
where N specifies the number of processes. 
See main method and its instantiation for more details and options to set problem/solver parameters.
"""

__author__ = "Vidar Stiernström"
__mail__ = "vidarsti@kth.se"
__copyright__ = "Copyright (c) 2025 {}".format(__author__)

import sys
from pathlib import Path

from petsc4py import PETSc

import numpy as np
from scipy.io import savemat

import basix.ufl
import dolfinx.fem.petsc
from dolfinx import fem, io, default_scalar_type

from particle_mesh import create_mesh, plot_mesh, on_exterior_boundary, on_interior_boundary
from variational_forms import create_bilinear_form, create_linear_form
from solvers import assemble_system_matrix, assemble_rhs_vector, create_direct_solver, create_iterative_solver, create_preconditioner
from helpers import strain_rate, deformation_rate, vector_norm, tensor_norm, assemble_scalar

from ufl import (
    conditional,
    TestFunction,
    TrialFunction,
    dx,
    nabla_grad,
    curl,
    inner,
)


def relaxed_strain_rate(T, T_yield, r, eta_p, var_form_str):
    """ Computes the updated relaxed strain rate used in the Uzawa ALM algorithm
     
        Args:
            T:  DOLFINx Function
                Intermediary stress tensor
            T_yield:  DOLFINx Constant
                      The yield stress
            r:  DOLFINx Constant
                The augmentation parameter
            eta_p:  DOLFINx Constant
                    The plastic viscosity
            var_form_str: String
                          Specifies the variational form used.
                          See variational_forms.py for further details.

        Returns:
            UFL expression
    """

    T_norm = tensor_norm(T)
    if var_form_str == 'RS03':
        return conditional(T_norm < T_yield , 0 * T, ( 1.0 - T_yield / T_norm ) * T / ( 2 * (r + eta_p)))
    else:
        return conditional(T_norm <= T_yield, 0 * T, ( 1.0 - T_yield / T_norm ) * T / ( (1 + r) * eta_p ))
    
def intermediary_stress(T_n, u, r, eta_p):
    """ Computes the intermediary stress used in the Uzawa ALM algorithm.

        The intermediary stress is used in the explicit update of the stress
        and strain rate tensors.
          
        Args:
            T_n:  DOLFINx Function
                  Stress tensor at previous iteration
            u:  DOLFINx Function
                Velocity vector at next iteration
            r:  DOLFINx Constant
                The augmentation parameter
            eta_p:  DOLFINx Constant
                    The plastic viscosity 

        Returns:
            UFL expression
            Expression for updating the intermediary stress
    """

    return T_n + r * eta_p * strain_rate(u)

def von_mises_criterion(T, T_yield):
    """ Evaluates the von Mises criterion

        Args:
            T:  DOLFINx Function
                  Stress tensor
            T_yield:  DOLFINx Constant
                      The yield stress

        Returns:
            UFL expression
            Expression returning 1 if the tensor norm (second invariant) of T is above
            the yield stress in the interpolating point, and 0 otherwise.
    """

    T_norm = tensor_norm(T)
    return conditional(T_norm <= T_yield, 0.0, 1.0)    

def create_slip_bc(V, rect_bc_val, circ_bc_val, rect_dim, circ_dim):
    """ Creates slip boundary conditions.
        
        Args:
            V:  DOLFINx FunctionSpace
                The function space for velocity (vector-valued)
            rect_bc_val: numpy two-element array
                         Values for slip on the rectangle (exterior) boundary
            rect_bc_val: numpy two-element array
                         Values for slip on the circle (interior) boundary             
            rect_dim: Tuple of floats (L,H).
                      L: rectangle length in x-direction.
                      H: rectangle length in y-direction.
            circ_dim: Triplet of floats (c_x,c_y,r).
                      c_x: x-coordinate of circle center.
                      c_y: y-coordinate of circle center.
                      r:   radius of circle.

        Returns:
            list of DOLFINx DirichletCondition
    """

    # DOFs for exterior rectangle boundary
    dofs_u_rect = fem.locate_dofs_geometrical(V, lambda x : on_exterior_boundary(x, rect_dim))
    # DOfs for inner circle boundary
    dofs_u_circ = fem.locate_dofs_geometrical(V, lambda x : on_interior_boundary(x, circ_dim))
    return [fem.dirichletbc(rect_bc_val, dofs_u_rect, V), 
            fem.dirichletbc(circ_bc_val, dofs_u_circ, V)]

def check_convergence(msh, u, u_n, q, q_n, tol, var_form_str):
    """ Computes iteration errors and check for convergence of the Uzawa ALM algorithm.

        Computes the iteratior error for velocity, relaxed strain rate, and the strain rate constraint
        See https://doi.org/10.1016/j.jnnfm.2019.104148.

        
        Args:
            msh:  DOLFINx Mesh
            u:  DOLFINx Function
                Updated velocity vector
            u_n:  DOLFINx Function
                  Previous velocity vector
            q:  DOLFINx Function
                Updated relaxed strain rate
            q_n:  DOLFINx Function
                  Previous relaxed strain rate
            tol: float
                  Tolerance value for convergence
            var_form_str: String
                          Specifies the variational form used.
                          See variational_forms.py for further details.

        Returns:
            converged: Bool
                       True if the maximal error is less than the tol
            res:  float
                  The largest error (residual)
            error_u:  float
                      Velocity iteration error
            error_q:  float
                      Relaxed strain rate iteration error
            error_gamma:  float
                          Error in constraint
    """

    converged = False
    error_u = assemble_scalar(vector_norm(u - u_n)*dx, msh.comm)
    error_q = assemble_scalar(tensor_norm(q - q_n)*dx, msh.comm)
    if var_form_str == 'RS03':
        # In RS03, the constraint is q = deformation_rate(u) rather than strain_rate
        error_gamma = assemble_scalar(tensor_norm(deformation_rate(u) - q)*dx, msh.comm)
    else:
        error_gamma = assemble_scalar(tensor_norm(strain_rate(u) - q)*dx, msh.comm)
    res = np.max([error_u, error_q, error_gamma])
    if res < tol:
        converged = True
    return converged, res, error_u, error_q, error_gamma

    
def compute_stream_function(Vs, u, rect_dim):
    """ Compute a stream function associated with the velocity.

        Computes the stream function defined by the Poisson problem with
        the curl of velocity as a forcing, homogenous Dirichlet conditions on 
        the exterior boundary and natural boundary conditions on the interior.
    
        Args:
              Vs: DOLFINx FunctionSpace
                  Function space for stream function
              u:  DOLFINx Function
                  Velocity vector
              rect_dim: Tuple of floats (L,H).
                    L: rectangle length in x-direction.
                    H: rectangle length in y-direction.
        Returns:
              DOLFINx LinearProblem
              The solved problem.
    """
    # Dirichlet BCs on exterior boundary. On inner boundary we impose Neumann BCs (handled in the bilinear form)
    dofs_exterior = fem.locate_dofs_geometrical(Vs, lambda x : on_exterior_boundary(x, rect_dim))
    bc = fem.dirichletbc(fem.Constant(Vs.mesh, default_scalar_type(0)), dofs_exterior, Vs)

    psi, phi = TrialFunction(Vs), TestFunction(Vs)
    a = inner(nabla_grad(psi),nabla_grad(phi))*dx
    L = -phi*curl(u)*dx
    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    return problem.solve()

def write_VTX(writer, iter, ux, uy, T_norm, q_norm, u, yielded_elems, T_norm_expr, q_norm_expr, yield_area_expr):
    """ Writes fields in the Uzawa ALM iteration to VTK

        TODO: Check if the functions and expressions used for writing could be created
              inside this function, without affecting performance etc. Would clean up the code a lot.
            
        Args:
              writer: VTX writer
              iter: int
                    Iteration counter
              ux:   DOLFINx Function
                    x-component of velocity
              uy:   DOLFINx Function
                    y-component of velocity
              T_norm: DOLFINx Function
                     Tensor norm (second invariant) of stress
              q_norm: DOLFINx Function
                      Tensor norm (second invariant) of strain rate
              u:  DOLFINx Function
                  Velocity vector
              yield_area: DOLFINx Function
                          Function storing the yielded elements
              T_norm_expr:  UFL Expression
                            Expression for evaluating the stress tensor norm
              q_norm_expr:  UFL Expression
                            Expression for evaluating the strain rate norm
              yield_area_expr:  UFL Expression
                                Expression for evaluating the yielded area
    """
    
    ux.interpolate(u.sub(0))
    ux.x.scatter_forward()
    
    uy.interpolate(u.sub(1))
    uy.x.scatter_forward()
    
    T_norm.interpolate(T_norm_expr)
    T_norm.x.scatter_forward()

    q_norm.interpolate(q_norm_expr)
    q_norm.x.scatter_forward()
    
    yielded_elems.interpolate(yield_area_expr)
    yielded_elems.x.scatter_forward()
    
    writer.write(iter)


def main(geo_dim, ns, particle_velocity, fluid_params, solver_params, do_plot_mesh = False):
    """ Runs the Uzawa ALM solver, writing the solution fields (velocity, stress, strain rate etc) to VTK
        as well as saving the solver history to .mat file.
            
        Args:
              geo_dim: (L, H, rad)
                        L: Length of domain in x-direction
                        H: Length of domain in y-direction
                        rad: radius of particle (circle centered in the middle of the domain)
              ns: (n_rect, n_circ)
                  n_rect: number of nodes along 1 side of the rectangle (exterior) boundary
                  n_circ: number of nodes along circle (interior) boundary
              particle_velocity: [float, float]
                                  velocity of particle (values used for the slip condition on the interior boundary).
              fluid_params: (plastic_viscosity, yield_stress
                            plastic_viscosity: Plastic viscosity of the fluid.
                            yield_stress: Yield stress of the fluid.
              solver_params: (aug_param, tol, maxiter, var_form_str, iterative, write_freq)
                              aug_param: Augmentation parameter in ALM formulation
                              tol: Tolerance of Uzawa algorithm
                              maxiter: Maximum number of iterations in Uzawa algorithm
                              var_form_str: String specifiyng the variational form to be used.
                                            Either 'CT19-alg1', 'CT19-alg3', 'freeFEM', 'RS03'
                                            See the variational_forms.py for more info.
                              iterative: Bool specifying whether the Stokes solver should use an
                                          interative (preconditioned MINRES) or direct (SuperLU) solver.
                              write_freq: The frequency at which the results are written to VTK.
              do_plot_mesh: Bool specifying whether to plot the mesh or not.
    """
    ## Unpack arguments
    L, H, rad = geo_dim # domain
    n_rect, n_circ = ns # resolution
    fluid_params
    plastic_viscosity, yield_stress = fluid_params
    aug_param, tol, maxiter, var_form_str, iterative, write_freq = solver_params

    ## Mesh 
    c_x = L/2.0 # Circle center x-coord
    c_y = H/2.0 # Circle center y-coord
    rect_dim = (L, H) # Rectangle dimensions
    circ_dim = (c_x, c_y, rad) # Circle dimensions
    msh, _, _ = create_mesh(rect_dim, 
                            circ_dim, 
                            n_rect, 
                            n_circ)
    if do_plot_mesh:
        plot_mesh(msh)

    ## Material parameters
    eta_p = fem.Constant(msh, default_scalar_type((plastic_viscosity))) # Plastic viscosity
    T_yield = fem.Constant(msh, default_scalar_type((yield_stress))) # Fluid yield stress    

    ## Functions and function spaces
    # Mixed (Taylor-Hood) finite element spaces with DG tensor spaces for Lagrange multipliers
    # and relaxed strain rate
    
    # Velocity space:
    # P2 vector-valued CG space
    V = fem.functionspace(msh, basix.ufl.element("Lagrange",
                                            msh.topology.cell_name(),
                                            2,
                                            shape=(msh.geometry.dim,)))
    # Pressure space:
    # P1 scalar-valued CG space
    Q = fem.functionspace(msh, basix.ufl.element("Lagrange", 
                                            msh.topology.cell_name(),
                                            1))
    # Stress space:
    # P1 symmetric two-tensor-valued DG space
    W = fem.functionspace(msh, basix.ufl.element("Discontinuous Lagrange",
                                             msh.topology.cell_name(),
                                             1,
                                             shape=(msh.geometry.dim, msh.geometry.dim), 
                                             symmetry=True))
    # Scalar DG space (for computing e.g., local norms)
    S = fem.functionspace(msh, basix.ufl.element("Discontinuous Lagrange", 
                                            msh.topology.cell_name(),
                                            1))
    # Streamline space
    # P2 scalar-valued CG space
    Vs = fem.functionspace(msh, basix.ufl.element("Lagrange",
                                msh.topology.cell_name(),
                                2))
    
    u = fem.Function(V) # Velocity vector at next Uzawa iteration
    u_n = fem.Function(V) # Velocity vector at current Uzawa iteration
    p = fem.Function(Q) # Pressure at next Uzawa iteration

    T = fem.Function(W) # Stress tensor at next Uzawa iteration
    T_n = fem.Function(W) # Stress tensor at current Uzawa iteration
    q = fem.Function(W) # Relaxed strain rate tensor at next Uzawa iteration
    q_n = fem.Function(W) # Relaxed strain rate tensor at current Uzawa iteration

    # Functions and expressions for plotting
    ux = fem.Function(S,name="ux")
    uy = fem.Function(S,name="uy")
    T_norm = fem.Function(S,name = "||T||")
    q_norm = fem.Function(S,name = "||q||")
    yield_area = fem.Function(S, name = "yield_area")
    T_norm_expr = fem.Expression(tensor_norm(T), S.element.interpolation_points())
    q_norm_expr = fem.Expression(tensor_norm(q), S.element.interpolation_points())
    yield_area_expr = fem.Expression(von_mises_criterion(T, T_yield), S.element.interpolation_points())

    ## Stokes solver setup
    # Create the bilinear and linear form for the Stokes problem
    # a(v,u,p,q) = L(v,q,T_n,q_n)
    regularize = True
    r = fem.Constant(msh, default_scalar_type((aug_param))) # ALM augmentation parameter
    a = create_bilinear_form(V, Q, r, eta_p, var_form_str, regularize)
    L = create_linear_form(V, Q, T_n, q_n, r, eta_p, var_form_str)

    # Boundary conditions
    circ_bc = np.array(particle_velocity, dtype=default_scalar_type)
    rect_bc = np.array([0, 0], dtype=default_scalar_type)
    bcs = create_slip_bc(V, rect_bc, circ_bc, rect_dim, circ_dim)
    
    # Assemble system Aw = b
    A = assemble_system_matrix(a, bcs, iterative)
    if iterative:
        P = create_preconditioner(Q, a, bcs)
        ksp = create_iterative_solver(A, P, msh.comm)
        w = PETSc.Vec().createNest([u.x.petsc_vec, p.x.petsc_vec])
    else:    
        ksp = create_direct_solver(A, msh.comm)
        w = A.createVecLeft()
    
    ## UFL expressions for computing the intermediary stress tensor
    # and updating the relaxed strain rate tensor and the Uzawa ALM algorithm.
    # The expressions are evaluated in the interpolation points of the DG tensor elements.
    intermediary_stress_expr = fem.Expression(intermediary_stress(T_n, u, r, eta_p), W.element.interpolation_points())
    relaxed_strain_rate_expr = fem.Expression(relaxed_strain_rate(T, T_yield, r, eta_p, var_form_str), W.element.interpolation_points())

    ## VTK writer
    dirname = Path("results/bingham_particle")
    dirname.mkdir(exist_ok=True, parents=True)
    filename = "fields_r" + str(int(aug_param)) + ".bp"
    writer = io.VTXWriter(msh.comm, dirname / var_form_str / filename, [ux, uy, T_norm, q_norm, yield_area], "BP4", io.VTXMeshPolicy.reuse)

    ## Error history
    error_u_hist = []
    error_q_hist = []
    error_gamma_hist = []

    ## Initialize stress and strain rate tensors for Uzawa iteration
    T_n.x.array[:] = 0.
    q_n.x.array[:] = 0.

    ## Uzawa iterative solver    
    converged = False
    iter = 0
    write_VTX(writer, iter, ux, uy, T_norm, q_norm, u, yield_area, T_norm_expr, q_norm_expr, yield_area_expr)
    
    if msh.comm.rank == 0:
        print("-----------------------------")
        print("Starting Uzawa ALM solver")
        print("ALM variational form: " + var_form_str)
        print("Tolerance: " + str(tol))
        print("Max iterations: " + str(maxiter))

    while not converged and iter < maxiter:
        # Uzawa iteration
        # Step 1.
        # Assemble rhs with current stress/relaxed strain rate
        b = assemble_rhs_vector(a, L, bcs, iterative)
        # Solve Stokes problem
        ksp.solve(b,w) # Solve for stacked solution vector w = [u p]
        assert ksp.getConvergedReason() > 0
        if not iterative: # Extract u and p manually from solution vector w
            offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
            u.x.array[:offset] = w.array_r[:offset]
            p.x.array[: (len(w.array_r) - offset)] = w.array_r[offset:]
        
        u.x.scatter_forward()
        p.x.scatter_forward()
        
        # Step 2. Explicit updates of stress and relaxed strain rate tensors
        # Compute intermediate stress, store temporary in T
        T.interpolate(intermediary_stress_expr)
        T.x.scatter_forward()
        # Update relaxed strain rate computed using intermediate stress
        q.interpolate(relaxed_strain_rate_expr)
        q.x.scatter_forward()
        # Update stress
        # (No need to interpolate a UFL expression here since it can be
        #  expressed using numpy operations)
        if var_form_str == 'RS03':
            T.x.array[:] =  T.x.array - 2 * r * q.x.array
        else:
            T.x.array[:] =  T.x.array - r * eta_p * q.x.array
        
        # Step 3. Check convergence
        converged, residual, error_u, error_q, error_gamma = check_convergence(msh, u, u_n, q, q_n, tol, var_form_str)
        iter += 1
        
        # Print iteration info
        if msh.comm.rank == 0:
            print("-----------------------------")
            print("Iteration " + str(iter) + " (max " + str(maxiter)+")")
            print("-----------------------------")
            print("Residual: " + str(residual) + " (tol " + str(tol)+")") 
            print("||u - u_n||_L2: " + str(error_u))
            print("||q - q_n||_L2: " + str(error_q))
            if var_form_str == 'RS03':
                print("||(deformation_rate(u) - q||_L2: " + str(error_gamma))
            else:
                print("||(strain_rate(u) - q||_L2: " + str(error_gamma))
            sys.stdout.flush()
            
            error_u_hist.append(error_u)
            error_q_hist.append(error_q)
            error_gamma_hist.append(error_gamma) 
        
        # Step 4. Update iteration varibles n -> n+1
        u_n.x.array[:] = u.x.array
        T_n.x.array[:] = T.x.array
        q_n.x.array[:] = q.x.array

        # TBD: These scatters are probably not needed.
        u_n.x.scatter_forward()
        T_n.x.scatter_forward()
        q_n.x.scatter_forward()
        
        if iter % write_freq == 0:
            write_VTX(writer, iter, ux, uy, T_norm, q_norm, u, yield_area, T_norm_expr, q_norm_expr, yield_area_expr)
    
    if msh.comm.rank == 0:
        print("-----------------------------")
        if converged:
            print("ALM stopped due to tolerance reached.")
        elif iter == maxiter:
            print("ALM finished due to max iterations reached.")

    # Write final iteration and close
    write_VTX(writer, iter, ux, uy, T_norm, q_norm, u, yield_area, T_norm_expr, q_norm_expr, yield_area_expr)
    writer.close()

    psi = compute_stream_function(Vs, u, rect_dim)
    psi.name = 'psi'
    filename = "streamlines_r" + str(int(aug_param)) + ".bp"
    with io.VTXWriter(msh.comm, dirname / var_form_str / filename, [psi], "BP4", io.VTXMeshPolicy.reuse) as writer:
        writer.write(0)
        writer.close()
    

    if msh.comm.rank == 0:
        iter_arr = range(iter)
        mdict = {"iter": iter_arr,
                "error_u": error_u_hist,
                "error_q": error_q_hist,
                "error_gamma": error_gamma_hist,
                "tol": [tol]}
        filename = "residuals_r" + str(int(aug_param)) + ".mat"
        savemat(dirname / var_form_str / filename, mdict)


if __name__ == "__main__":

    # Domain
    L = 15 # Rectangle length x
    H = 15 # Rectangle length y
    rad = 1.0 # circular particle radius
    geo_dim = (L, H, rad)
    
    # Resolution
    n_rect = 30 # number of grid points on rectangle (exterior) boundary
    n_circ = 100 # number of grid points on circle (interior) boundary
    ns = (n_rect, n_circ)
    
    # velocity of sedimenting particle
    particle_velocity = [1.0, 0] # x,y components
    
    # Fluid parameters
    plastic_viscosity = 1.0
    yield_stress = 100.0
    fluid_params = (plastic_viscosity, yield_stress)
    
    # Solver params
    aug_param = 1e3 # Augmentation parameter
    tol = 1e-4 # Tolerance of Uzawa
    maxiter = 4000 # Max iterations of Uzawa 
    var_form_str = 'CT19-alg3' # String specifying the ALM variational form
    iterative_stokes_solver = False # Iterative (True) or direct (False) Stokes solver
    write_freq = 10 # Frequency for writing results to file
    solver_params = (aug_param, tol, maxiter, var_form_str, iterative_stokes_solver, write_freq)

    main(geo_dim, ns, particle_velocity, fluid_params, solver_params)

    sys.exit()


