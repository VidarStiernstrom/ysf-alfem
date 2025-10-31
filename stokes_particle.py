"""
Script for verifying the Stokes solver on the particle mesh. 
The code is heavily based on the FEniCSx resources for solving Stokes equation, found at
    - https://jsdokken.com/fenics22-tutorial/comparing_elements.html
    - https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_stokes.html 
    
Run with
    `python stokes_particle.py mode`
where mode is either `plot` (plotting the mesh, velocity and pressure fields) or `conv`
(performs a convergence study using a manufactued solution).
"""

__author__ = "Vidar Stiernström"
__mail__ = "vidarsti@kth.se"
__copyright__ = "Copyright (c) 2025 {}".format(__author__)

import sys
import argparse
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import pyvista
import matplotlib.pylab as plt
import numpy as np

import basix.ufl
from dolfinx import fem, mesh
import dolfinx.plot as plot

from particle_mesh import create_mesh, refine_mesh, plot_mesh
from variational_forms import create_bilinear_form
from solvers import create_nullspace, assemble_system_matrix, assemble_rhs_vector, create_preconditioner, create_iterative_solver, create_direct_solver
from helpers import assemble_scalar, validate_args


from ufl import (
    SpatialCoordinate,
    TestFunction,
    as_vector,
    cos,
    nabla_div,
    dx,
    nabla_grad,
    inner,
    sym,
    pi,
    sin
)

def u_ex(x):
    """ Velocity component of manufactured solution.

        Args:
            x: Spatial coordinates of mesh
        Returns:
            UFL Expression
    """
    sinx = sin(pi * x[0])
    siny = sin(pi * x[1])
    cosx = cos(pi * x[0])
    cosy = cos(pi * x[1])
    c_factor = 2 * pi * sinx * siny
    return c_factor * as_vector((cosy * sinx, -cosx * siny))
    
def p_ex(x):
    """ Pressure component of manufactured solution.

        Args:
            x: Spatial coordinates of mesh
        Returns:
            UFL Expression
    """
    return sin(2 * pi * x[0]) * sin(2 * pi * x[1])

def source(x):
    """ Source function used to obtained manufactured solutions.

        Args:
            x: Spatial coordinates of mesh
        Returns:
            UFL Expression
    """
    u, p = u_ex(x), p_ex(x)
    return -nabla_div(nabla_grad(u)) + nabla_grad(p)

# TODO: Should probably jut interpolate the above UFL symbolic expressions
#       instead of reimplementing them as numpy expressions...
def u_ex_np(x):
    """ Velocity component of manufactured solution as a numpy array

        Args:
            x: Spatial coordinates of mesh
        Returns:
            numpy array
    """
    sinx = np.sin(np.pi * x[0])
    siny = np.sin(np.pi * x[1])
    cosx = np.cos(np.pi * x[0])
    cosy = np.cos(np.pi * x[1])
    c_factor = 2 * np.pi * sinx * siny
    u0 = c_factor * cosy * sinx
    u1 = -c_factor * cosx * siny
    return np.array([u0, u1])

def p_ex_np(x):
    """ Pressure component of manufactured solution as a numpy array

        Args:
            x: Spatial coordinates of mesh
        Returns:
            numpy array
    """
    return np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])

def create_linear_form(V, Q):
    """ Create the linear form for the Stokes problem with
        the manufactured source function.

        Args:
            V:  DOLFINx FunctionSpace
                Function space for velocity
            Q:  DOLFINx FunctionSpace
                Function space for pressure
        Returns:
            UFL Form
    """
    v, q = TestFunction(V), TestFunction(Q)
    msh = V.mesh 
    f = source(SpatialCoordinate(msh))
    return fem.form([inner(f, v) * dx, inner(fem.Constant(msh, 0.0), q) * dx])

def create_dirichlet_bc(V):
    """ Create Dirichlet (slip) boundary conditions using the manufactured
        solutions as data on the boundaries.

        Args:
            V:  DOLFINx FunctionSpace
                Function space for velocity
        Returns:
            List of DOLFINx DirichletBC
    """
    msh = V.mesh
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    bdry_facets = mesh.exterior_facet_indices(msh.topology)
    dofs_u = fem.locate_dofs_topological(V, tdim - 1, bdry_facets)
    
    # Use exact solution as data to impose Dirichlet (slip) condition.
    g = fem.Function(V)
    g.interpolate(u_ex_np)
    g.x.scatter_forward()
    return [fem.dirichletbc(g, dofs_u)]

def compute_errors(u, p):
    """ Compute the H_1 and L_2 errors of u and p respectively.

        Args:
            u:  DOLFINx Function
                Velocity function
            p:  DOLFINx Function
                Pressure function
        Returns:
            velocity_error: Float
            H_1 error of velocity
            L_2 error of pressure
    """
    msh = u.function_space.mesh
    x = SpatialCoordinate(msh)
    error_u = u - u_ex(x)
    H1_u = inner(error_u, error_u) * dx
    H1_u += inner(sym(nabla_grad(error_u)), sym(nabla_grad(error_u))) * dx
    velocity_error = np.sqrt(assemble_scalar(H1_u, msh.comm))

    error_p = p - p_ex(x)
    L2_p = fem.form(error_p * error_p * dx)
    pressure_error = np.sqrt(assemble_scalar(L2_p, msh.comm))
    return velocity_error, pressure_error

def compute_mean_pressure(p):
    """ Compute the (approximate) mean value of pressure, by numerically
        integrating it over the domain

        Args:
            p:  DOLFINx Function
                Pressure function
        Returns:
            mean_p: Float
            mean value of pressure
    """
    msh = p.function_space.mesh
    L2_p = fem.form(p * dx)
    mean_p = assemble_scalar(L2_p, msh.comm)
    return mean_p

def solve_stokes(msh, order_u, order_p, iterative = True, regularize = False):
    """ Solves the Stokes problem.
        
        Args:
            order_u:  int
                      element order for velocity element
            order_p:  int
                      element order for pressure element
            iterative:  bool
                        If True, the problem is solved using preconditioned MINRES (iterative solver)
                        If False, the problem is solved using SuperLU (direct solver)
            regularise: bool
                        If True, the Stokes problem is regularized by adding a penalty
                        term to the zero-block of the system, enforcing zero mean pressure
                        If False and an iterative solver is used, the pressure nullspace of
                        of the problem is instead eliminated. This does not work when using the direct solver.
                    
        Returns:
            u:  DOLFINx Function
                Velocity solution
            p:  DOLFINx Function
                Pressure solution
            e_u: float
                  H_1 error of velocity.
            e_p: float
                  L_2 error of pressure.
    """


    elem_u = basix.ufl.element("Lagrange", msh.topology.cell_name(), order_u, shape=(msh.geometry.dim, ))
    elem_p = basix.ufl.element("Lagrange", msh.topology.cell_name(), order_p)
    V = fem.functionspace(msh, elem_u)
    Q = fem.functionspace(msh, elem_p)

    u, p = fem.Function(V, name = "u"), fem.Function(Q, name = "p")
    bilin_form = create_bilinear_form(V, Q, 1.0, 1.0, 'CT19-alg1', regularize)
    lin_form = create_linear_form(V, Q)

    bcs = create_dirichlet_bc(V)
    A = assemble_system_matrix(bilin_form, bcs, iterative)
    b = assemble_rhs_vector(bilin_form, lin_form, bcs, iterative)
    
    # Create KSP solver and solution vector
    if iterative:
        if not regularize:
            # Instead of regularizing we eliminate the part of the nullspace
            # spanning constant pressure solutions.
            nsp = create_nullspace(lin_form)
            assert nsp.test(A)
            A.setNearNullSpace(nsp)
        P = create_preconditioner(Q, bilin_form, bcs)
        ksp = create_iterative_solver(A, P, msh.comm)
        w = PETSc.Vec().createNest([u.x.petsc_vec, p.x.petsc_vec])
    else: # direct solver
        assert regularize, "Null space elimination not implemented for direct solver"
        ksp = create_direct_solver(A, msh.comm)
        w = A.createVecLeft()

    ksp.solve(b, w)
    assert ksp.getConvergedReason() > 0
    
    if not iterative: # Extract u and p manually from solution vector w
        offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        u.x.array[:offset] = w.array_r[:offset]
        p.x.array[: (len(w.array_r) - offset)] = w.array_r[offset:]
    
    u.x.scatter_forward()
    p.x.scatter_forward()
    e_u, e_p = compute_errors(u, p)
    return u, p, e_u, e_p

def plot_scalar(v, do_save_fig):
    """ Plots a scalar function
        
        Args:
            v:  DOLFINx Function
            do_save_fig: bool
    """
    # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
    # otherwise create interactive plot
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)

    V = v.function_space
    scalar_name = v.name
    topology, cell_types, geometry = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data[scalar_name] = v.x.array.real
    grid.set_active_scalars(scalar_name)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    plotter.add_title(scalar_name)
    if not pyvista.OFF_SCREEN:
        if do_save_fig:
            dirname = Path("results/stokes_particle")
            dirname.mkdir(exist_ok=True, parents=True)
            filename = scalar_name + ".svg"
            plotter.save_graphic(dirname/filename)
        plotter.show()
            
def solve_and_plot(geo_dims, elem_orders, ns, iterative, regularize, do_plot_mesh = True, do_save_fig = True):
    """ Solves the Stokes problem, plotting the solutions and their errors, and checks the mean
        value of pressure. 
    """
    # Unpack input
    rect_dim, disk_dim = geo_dims
    order_u, order_p = elem_orders
    n_rect, n_circle = ns
    # Create mesh
    msh, _, _ = create_mesh(rect_dim, disk_dim, n_rect, n_circle)
    if do_plot_mesh:
        plot_mesh(msh)
    # Solve problem
    u, p, _, _ = solve_stokes(msh, order_u, order_p, iterative = iterative, regularize = regularize)
    
    # Plot results and errors
    V = u.function_space
    Q = p.function_space
    # Pressure
    plot_scalar(p, do_save_fig)
    p_exact = fem.Function(Q)
    p_exact.interpolate(p_ex_np), p_exact.x.scatter_forward()
    p_err = fem.Function(Q, name = "p_err")
    p_err.x.array[:] = p_exact.x.array - p.x.array
    plot_scalar(p_err, do_save_fig)
    # x-component of velocity
    Vx, _ = V.sub(0).collapse()
    ux = fem.Function(Vx, name = "ux")
    ux.interpolate(u.sub(0)), ux.x.scatter_forward()
    plot_scalar(ux, do_save_fig)
    ux_exact = fem.Function(Vx)
    ux_exact.interpolate(lambda x : u_ex_np(x)[0]), ux_exact.x.scatter_forward()
    ux_err = fem.Function(Vx, name = "ux_err")
    ux_err.x.array[:] = ux_exact.x.array - ux.x.array
    plot_scalar(ux_err, do_save_fig)
    # y-component of velocity
    Vy, _ = V.sub(1).collapse()
    uy = fem.Function(Vy,name = "uy")
    uy.interpolate(u.sub(1)), uy.x.scatter_forward()
    plot_scalar(uy, do_save_fig)
    uy_exact = fem.Function(Vy)
    uy_exact.interpolate(lambda x : u_ex_np(x)[1]), ux_exact.x.scatter_forward()
    uy_err = fem.Function(Vy, name = "uy_err")
    uy_err.x.array[:] = uy_exact.x.array - uy.x.array
    plot_scalar(uy_err, do_save_fig)

    # Print mean value of pressure
    print("Mean value of computed pressure:", compute_mean_pressure(p))
    print("Mean value of exact pressure:", compute_mean_pressure(p_exact))

def convergence_study(geo_dims, elem_orders, reference_rate, n0, refinements, iterative, regularize):
    """ Performs a convergence study using the manufactured solution.
    """
    # Unpack input
    rect_dim, disk_dim = geo_dims
    order_u, order_p = elem_orders
    
    # Initialize mesh spacing and error vectors
    hs = np.zeros(refinements)
    u_errors = np.zeros(refinements)
    p_errors = np.zeros(refinements)

    # Solve on initial mesh
    n_rect = n0
    n_circle = round(3/2*n_rect)
    msh, _, _ = create_mesh(rect_dim, disk_dim, n_rect, n_circle)
    _, _, u_errors[0], p_errors[0] = solve_stokes(msh, order_u, order_p, iterative = iterative, regularize = regularize)
    L = rect_dim[0]
    hs[0] = L / n0

    # Solve on refined meshes
    N = n0
    for i in range(1, refinements):
        N = (N-1)*2
        msh, _, _ = refine_mesh()
        _, _, u_errors[i], p_errors[i] = solve_stokes(msh, order_u, order_p, iterative = iterative, regularize = regularize)
        hs[i] = L / N
    
    # Error plots
    if MPI.COMM_WORLD.rank == 0:
        legend = []
        if reference_rate is not None:
            y_value = u_errors[-1] * 1.4
            plt.plot(
                [hs[0], hs[-1]],
                [y_value * (hs[0] / hs[-1]) ** reference_rate, y_value],
                "k--",
            )
            legend.append(f"order {reference_rate}")

        plt.plot(hs, u_errors, "bo-")
        plt.plot(hs, p_errors, "ro-")
        legend += [r"$H^1(\mathbf{u_h}-\mathbf{u}_{ex})$", r"$L^2(p_h-p_{ex})$"]
        plt.legend(legend)
        plt.xscale("log")
        plt.yscale("log")
        plt.axis("equal")
        plt.ylabel("Error in energy norm")
        plt.xlabel("$h$ (reference length scale of element)")
        plt.xlim(plt.xlim()[::-1])
        plt.grid(True)
        plt.show()


def main(mode):
    """ Runs the Stokes solver with the manufactured solution.

        Args:
              mode: String with either 'plot' or 'conv'.
                    'plot': Solve the problem and plot the solution and errors.
                    'conv': Perform convergence study.
    """
    L = 1. # Rectangle half length
    H = 1. # Rectangle half height
    c_x = c_y = 0.5 # disk center
    r = 0.25 # Disk radii
    rect_dim = (L, H)
    disk_dim = (c_x, c_y, r)
    geo_dims = (rect_dim, disk_dim)
    # Taylor-Hood elements
    # Set orders for velocity and pressure here
    order_u = 2
    order_p = 1
    elem_orders = (order_u, order_p)

    # Specify solver methods.
    iterative = True
    regularize = True
    
    if mode == 'plot':
        n_rect = 20 # Number of nodes along 1 side of the rectangle (exterior boundary)
        n_circle = round(n_rect*3/2) # Number of nodes along the circle (interior boundary)
        ns = (n_rect, n_circle)
        solve_and_plot(geo_dims, elem_orders, ns, iterative, regularize)
    elif mode == 'conv':
        reference_rate = 2
        refinements = 5
        n0 = 10
        convergence_study(geo_dims, elem_orders, reference_rate, n0, refinements, iterative, regularize)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type = str)
    args = parser.parse_args()
    mode = args.mode
    validate_args(mode, ('plot', 'conv'))
    sys.exit(main(mode))
