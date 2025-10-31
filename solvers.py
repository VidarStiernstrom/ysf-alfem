"""
Functions assembling the linear system and setting up PETSc KSP solvers
for the Stokes-type flow problem.
"""
__author__ = "Vidar Stiernström"
__mail__ = "vidarsti@kth.se"
__copyright__ = "Copyright (c) 2025 {}".format(__author__)

from petsc4py import PETSc
import dolfinx.fem.petsc

from dolfinx import fem

from ufl import (
    TestFunction,
    TrialFunction,
    dx,
    inner,
)

def create_iterative_solver(A, P, comm, rtol = 1e-9):
    """ Creates an preconditioned iterative solver for the Stokes-type system.

        Creates a preconditioned nested block iterative solver for the
        system A*w = b. The iterative solver uses MINRES and is 
        block-preconditioned with AMG for the velocity block and Jacobi
        for the pressure block.

        Args:
            A:  PETSc matrix
                Linear system matrix
            P:  PETSc matrix
                Preconditioner
            comm: MPI communicator
            rtol: relative tolerance of iterative solver

        Returns:
            PETSC KSP solver object
    """

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A, P)
    ksp.setType("minres")
    ksp.setTolerances(rtol = rtol)
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    nested_IS = P.getNestISs()
    ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]))

    # Set the preconditioners for each block
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("gamg")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("jacobi")

    # Monitor the convergence of the KSP
    ksp.setFromOptions()
    return ksp

def create_direct_solver(A, comm):
    """ Creates an preconditioned iterative solver for the Stokes-type system.

        Creates an LU-factorized direct solver using SuperLU.

        Args:
            A:  PETSc matrix
                Linear system matrix
            comm: MPI communicator

        Returns:
            ksp: PETSC KSP solver object
    """

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("superlu_dist")
    
    return ksp

def create_preconditioner(Q, bilin_form, bcs):
    """ Creates a preconditioner for the Stokes-type system.
     
        Args:
            Q:  DOLFINx FunctionSpace
                Function space for pressure
            bilin_form: UFL Form
                        The bilinear part of the variational form
            bcs: DOLFINx representation of the boundary conditions
        Returns:
            P:  PETSC matrix
                The preconditioner
    """
    p, p_test = TrialFunction(Q), TestFunction(Q)
    a_p11 = fem.form(inner(p, p_test) * dx)
    a_p = fem.form([[bilin_form[0][0], None], [None, a_p11]])
    P = dolfinx.fem.petsc.assemble_matrix_nest(a_p, bcs)
    P.assemble()
    return P

def create_nullspace(lin_form):
    """ Creates a nullspace for Stokes problems where only slip boundary
        conditions are specified.

        For Stokes problems where slip boundary conditions are supplied
        the pressure is only determined up to a constant, i.e., there is
        a nullspace to the system matrix.
        It this case, we can supply a vector that spans the nullspace
        to the solver, and any component of the solution in this direction
        will be eliminated during the solution process.
     
        Args:
            lin_form: UFL Form
                      The linear part of the variational form

        Returns:
            nsp: PETSC nullspace object
    """
    null_vec = fem.petsc.create_vector_nest(lin_form)
    null_vecs = null_vec.getNestSubVecs()

    # Set velocity component (1st) to zero and the pressure component (2nd) to a non-zero
    # constant.
    null_vecs[0].set(0.0) # First component is velocity
    null_vecs[1].set(1.0) # second component is pressure.
    # Normalize the vector that spans the nullspace, create a nullspace
    # object
    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    return nsp

def assemble_system_matrix(bilin_form, bcs, iterative):
    """ Assembles the Stokes-type problem system matrix
     
        Args:
            bilin_form: UFL Form
                      Left-hand-side of the variational form
            bcs:   DOLFINx representation of the boundary conditions
            iterative:  Bool
                        True: Assembles a nested matrix
                        False: Assembles a block matrix

        Returns:
            A:  PETSC matrix
                The system matrix
    """
    if iterative:
        A = dolfinx.fem.petsc.assemble_matrix_nest(bilin_form, bcs=bcs)
    else:
        A = dolfinx.fem.petsc.assemble_matrix_block(bilin_form, bcs=bcs)
    A.assemble()
    return A
    
def assemble_rhs_vector(bilin_form, lin_form, bcs, iterative):
    """ Assembles the right-hand-side vector of the Stokes-type problem
     
        Args:
            bilin_form: UFL Form
                      Left-hand-side of the variational form
            lin_form: UFL Form
                      Right-hand-side of the variational form
            bcs:   DOLFINx representation of the boundary conditions
            iterative:  Bool
                        True: Assembles a nested vector
                        False: Assembles a block vector

        Returns:
            A:  PETSC Vector
                The right-hand-side vector
    """
    if iterative:
        b = dolfinx.fem.petsc.assemble_vector_nest(lin_form)
        dolfinx.fem.petsc.apply_lifting_nest(b, bilin_form, bcs=bcs)
        for b_sub in b.getNestSubVecs():
            b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        spaces = fem.extract_function_spaces(lin_form)
        bcs0 = fem.bcs_by_block(spaces, bcs)
        dolfinx.fem.petsc.set_bc_nest(b, bcs0)
    else:
        b = dolfinx.fem.petsc.assemble_vector_block(lin_form, bilin_form, bcs = bcs)
    return b