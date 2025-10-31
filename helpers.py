"""
A collection of helper functions.
"""
__author__ = "Vidar Stiernström"
__mail__ = "vidarsti@kth.se"
__copyright__ = "Copyright (c) 2025 {}".format(__author__)

from mpi4py import MPI
from dolfinx import fem

from ufl import (
    inner,
    sqrt,
    sym,
    nabla_grad
)

## Constitutive relations
def strain_rate(v):
    """ Computes the strain rate from the velocity v
     
        Args:
            v:  DOLFINx Function defined on vector-element function space
                The velocity vector

        Returns:
            UFL expression for the strain rate
    """
    return 2 * deformation_rate(v)

def deformation_rate(v):
    """ Computes the defomration rate from the velocity v
     
        Args:
            v:  DOLFINx Function defined on vector-element function space
                The velocity vector

        Returns:
            UFL expression for the deformation rate
    """
    return sym(nabla_grad(v))

## Vector and tensor operations
def tensor_norm(T):
    """ Computes the norm (second invariant) of a two-tensor T
     
        Args:
            T:  DOLFINx Function defined on tensor-element function space

        Returns:
            UFL expression for the norm of T
    """
    return sqrt(0.5 * inner(T,T))

def vector_norm(v):
    """ Computes the 2-norm of a vector v
     
        Args:
            v: DOLFINx Function defined on vector-element function space

        Returns:
            UFL expression for 2-norm of v
    """
    return sqrt(inner(v,v))

## MPI
def assemble_scalar(J, comm: MPI.Comm):
    """ Assembles a scalar from the scalar/functional J across
        an MPI communicator group.
      
        Args:
            J: UFL functional or scalar expression
            com: MPI communicator

        Returns:
            The MPI reduced scalar
    """
    scalar_form = fem.form(J)
    local_J = fem.assemble_scalar(scalar_form)
    return comm.allreduce(local_J, op=MPI.SUM)

## Misc
def validate_args(arg, valid_args):
    """ Assert that arg is within the specified valid arguments.

        Args:
            arg: argument to check
            valid_args: iterable containing valid arguments
    """
    assert any(arg == x for x in valid_args), "Valid arguments are " + str([x for x in valid_args]) + "."
