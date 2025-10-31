"""
A collection of variational forms for augmented Lagrange formulation
of the yield-stress fluid flow problem.
"""

__author__ = "Vidar Stiernström"
__mail__ = "vidarsti@kth.se"
__copyright__ = "Copyright (c) 2025 {}".format(__author__)

from dolfinx import fem, default_scalar_type

from ufl import (
    TestFunction,
    TrialFunction,
    dx,
    nabla_div,
    nabla_grad,
    inner,
)

from helpers import strain_rate, validate_args


def create_bilinear_form(V, Q, r, eta_p, var_form_str, regularize, C = 1e-10):
    """ Create a bilinear form for the Stokes-type flow problem.
     
        Args:
            V:  DOLFINx FunctionSpace
                The function space for velocity (vector-valued)
            Q:  DOLFINx FunctionSpace
                The function space for pressure (scalar-valued)
            r:  DOLFINx Constant
                The augmentation parameter
            eta_p:  DOLFINx Constant
                    The plastic viscosity
            var_form_str: String
                          Specifies the variational form to be used.
                          Either 'CT19-alg1', 'CT19-alg3', 'freeFEM', 'RS03'
                          See the deferred methods for further details.
            regularise: Bool
                        True: Regularizes the zero-block through a penalty term on pressure
                              enforcing mean zero pressure (useful when only slip bc are
                              specified). See https://doi.org/10.1016/j.cma.2005.04.004.
            C:  float 
                Constant used in penalty
        Returns:
            DOLFINx Form
            The bilinear form
    """
    validate_args(var_form_str, ('CT19-alg1','CT19-alg3','freeFEM','RS03'))
    if var_form_str == 'CT19-alg1':
        a_uu, a_up, a_pu =  bilinear_form_CT19_alg1(V, Q, r, eta_p)
    elif var_form_str == 'CT19-alg3':
        a_uu, a_up, a_pu =  bilinear_form_CT19_alg3(V, Q, r, eta_p)
    elif var_form_str == 'freeFEM':
       a_uu, a_up, a_pu =  bilinear_form_freeFEM(V, Q, r, eta_p)
    elif var_form_str == 'RS03':
       a_uu, a_up, a_pu =  bilinear_form_RS03(V, Q, r)

    if regularize:
        # Enforce mean zero pressure through a penalty term, as done in
        # https://doi.org/10.1016/j.cma.2005.04.004
        # TBD: Enforcing zero mean pressure through a scalar Lagrange multiplier might be perferable.
        msh = V.mesh
        p, p_test = TrialFunction(Q), TestFunction(Q)
        a_pp = -fem.Constant(msh, default_scalar_type(C)) * inner(p, fem.Constant(msh, 1.0)) * inner(fem.Constant(msh, 1.0), p_test) * dx
    else:
        a_pp = None

    return fem.form([[a_uu, a_up], [a_pu, a_pp]])

def create_linear_form(V, Q, T_n, q_n, r, eta_p, var_form_str):
    """ Create a linear form for the Stokes-type flow problem.
     
        Args:
            V:  DOLFINx FunctionSpace
                The function space for velocity (vector-valued)
            Q:  DOLFINx FunctionSpace
                The function space for pressure (scalar-valued)
            T_n:  DOLFINx Function
                  Stress tensor
            q_n:  DOLFINx Function
                  Relaxed strain-rate tensor
            r:  DOLFINx Constant
                The augmentation parameter
            eta_p:  DOLFINx Constant
                    The plastic viscosity
            var_form_str: String
                          Specifies the variational form to be used.
                          Either 'CT19-alg1', 'CT19-alg3', 'freeFEM', 'RS03'
                          See the deferred methods for further details.
        Returns:
            DOLFINx Form
            The linear form
    """
    validate_args(var_form_str, ('CT19-alg1','CT19-alg3','freeFEM','RS03'))
    if var_form_str == 'CT19-alg1':
        L_v = linear_form_CT19_alg1(V, T_n, q_n, r, eta_p)
    elif var_form_str == 'CT19-alg3':
        L_v = linear_form_CT19_alg3(V, T_n, q_n, r, eta_p)
    elif var_form_str == 'freeFEM':
        L_v = linear_form_freeFEM(V, T_n, q_n, r, eta_p)
    elif var_form_str == 'RS03':
        L_v = linear_form_RS03(V, T_n, q_n, r)

    p_test = TestFunction(Q)
    msh = V.mesh
    L_q = inner(fem.Constant(msh, default_scalar_type(0)), p_test) * dx
    return fem.form([L_v, L_q])

def bilinear_form_CT19_alg1(V, Q, r, eta_p):
    """ The components of the bilinear form of the Stokes-type flow problem
        in Algorithm 1. of Chaparian, Tammisola 2019,
        https://doi.org/10.1016/j.jnnfm.2019.104148

        Args:
            V:  DOLFINx FunctionSpace
                The function space for velocity (vector-valued)
            Q:  DOLFINx FunctionSpace
                The function space for pressure (scalar-valued)
            r:  DOLFINx Constant
                The augmentation parameter
            eta_p:  DOLFINx Constant
                    The plastic viscosity

        Returns:
            a_uu: UFL expression
                  Velocity-velocity component
            a_up: UFL expression
                  Velocity-pressure component
            a_pu: UFL expression
                  Pressure-velocity component
    """
    u, p = TrialFunction(V), TrialFunction(Q)
    u_test, p_test = TestFunction(V), TestFunction(Q)
    a_uu = r * eta_p * inner( nabla_grad(u),  nabla_grad(u_test)) * dx
    a_up =  -inner(p, nabla_div(u_test)) * dx
    a_pu =  -inner(nabla_div(u), p_test) * dx
    return a_uu, a_up, a_pu

def linear_form_CT19_alg1(V, T_n, q_n, r, eta_p):
    """ The velocity component of linear form of the Stokes-type flow problem
        in Algorithm 1. of Chaparian, Tammisola 2019,
        https://doi.org/10.1016/j.jnnfm.2019.104148

        Args:
            V:  DOLFINx FunctionSpace
                The function space for velocity (vector-valued)
            T_n:  DOLFINx Function
                  Stress tensor
            q_n:  DOLFINx Function
                  Relaxed strain-rate tensor
            r:  DOLFINx Constant
                The augmentation parameter
            eta_p:  DOLFINx Constant
                    The plastic viscosity

        Returns:
            L_v: UFL expression
                  Velocity component of linear form
    """

    u_test = TestFunction(V)
    L_v = -inner(T_n - r * eta_p * q_n, nabla_grad(u_test))*dx
    return L_v

def bilinear_form_CT19_alg3(V, Q, r, eta_p):
    """ The components of the bilinear form of the Stokes-type flow problem
        in Algorithm 3. of Chaparian, Tammisola 2019,
        https://doi.org/10.1016/j.jnnfm.2019.104148.
        
        Args:
            V:  DOLFINx FunctionSpace
                The function space for velocity (vector-valued)
            Q:  DOLFINx FunctionSpace
                The function space for pressure (scalar-valued)
            r:  DOLFINx Constant
                The augmentation parameter
            eta_p:  DOLFINx Constant
                    The plastic viscosity

        Returns:
            a_uu: UFL expression
                  Velocity-velocity component
            a_up: UFL expression
                  Velocity-pressure component
            a_pu: UFL expression
                  Pressure-velocity component
    """
    u, p = TrialFunction(V), TrialFunction(Q)
    u_test, p_test = TestFunction(V), TestFunction(Q)
    a_uu = r * eta_p * inner(strain_rate(u),  strain_rate(u_test)) * dx
    a_up =  -inner(p, nabla_div(u_test)) * dx
    a_pu =  -inner(nabla_div(u), p_test) * dx
    return a_uu, a_up, a_pu

def linear_form_CT19_alg3(V, T_n, q_n, r, eta_p):
    """ The velocity component of linear form of the Stokes-type flow problem
        in Algorithm 3. of Chaparian, Tammisola 2019,
        https://doi.org/10.1016/j.jnnfm.2019.104148.

        Args:
            V:  DOLFINx FunctionSpace
                The function space for velocity (vector-valued)
            T_n:  DOLFINx Function
                  Stress tensor
            q_n:  DOLFINx Function
                  Relaxed strain-rate tensor
            r:  DOLFINx Constant
                The augmentation parameter
            eta_p:  DOLFINx Constant
                    The plastic viscosity

        Returns:
            L_v: UFL expression
                  Velocity component of linear form
    """

    u_test = TestFunction(V)
    L_v = -inner(T_n - r * eta_p * q_n, strain_rate(u_test)) * dx
    return L_v

def bilinear_form_freeFEM(V, Q, r, eta_p):
    """ The components of the bilinear form of the Stokes-type flow problem
        based on a reference FreeFEM implementation of the Bingham
        particle problem.

        Args:
            V:  DOLFINx FunctionSpace
                The function space for velocity (vector-valued)
            Q:  DOLFINx FunctionSpace
                The function space for pressure (scalar-valued)
            r:  DOLFINx Constant
                The augmentation parameter
            eta_p:  DOLFINx Constant
                    The plastic viscosity

        Returns:
            a_uu: UFL expression
                  Velocity-velocity component
            a_up: UFL expression
                  Velocity-pressure component
            a_pu: UFL expression
                  Pressure-velocity component
    """
    u, p = TrialFunction(V), TrialFunction(Q)
    u_test, p_test = TestFunction(V), TestFunction(Q)
    
    a_uu = r * eta_p * inner( strain_rate(u),  nabla_grad(u_test) ) * dx
    a_up =  -inner(p, nabla_div(u_test)) * dx
    a_pu =  -inner(nabla_div(u), p_test) * dx
    return a_uu, a_up, a_pu

def linear_form_freeFEM(V, T_n, q_n, r, eta_p):
    """ The velocity component of linear form of the Stokes-type flow problem
        based on a reference FreeFEM implementation of the Bingham
        particle problem.

        Args:
            V:  DOLFINx FunctionSpace
                The function space for velocity (vector-valued)
            T_n:  DOLFINx Function
                  Stress tensor
            q_n:  DOLFINx Function
                  Relaxed strain-rate tensor
            r:  DOLFINx Constant
                The augmentation parameter
            eta_p:  DOLFINx Constant
                    The plastic viscosity

        Returns:
            L_v: UFL expression
                  Velocity component of linear form
    """
    u_test = TestFunction(V)
    L_v = -inner(T_n - r * eta_p * q_n, nabla_grad(u_test))*dx
    return L_v

def bilinear_form_RS03(V, Q, r):
    """ The components of the bilinear form of the Stokes-type flow problem
        in Roquet, Samarito 2003, https://doi.org/10.1016/S0045-7825(03)00262-7

        Args:
            V:  DOLFINx FunctionSpace
                The function space for velocity (vector-valued)
            Q:  DOLFINx FunctionSpace
                The function space for pressure (scalar-valued)
            r:  DOLFINx Constant
                The augmentation parameter

        Returns:
            a_uu: UFL expression
                  Velocity-velocity component
            a_up: UFL expression
                  Velocity-pressure component
            a_pu: UFL expression
                  Pressure-velocity component
    """
    u, p = TrialFunction(V), TrialFunction(Q)
    u_test, p_test = TestFunction(V), TestFunction(Q)
    # Note: Factor 2 difference in a_uu compared to 
    # the equation given in Roquet Samarito 2003
    # Would not converge otherwise.
    a_uu = 2 * r * inner(nabla_grad(u),  nabla_grad(u_test)) * dx
    a_up =  -inner(p, nabla_div(u_test)) * dx
    a_pu =  -inner(nabla_div(u), p_test) * dx
    return a_uu, a_up, a_pu

def linear_form_RS03(V, T_n, q_n, r):
    """ The velocity component of linear form of the Stokes-type flow problem
        in Roquet, Samarito 2003, https://doi.org/10.1016/S0045-7825(03)00262-7

        Args:
            V:  DOLFINx FunctionSpace
                The function space for velocity (vector-valued)
            T_n:  DOLFINx Function
                  Stress tensor
            q_n:  DOLFINx Function
                  Relaxed strain-rate tensor
            r:  DOLFINx Constant
                The augmentation parameter

        Returns:
            L_v: UFL expression
                  Velocity component of linear form
    """
        
    u_test = TestFunction(V)
    L_v = -inner(T_n - 2 * r * q_n, nabla_grad(u_test))*dx
    return L_v
