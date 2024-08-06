import numpy as np 
import jax.numpy as jnp
from jax import jit, lax
from functools import partial

def epsr_vac_gen(size):
    return np.ones(size)

def operator_proj(rho, eta=0.5, beta=100):
    if beta == 0:
        return rho
    return np.divide(np.tanh(beta * eta) + np.tanh(beta * (rho - eta)),
                            np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))

def epsr_parameterization(rho, eps, epsbkg, design_mask, beta=0, eta=0.5, N_proj=1):
    rho_proj = operator_proj(rho, beta=beta)
    epsr = eps + (epsbkg-eps) * rho_proj
    epsr = epsr*design_mask + epsbkg*np.logical_not(design_mask)
    return epsr

def chi_parametrization(rho, chi, chibkg, design_mask, beta=0, eta=0.5):
    return epsr_parameterization(rho, chi, chibkg, design_mask, beta, eta)


######################## Objectives ########################

# def absorbance_2field(field1, field2, omega, epsr, dl, norm):
#     # The extra negative sign is due to the sign convention of Im(epsr), don't remove it! 
#     a1 = -dl*dl*(omega/2)*np.real(np.sum(np.conjugate(field1) * (-1) * np.imag(epsr) *field1)) / norm 
#     a2 = -dl*dl*(omega/2)*np.real(np.sum(np.conjugate(field2) * (-1) * np.imag(epsr) *field2)) / norm
#     return np.real(a1+a2)

# @jit 
def absorbance(field, omega, epsr, dl, norm):
    return np.real((omega/2)*np.real(np.sum(np.conjugate(field) * (1) * np.imag(epsr) *field)) / norm)


@jit
def absorbance_2field(field1, field2, omega, epsr, dl, norm):
    # If your convention is chi = a+bi, then keep the (1) as is 
    # If your convention is chi = a-bi, then change the (1) to (-1)
    a1 = dl*dl*(omega/2)*jnp.real(jnp.sum(jnp.conjugate(field1) * (1) * jnp.imag(epsr) *field1)) / norm 
    a2 = dl*dl*(omega/2)*jnp.real(jnp.sum(jnp.conjugate(field2) * (1) * jnp.imag(epsr) *field2)) / norm
    return jnp.real(a1+a2)