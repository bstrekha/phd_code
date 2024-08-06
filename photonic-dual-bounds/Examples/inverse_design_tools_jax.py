import numpy as np 
import jax.numpy as jnp
from jax import jit, lax
from functools import partial

def epsr_vac_gen(size):
    return np.ones(size)

@jit
def operator_proj(rho, eta=0.5, beta=100):
    return jnp.true_divide(jnp.tanh(beta * eta) + jnp.tanh(beta * (rho - eta)),
                            jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta)))

@jit
def epsr_parameterization(rho, eps, epsbkg, design_mask, beta=0, eta=0.5, N_proj=1):
    partial_proj = lambda r : operator_proj(r, eta, beta)
    identity = lambda r : r # This should be improved 
    rho_proj = lax.cond(beta, partial_proj, identity, rho)
    epsr = eps + (epsbkg-eps) * rho_proj
    epsr = epsr*design_mask + epsbkg*jnp.logical_not(design_mask)
    return epsr

@jit
def chi_parametrization(rho, chi, chibkg, design_mask, beta=0, eta=0.5):
    return epsr_parameterization(rho, chi, chibkg, design_mask, beta, eta)

######################## Objectives ########################

@jit 
def absorbance(field, omega, epsr, dl, norm):
    return jnp.real(dl*dl*(omega/2)*jnp.real(jnp.sum(jnp.conjugate(field) * (1) * jnp.imag(epsr) *field)) / norm)

@jit
def absorbance_2field(field1, field2, omega, epsr, dl, norm):
    # If your convention is chi = a+bi, then keep the (1) as is 
    # If your convention is chi = a-bi, then change the (1) to (-1)
    a1 = dl*dl*(omega/2)*jnp.real(jnp.sum(jnp.conjugate(field1) * (1) * jnp.imag(epsr) *field1)) / norm 
    a2 = dl*dl*(omega/2)*jnp.real(jnp.sum(jnp.conjugate(field2) * (1) * jnp.imag(epsr) *field2)) / norm
    return jnp.real(a1+a2)