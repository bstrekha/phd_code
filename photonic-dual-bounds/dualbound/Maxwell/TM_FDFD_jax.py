#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:32:29 2021

@author: alessio

TODO(alessio): Rename functions with _ for private 
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from functools import partial

import jax.numpy as jnp
import jax.experimental.sparse as jsp
import jax 
#jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_enable_x64", True)
from jax import jit, vmap, grad, lax
import sys 
sys.path.append('../../')
import dualbound.Maxwell.jaxtools as jt

@partial(jit, static_argnums=(1,2))
def _make_Dxc(dL, Nx, Ny, bloch_x=0.0):
    """ center differences derivative in x """
    fx_phasor = jnp.exp(1j*bloch_x)
    bx_phasor = jnp.exp(-1j*bloch_x)
    Dxc = (jt.diag_m1(jnp.repeat(-1, Nx-1), -bx_phasor) + jt.diag_p1(jnp.repeat(1, Nx-1), fx_phasor)).astype(complex)
    Dxc = (1/2/dL) * jt.kron(Dxc, jsp.eye(Ny))
    return Dxc

@partial(jit, static_argnums=(1,2))
def _make_Dxc_dense(dL, Nx, Ny, bloch_x=0.0):
    """ center differences derivative in x """
    fx_phasor = jnp.exp(1j*bloch_x)
    bx_phasor = jnp.exp(-1j*bloch_x)
    Dxc = (jnp.diag(jnp.repeat(-1, Nx-1), -1) + jnp.diag(jnp.repeat(1, Nx-1), 1)).astype(complex)
    Dxc = Dxc.at[Nx-1, 0].add(fx_phasor)
    Dxc = Dxc.at[0, Nx-1].add(-bx_phasor)
    Dxc = (1/2/dL) * jnp.kron(Dxc, jnp.eye(Ny))
    return Dxc

@partial(jit, static_argnums=(1,2))
def _make_Dxf(dL, Nx, Ny, bloch_x=0.0):
    """ Forward derivative in x """
    phasor_x = jnp.exp(1j * bloch_x)
    Dxf = (jt.diag(jnp.repeat(-1, Nx)) + jt.diag_p1(jnp.repeat(1, Nx-1), phasor_x)).astype(complex)
    Dxf = (1/dL) * jt.kron(Dxf, jsp.eye(Ny))
    return Dxf

@partial(jit, static_argnums=(1,2))
def _make_Dxf_dense(dL, Nx, Ny, bloch_x=0.0):
    """ Forward derivative in x """
    phasor_x = jnp.exp(1j * bloch_x)
    Dxf = (jnp.diag(jnp.repeat(-1, Nx), 0) + jnp.diag(jnp.repeat(1, Nx-1), 1)).astype(complex)
    Dxf = Dxf.at[Nx-1, 0].add(phasor_x)
    Dxf = (1/dL) * jnp.kron(Dxf, jnp.eye(Ny))
    return Dxf

@partial(jit, static_argnums=(1,2))
def _make_Dxb(dL, Nx, Ny, bloch_x=0.0):
    """ Backward derivative in x """
    phasor_x = jnp.exp(1j * bloch_x)
    Dxb = (jt.diag(jnp.repeat(1, Nx)) + jt.diag_m1(jnp.repeat(-1, Nx-1), -jnp.conj(phasor_x))).astype(complex)
    Dxb = (1/dL) * jt.kron(Dxb, jsp.eye(Ny))
    return Dxb

@partial(jit, static_argnums=(1,2))
def _make_Dxb_dense(dL, Nx, Ny, bloch_x=0.0):
    """ Backward derivative in x """
    phasor_x = jnp.exp(1j * bloch_x)
    Dxb = (jnp.diag(jnp.repeat(1, Nx), 0) + jnp.diag(jnp.repeat(-1, Nx-1), -1)).astype(complex)
    Dxb = Dxb.at[0, Nx-1].add(-jnp.conj(phasor_x))
    Dxb = (1/dL) * jnp.kron(Dxb, jnp.eye(Ny))
    return Dxb

@partial(jit, static_argnums=(1,2))
def _make_Dyc(dL, Nx, Ny, bloch_y=0.0):
    """ center differences derivative in y """
    fy_phasor = jnp.exp(1j*bloch_y)
    by_phasor = jnp.exp(-1j*bloch_y)
    Dyc = jnp.zeros((Ny, Ny), dtype=complex)
    Dyc = (jt.diag_m1(jnp.repeat(-1, Ny-1), -by_phasor) + jt.diag_p1(jnp.repeat(1, Ny-1), fy_phasor)).astype(complex)
    Dyc = (1/2/dL) * jt.kron(jsp.eye(Nx), Dyc)

    return Dyc

@partial(jit, static_argnums=(1,2))
def _make_Dyc_dense(dL, Nx, Ny, bloch_y=0.0):
    """ center differences derivative in y """
    fy_phasor = jnp.exp(1j*bloch_y)
    by_phasor = jnp.exp(-1j*bloch_y)
    Dyc = (jnp.diag(jnp.repeat(-1, Ny-1), -1) + jnp.diag(jnp.repeat(1, Ny-1), 1)).astype(complex)
    Dyc = Dyc.at[Ny-1, 0].add(fy_phasor)
    Dyc = Dyc.at[0, Ny-1].add(-by_phasor)
    Dyc = (1/2/dL) * jnp.kron(jnp.eye(Nx), Dyc)
    return Dyc

@partial(jit, static_argnums=(1,2))
def _make_Dyf(dL, Nx, Ny, bloch_y=0.0):
    """ Forward derivative in y """
    phasor_y = jnp.exp(1j * bloch_y)
    Dyf = (jt.diag(jnp.repeat(-1, Ny)) + jt.diag_p1(jnp.repeat(1, Ny-1), phasor_y)).astype(complex)
    Dyf = (1/dL) * jt.kron(jsp.eye(Nx), Dyf)
    return Dyf

@partial(jit, static_argnums=(1,2))
def _make_Dyf_dense(dL, Nx, Ny, bloch_y=0.0):
    """ Forward derivative in y """
    phasor_y = jnp.exp(1j * bloch_y)
    Dyf = (jnp.diag(jnp.repeat(-1, Ny), 0) + jnp.diag(jnp.repeat(1, Ny-1), 1)).astype(complex)
    Dyf = Dyf.at[Ny-1, 0].add(phasor_y)
    Dyf = (1/dL) * jnp.kron(jnp.eye(Nx), Dyf)
    return Dyf

@partial(jit, static_argnums=(1,2))
def _make_Dyb(dL, Nx, Ny, bloch_y=0.0):
    """ Backward derivative in y """
    phasor_y = jnp.exp(1j * bloch_y)
    Dyb = (jt.diag(jnp.repeat(1, Ny)) + jt.diag_m1(jnp.repeat(-1, Ny-1), -jnp.conj(phasor_y))).astype(complex)
    Dyb = (1/dL) * jt.kron(jsp.eye(Nx), Dyb)
    return Dyb

@partial(jit, static_argnums=(1,2))
def _make_Dyb_dense(dL, Nx, Ny, bloch_y=0.0):
    """ Backward derivative in y """
    phasor_y = jnp.exp(1j * bloch_y)
    Dyb = jnp.zeros((Ny, Ny), dtype=complex)
    Dyb = (jnp.diag(jnp.repeat(1, Ny), 0) + jnp.diag(jnp.repeat(-1, Ny-1), -1)).astype(complex)
    Dyb = Dyb.at[0, Ny-1].add(-jnp.conj(phasor_y))
    Dyb = (1/dL) * jnp.kron(jnp.eye(Nx), Dyb)
    return Dyb

C_0 = 1.0 #dimensionless units
EPSILON_0 = 1.0
ETA_0 = 1.0

#########################PML###################################################
@partial(jit, static_argnums=(1,2,3,4))
def _create_S_matrices(omega, Nx, Ny, npmlx, npmly, dL):
    """ Makes the 'S-matrices'.  When dotted with derivative matrices, they add PML """

    # strip out some information needed
    shape = (Nx, Ny)

    # Create the sfactor in each direction and for 'f' and 'b'
    dw_x = npmlx*dL 
    dw_y = npmly*dL
    s_vector_x_f = lax.cond(npmlx > 0, partial(_create_sfactor_f, Nx), partial(_get_ones_wrapper, Nx), omega, dL, npmlx, dw_x)
    s_vector_x_b = lax.cond(npmlx > 0, partial(_create_sfactor_b, Nx), partial(_get_ones_wrapper, Nx), omega, dL, npmlx, dw_x)
    s_vector_y_f = lax.cond(npmly > 0, partial(_create_sfactor_f, Ny), partial(_get_ones_wrapper, Ny), omega, dL, npmly, dw_y)
    s_vector_y_b = lax.cond(npmly > 0, partial(_create_sfactor_b, Ny), partial(_get_ones_wrapper, Ny), omega, dL, npmly, dw_y)

    # return s_vector_x_f, s_vector_x_b, s_vector_y_f, s_vector_y_b

    # Instead of doing 2D arrays and unrolling them (see numpy code), just make the 1D array from the beginning 
    jSx_f_vec = jnp.repeat(1/s_vector_x_f, Ny)
    jSx_b_vec = jnp.repeat(1/s_vector_x_b, Ny)
    jSy_f_vec = jnp.tile(1/s_vector_y_f, Nx)
    jSy_b_vec = jnp.tile(1/s_vector_y_b, Nx)

    # Construct the 1D total s-vecay into a diagonal matrix
    indices_xf = jnp.stack((jnp.arange(Nx*s_vector_x_f.shape[0]), jnp.arange(Nx*s_vector_x_f.shape[0])), axis=1)
    indices_xb = jnp.stack((jnp.arange(Nx*s_vector_x_b.shape[0]), jnp.arange(Nx*s_vector_x_b.shape[0])), axis=1)
    indices_yf = jnp.stack((jnp.arange(Ny*s_vector_y_f.shape[0]), jnp.arange(Ny*s_vector_y_f.shape[0])), axis=1)
    indices_yb = jnp.stack((jnp.arange(Ny*s_vector_y_b.shape[0]), jnp.arange(Ny*s_vector_y_b.shape[0])), axis=1)

    Sx_f = jsp.BCOO((jSx_f_vec, indices_xf), shape=(Nx*Ny, Nx*Ny))
    Sx_b = jsp.BCOO((jSx_b_vec, indices_xb), shape=(Nx*Ny, Nx*Ny))
    Sy_f = jsp.BCOO((jSy_f_vec, indices_yf), shape=(Nx*Ny, Nx*Ny))
    Sy_b = jsp.BCOO((jSy_b_vec, indices_yb), shape=(Nx*Ny, Nx*Ny))

    # Sx_f = jnp.diag(jSx_f_vec, 0)   
    # Sx_b = jnp.diag(jSx_b_vec, 0)   
    # Sy_f = jnp.diag(jSy_f_vec, 0)   
    # Sy_b = jnp.diag(jSy_b_vec, 0) 

    return Sx_f, Sx_b, Sy_f, Sy_b

@partial(jit, static_argnums=(0))
def _get_ones_wrapper(N, omega, dL, N_pml, dw):
    return jnp.ones(N, dtype=complex)

# These two functions could be condensed into one, just changing the 1 to a 0.5
@partial(jit, static_argnums=(0))
def _create_sfactor_f(N, omega, dL, N_pml, dw):
    """ S-factor profile for forward derivative matrix """
    # Define the functions to be applied at variable index i 
    f1 = lambda i : _s_value(dL * (N_pml - i + 0.5), dw, omega)
    f2 = lambda i : 1+0j
    f3 = lambda i : _s_value(dL * (i - (N - N_pml) - 0.5), dw, omega)

    li = jnp.arange(N) # loop indices

    # Define the three conditions 
    cond1 = li <= N_pml
    cond2 = jnp.logical_and(li > N_pml, li <= N - N_pml)
    cond3 = jnp.logical_and(li > N - N_pml, jnp.logical_not(cond1))

    # Select where each condition should be applied 
    func_indices = jnp.argwhere(jnp.array([cond1, cond2, cond3]), size=N).squeeze()[:, 0]

    # Define a switch that applies the correct function to i based on func_indices 
    switch = lambda i : lax.switch(func_indices[i], [f1, f2, f3], i)

    return vmap(switch)(li) # vectorize the switch function and apply it to the input indices

@partial(jit, static_argnums=(0))
def _create_sfactor_b(N, omega, dL, N_pml, dw):
    """ S-factor profile for backward derivative matrix """
    li = jnp.arange(N) # loop indices

    # Define the three conditions 
    cond1 = li <= N_pml
    cond2 = jnp.logical_and(li > N_pml, li <= N - N_pml)
    cond3 = jnp.logical_and(li > N - N_pml, jnp.logical_not(cond1))

    # Select where each condition should be applied 
    func_indices = jnp.argwhere(jnp.array([cond1, cond2, cond3]), size=N).squeeze()[:, 0]

    # Define the functions to be applied at variable index i 
    f1 = lambda i : _s_value(dL * (N_pml - i + 1), dw, omega)
    f2 = lambda i : 1+0j
    f3 = lambda i : _s_value(dL * (i - (N - N_pml) - 1), dw, omega)

    # Define a switch that applies the correct function to i based on func_indices 
    switch = lambda i : lax.switch(func_indices[i], [f1, f2, f3], i)

    return vmap(switch)(li) # vectorize the switch function and apply it to the input indices
    
@jit
def _sig_w(l, dw, m=3, lnR=-30):
    """ Fictional conductivity, note that these values might need tuning """
    sig_max = -(m + 1) * lnR / (2 * ETA_0 * dw)
    return sig_max * (l / dw)**m

@jit
def _s_value(l, dw, omega):
    """ S-value to use in the S-matrices """
    return 1 + 1j * _sig_w(l, dw) / (omega * EPSILON_0)
    
@partial(jit, static_argnums=(2,3,4,5))
def _get_TM_MaxwellOp(wvlgth, dL, Nx, Ny, Npmlx, Npmly, bloch_x=0.0, bloch_y=0.0):
    """
    uniform grid 2D scalar E field Maxwell operator
    Npml can be both an int or a tuple
    Parameters
    ----------
    omega : complex
        circular frequency, can be complex to allow for finite bandwidth effects
    dL : float
        finite difference grid pixel size, in units of 1 
    Nx : int
        number of pixels along the x direction.
    Ny : int
        number of pixels along the y direction.
    Npml : int or tuple
        number of pixels in the PML region (part of Nx and Ny).
        if Npml is int it will be promoted to tuple (Npml,Npml).
    bloch_x : float, optional
        x-direction phase shift associated with the periodic boundary condtions. The default is 0.0.
    bloch_y : float, optional
        y-direction phase shift associated with the periodic boundary condtions. The default is 0.0.

    Returns
    -------
    M : sparse complex matrix
        Maxwell operator in sparse matrix format.

    """
    # shape = (Nx,Ny)
    # Npml = (Npmlx, Npmly)
    
    # Necessary for now, since Dxf etc do not output sparse matrices
    Dxf = _make_Dxf_dense(dL, Nx, Ny, bloch_x=bloch_x)
    Dxb = _make_Dxb_dense(dL, Nx, Ny, bloch_x=bloch_x)
    Dyf = _make_Dyf_dense(dL, Nx, Ny, bloch_y=bloch_y)
    Dyb = _make_Dyb_dense(dL, Nx, Ny, bloch_y=bloch_y)

    Qabs = np.inf
    omega = (2*np.pi*C_0/wvlgth) * (1 + 1j/2/Qabs) 
    Sxf, Sxb, Syf, Syb = _create_S_matrices(omega, Nx, Ny, Npmlx, Npmly, dL)
    
    # dress the derivative functions with pml
    Dxf = (Sxf @ Dxf)
    Dxb = (Sxb @ Dxb)
    Dyf = (Syf @ Dyf)
    Dyb = (Syb @ Dyb)

    M = -Dxf @ Dxb - Dyf @ Dyb - EPSILON_0*omega**2 * jnp.eye(Nx*Ny)
    return M

@jit
def _get_diagM_from_chigrid_dense(omega, chigrid):
    return -jnp.diag(jnp.ravel(chigrid) * omega**2)

@jit
def _get_diagM_from_chigrid(omega, chigrid):
    return jt.diag(-jnp.ravel(chigrid) * omega**2)

@partial(jit, static_argnums=(2,3,6,7))
def get_TM_dipole_field_dense(wvlgth, dL, Nx, Ny, cx, cy, Npmlx, Npmly, bloch_x=0.0, bloch_y=0.0, amp=1.0, chigrid=None):
    """
    get the field of a TM dipole source at position (cx,cy) in a grid of size (Nx,Ny)
    and material distribution given by chigrid.
    
    Parameters
    ----------
    omega : complex
        wavelength of interest.
    dL : float
        size of a single pixel of the finite difference grid.
    Nx : int
        Number of pixels along the x direciton.
    Ny : int
        Number of pixels along the y direction.
    cx : int
        x-coordinate of the dipole source.
    cy : int
        y-coordinate of the dipole source.
    Npml : int or tuple
        number of grid points in the PML.
    loch_x : float, optional
        x-direction phase shift associated with the periodic boundary condtions. The default is 0.0.
    bloch_y : float, optional
        y-direction phase shift associated with the periodic boundary condtions. The default is 0.0.
    Qabs : float, optional
        Q parameter specifying bandwidth of sources. The default is np.inf.
    chigrid : 2D numpy complex array, optional
        spatial distribution of material susceptibility. The default is None, corresponding to vacuum.

    Returns
    -------
    Ez : 2D numpy complex array
        Field of the dipole source.

    """
    shape = (Nx,Ny)

    M = _get_TM_MaxwellOp(wvlgth, dL, Nx, Ny, Npmlx, Npmly, bloch_x, bloch_y)

    Qabs = np.inf
    omega = (2*np.pi*C_0/wvlgth) * (1 + 1j/2/Qabs) 
    if not (chigrid is None):
        M += _get_diagM_from_chigrid_dense(omega, chigrid)
    
    sourcegrid = jnp.zeros((Nx,Ny), dtype=complex)
    sourcegrid = sourcegrid.at[cx, cy].set(amp / dL**2)
    RHS = jnp.ravel(1j*omega*sourcegrid)

    Ez = jnp.linalg.solve(M, RHS)
    Ez = jnp.reshape(Ez, shape)
    return Ez

# @partial(jit, static_argnums=(3, 4))
def get_TM_dipole_field(M0, wvlgth, dL, Nx, Ny, cx, cy, amp, chigrid=None):
    """
    get the field of a TM dipole source at position (cx,cy) in a grid of size (Nx,Ny)
    and material distribution given by chigrid.
    
    Parameters
    ----------
    M0 : jax.experimental.BCOO sparse matrix
        Vacuum Maxwell operator from get_TM_MaxwellOp as a sparse matrix. It must be pre-computed with either this code or TM_FDFD.py.
    wvlgth : real
        wavelength of interest.
    dL : float
        size of a single pixel of the finite difference grid.
    Nx : int
        Number of pixels along the x direciton.
    Ny : int
        Number of pixels along the y direction.
    cx : int
        x-coordinate of the dipole source.
    cy : int
        y-coordinate of the dipole source.
    chigrid : 2D numpy complex array, optional
        spatial distribution of material susceptibility. The default is None, corresponding to vacuum.

    Returns
    -------
    Ez : 2D numpy complex array
         Field of the dipole source.
    """
    Qabs = np.inf # Quality of mode, can be passed as a parameter but just not generalized yet
    omega = (2*np.pi*C_0/wvlgth) * (1 + 1j/2/Qabs) 
    if not chigrid is None:
        M = _get_diagM_from_chigrid(omega, chigrid)
        A = jsp.BCSR.from_bcoo(jsp.BCOO.sum_duplicates(M+M0, nse=M0.nse))
    else:
        A = jsp.BCSR.from_bcoo(M0)

    sourcegrid = jnp.zeros((Nx,Ny), dtype=complex)
    sourcegrid = sourcegrid.at[cx, cy].set(amp / dL**2)
    RHS = jnp.ravel(1j*omega*sourcegrid)
    
    tree, info = A.tree_flatten()
    data, indices, indptr = tree     
    Ez = jsp.linalg.spsolve(data, indices, indptr, RHS) 

    Ez = jnp.reshape(Ez, (Nx, Ny))
    return Ez

@partial(jit, static_argnums=(3, 4))
def get_TM_linesource_field(M0, wvlgth, dL, Nx, Ny, cx, amp, chigrid=None):
    Qabs = np.inf
    omega = (2*np.pi*C_0/wvlgth) * (1 + 1j/2/Qabs) 
    if not chigrid is None:
        M = _get_diagM_from_chigrid(omega, chigrid)
        A = jsp.BCSR.from_bcoo(jsp.BCOO.sum_duplicates(M+M0, nse=M0.nse))
    else:
        A = jsp.BCSR.from_bcoo(M0)
    
    sourcegrid = jnp.zeros((Nx,Ny), dtype=complex)
    sourcegrid = sourcegrid.at[cx, :].set(amp / dL)
    RHS = jnp.ravel(1j*omega*sourcegrid)
    
    tree, info = A.tree_flatten()
    data, indices, indptr = tree     
    Ez = jsp.linalg.spsolve(data, indices, indptr, RHS) 

    # Ez = jnp.linalg.solve(M.todense(), RHS)
    # Ez = spla.spsolve(sp.csr_matrix(M.todense()).astype(), RHS)
    Ez = jnp.reshape(Ez, (Nx, Ny))
    return Ez

def get_TM_field(M0, wvlgth, Nx, Ny, sourcegrid, chigrid):
    shape = (Nx,Ny)
    omega = (2*np.pi*C_0/wvlgth) * (1 + 1j/2/np.inf)    
    if not chigrid is None:
        M = _get_diagM_from_chigrid(omega, chigrid)
        A = jsp.BCSR.from_bcoo(jsp.BCOO.sum_duplicates(M+M0, nse=M0.nse))
    else:
        A = jsp.BCSR.from_bcoo(M0)

    RHS = 1j*omega*sourcegrid.flatten()
    
    tree, info = A.tree_flatten()
    data, indices, indptr = tree     
    Ez = jsp.linalg.spsolve(data, indices, indptr, RHS)
    Ez = jnp.reshape(Ez, (Nx, Ny))

    return Ez


if __name__ == "__main__":
    import TM_FDFD as TM
    import time 
    import sys 

    JIT_BEFORE = 0
    TEST_DERIVATIVES = 0
    TEST_S_MATRICES = 0
    TEST_TM_MAXWELLOP = 0
    TEST_TM_DIPOLE_FIELD = 0
    TEST_SPARSE = 1

    # print("Testing jax vs numpy. Each test will verify that the output of the jax functions are the same as the numpy functions.")
    Nx, Ny = 100, 100
    gpr = 50
    npml = 20
    dw = 50/200
    chigrid =  jnp.ones((Nx, Ny), dtype=complex)
    chigrid_np = np.ones((Nx, Ny), dtype=complex)

    if JIT_BEFORE:
        Dxc = _make_Dxc_dense(1/gpr, Nx, Ny, bloch_x=0)
        Dxf = _make_Dxf_dense(1/gpr, Nx, Ny, bloch_x=0)
        Dxb = _make_Dxb_dense(1/gpr, Nx, Ny, bloch_x=0)
        Dyc = _make_Dyc_dense(1/gpr, Nx, Ny, bloch_y=0)
        Dyf = _make_Dyf_dense(1/gpr, Nx, Ny, bloch_y=0)
        Dyb = _make_Dyb_dense(1/gpr, Nx, Ny, bloch_y=0)
        sfactor_b = _create_sfactor_b(Nx, 1, 1/gpr, npml, dw)
        sfactor_f = _create_sfactor_f(Nx, 1, 1/gpr, npml, dw)
        S_matrix_ega, S_matrix_egb, S_matrix_egc, S_matric_egd = _create_S_matrices(1, Nx, Ny, npml, npml, 1/gpr) # This should also compile s_value, sig_w, and create_sfactor functions
        M_op = _get_TM_MaxwellOp(1, 1/gpr, Nx, Ny, npml, npml, bloch_x=0, bloch_y=0)

    ### Lazy profiling of jax vs numpy derivative functions
    if TEST_DERIVATIVES:
        print("Testing Dx_ functions, jax vs numpy")
        # Run functions once to jit them 
        t1jnp = time.time()
        Dxc = _make_Dxc_dense(1/gpr, Nx, Ny, bloch_x=0)
        Dxf = _make_Dxf_dense(1/gpr, Nx, Ny, bloch_x=0)
        Dxb = _make_Dxb_dense(1/gpr, Nx, Ny, bloch_x=0)
        Dyc = _make_Dyc_dense(1/gpr, Nx, Ny, bloch_y=0)
        Dyf = _make_Dyf_dense(1/gpr, Nx, Ny, bloch_y=0)
        Dyb = _make_Dyb_dense(1/gpr, Nx, Ny, bloch_y=0)
        print(f'jax: {time.time() - t1jnp}')

        t1np = time.time()
        Dxc_np = TM.make_Dxc(1/gpr, (Nx, Ny), bloch_x=0)
        Dxf_np = TM.make_Dxf(1/gpr, (Nx, Ny), bloch_x=0)
        Dxb_np = TM.make_Dxb(1/gpr, (Nx, Ny), bloch_x=0)
        Dyc_np = TM.make_Dyc(1/gpr, (Nx, Ny), bloch_y=0)
        Dyf_np = TM.make_Dyf(1/gpr, (Nx, Ny), bloch_y=0)
        Dyb_np = TM.make_Dyb(1/gpr, (Nx, Ny), bloch_y=0)

        print(f' np: {time.time() - t1np}')
        if Nx <= 100:
            assert np.allclose(np.array(Dxc), Dxc_np.todense())
            assert np.allclose(np.array(Dxf), Dxf_np.todense())
            assert np.allclose(np.array(Dxb), Dxb_np.todense())
            assert np.allclose(np.array(Dyc), Dyc_np.todense())
            assert np.allclose(np.array(Dyf), Dyf_np.todense())
            assert np.allclose(np.array(Dyb), Dyb_np.todense())

            print("Passed! \n")

    if TEST_S_MATRICES:
        print("Testing create_sfactors")
        t1jnp = time.time()
        sfactor_b = _create_sfactor_b(Nx, 1, 1/gpr, npml, dw)
        sfactor_f = _create_sfactor_f(Nx, 1, 1/gpr, npml, dw)
        print(f"jax: {time.time() - t1jnp}")

        t1np = time.time()
        sfactor_b_np = TM.create_sfactor_b(1, 1/gpr, Nx, npml, dw)
        sfactor_f_np = TM.create_sfactor_f(1, 1/gpr, Nx, npml, dw)
        print(f"np: {time.time() - t1np}")
        print()

        assert np.allclose(sfactor_b, sfactor_b_np, atol=1e-8)
        assert np.allclose(sfactor_f, sfactor_f_np, atol=1e-8)

        print("Testing create_S_matrices function, jax vs numpy")
        t1jnp = time.time()
        S_matrix_ega, S_matrix_egb, S_matrix_egc, S_matric_egd = _create_S_matrices(1, Nx, Ny, npml, npml, 1/gpr)
        print(f'jax: {time.time() - t1jnp}')

        t1np = time.time()
        S_matrix_ega_np, S_matrix_egb_np, S_matrix_egc_np, S_matric_egd_np = TM.create_S_matrices(1, (Nx, Ny), (npml, npml), 1/gpr)
        print(f' np: {time.time() - t1np}')

        assert np.allclose(np.array(S_matrix_ega), S_matrix_ega_np.todense(), atol=1e-8)
        assert np.allclose(np.array(S_matrix_egb), S_matrix_egb_np.todense(), atol=1e-8)
        assert np.allclose(np.array(S_matrix_egc), S_matrix_egc_np.todense(), atol=1e-8)
        assert np.allclose(np.array(S_matric_egd), S_matric_egd_np.todense(), atol=1e-8)

        print("Passed! \n")

    if TEST_TM_MAXWELLOP:
        print("Testing get_TM_MaxwellOp function, jax vs numpy")
        t1jnp = time.time()
        M_op = _get_TM_MaxwellOp(1, 1/gpr, Nx, Ny, npml, npml, bloch_x=0, bloch_y=0)
        print(f'jax: {time.time() - t1jnp}')

        t1np = time.time()
        M_op_np = TM.get_TM_MaxwellOp(1, 1/gpr, Nx, Ny, (npml, npml), bloch_x=0, bloch_y=0)
        print(f' np: {time.time() - t1np}')

        assert np.allclose(M_op, M_op_np.todense())
        print("Passed! \n")

    if TEST_TM_DIPOLE_FIELD:
        print("Testing dipole field jax vs np")
        t1jnp = time.time()
        Ez = get_TM_dipole_field(1, 1/gpr, Nx, Ny, Nx//2, Ny//2, npml, npml)
        print(f'jax: {time.time() - t1jnp}')

        t1np = time.time()
        Ez_np = TM.get_TM_dipole_field(1, 1/gpr, Nx, Ny, Nx//2, Ny//2, (npml, npml))
        print(f' np: {time.time() - t1np}')

        for atol in [1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-12]:
            try:
                assert np.allclose(Ez, Ez_np, atol=atol)
                print(f"Passed with atol={atol}")
            except:
                print(f"Failed with atol={atol}")
                break

        d = jit(lambda chi: jnp.real(jnp.sum(get_TM_dipole_field(1, 1/gpr, Nx, Ny, Nx//2, Ny//2, npml, npml, chigrid=chi))))
        res = grad(d)(jnp.ones((Nx, Ny), dtype=complex))

        print("\nGradient calculation did not fail.")
        # print(res)

    if TEST_SPARSE:
        def comparison(x1, x2):
            for atol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-12]:
                try:
                    assert np.allclose(x1, x2, atol=atol)
                    print(f"Passed with atol={atol}")
                except:
                    print(f"Failed with atol={atol}")
                    break
        
        M0_np = TM.get_TM_MaxwellOp(1, 1/gpr, Nx, Ny, (npml, npml), bloch_x=0.0, bloch_y=0.0, Qabs=np.inf)
        M0 = jsp.BCOO.from_scipy_sparse(M0_np)
        # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        Ez = get_TM_dipole_field(M0, 1, 1/gpr, Nx, Ny, Nx//2, Ny//2, 1.0, chigrid=chigrid)#[npml:-npml, npml:-npml]
        # Ez.block_until_ready()

        assert np.allclose(M0.todense(), M0_np.todense())
        jax.profiler.start_trace("/tmp/tensorboard")

        t1 = time.time()
        Ez = get_TM_dipole_field(M0, 1, 1/gpr, Nx, Ny, Nx//2, Ny//2, 1.0, chigrid=chigrid) #[npml:-npml, npml:-npml]
        Ez.block_until_ready()

        t2 = time.time()

        print(f"jax sparse: {t2 - t1}")
        Ez_np = TM.get_TM_dipole_field(1, 1/gpr, Nx, Ny, Nx//2, Ny//2, (npml, npml), chigrid=chigrid_np) #[npml:-npml, npml:-npml]
        print(f"np sparse: {time.time() - t2}")

        comparison(Ez, Ez_np)

        Ez = get_TM_linesource_field(M0, 1, 1/gpr, Nx, Ny, Nx//2, 1.0, chigrid=chigrid)[npml:-npml, npml:-npml]
        Ez_np = TM.get_TM_linesource_field(1, 1/gpr, Nx, Ny, Nx//2, (npml, npml), chigrid=chigrid_np)[npml:-npml, npml:-npml]
        comparison(Ez, Ez_np)

        jax.profiler.stop_trace()

        #d = jit(lambda chi: jnp.real(jnp.sum(get_TM_dipole_field(M0, 1, 1/gpr, Nx, Ny, Nx//2, Ny//2, 1.0, chigrid=chi))))
        #res = grad(d)(jnp.ones((Nx, Ny), dtype=complex))
        
        #print("gradient calculation did not fail")
