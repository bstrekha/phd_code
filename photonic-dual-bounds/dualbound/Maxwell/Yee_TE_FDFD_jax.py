#!/usr/bin/env python3

"""
Note about jit: in jax jit'd code, size of slices can't be functions of arguments values 
but it can be functions of argument shapes. Below, whenver we need an array of size Nx 
or Ny and there isn't a convenient array with that shape we just jit with the static_argnums keyword. 
This is ok! In an inverse design run, we will be calling the same functions with the _same_ Nx, Ny
over and over again. 
"""

import numpy as np
import sys

sys.path.append('/home/aa7881/jax') # This is the path to the jax repo on the cluster
# Not worth fixing this right now, as soon as jax v0.4.15 is released, we will not need a custom compiled jax 

import jax.numpy as jnp
from jax import jit, grad
import jax 
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from functools import partial
import jax.experimental.sparse as jsparse
jax.config.update('jax_platform_name', 'cpu')
np.set_printoptions(linewidth=400)

@partial(jit, static_argnums=(1, 2))
def _get_pml_x(omega, Nx, Ny, Npmlx, dx, m=3, lnR=-20):
    '''
    This function gets the pml cells in the x direction, and leaves the middle region 
    without the PML analytic continuation. 

    Parameters
    ----------
    omega : complex
        Angular frequency
    Nx : int
        Size of domain in x
    Ny : int
        Size of domain in y
    Npmlx : int
        Number of pml pixels in x
    dx : float
        Grid spacing in x
    m, lnR : ints
        PML parameters that control the strength of the PML
    '''
    x = jnp.arange(Nx)
    y = jnp.arange(Ny)
    X,Y = jnp.meshgrid(x, y, indexing='ij')
    w_pml = Npmlx * dx
    sigma_max = -(m+1)*lnR / (2*w_pml) / omega
    
    mask1 = jnp.meshgrid(x < Npmlx, y, indexing='ij')[0]
    mask2 = jnp.meshgrid(x >= Nx-Npmlx, y, indexing='ij')[0]
    pml_x_Hz = jnp.where(mask1, 1.0 / (1.0 + 1j * sigma_max * ((Npmlx-X) / Npmlx)**m), jnp.ones((Nx, Ny), dtype=complex))
    pml_x_Hz = jnp.where(mask2, 1.0 / (1.0 + 1j * sigma_max * ((X-Nx+1+Npmlx)/Npmlx)**m), pml_x_Hz)

    pml_x_Ey = jnp.where(mask1, 1.0 / (1.0 + 1j * sigma_max * ((Npmlx-X+0.5) / Npmlx)**m), jnp.ones((Nx, Ny), dtype=complex))
    pml_x_Ey = jnp.where(mask2, 1.0 / (1.0 + 1j * sigma_max * ((X-Nx+1+Npmlx-0.5)/Npmlx)**m), pml_x_Ey)

    return pml_x_Hz, pml_x_Ey

@partial(jit, static_argnums=(1, 2))
def _get_pml_y(omega, Nx, Ny, Npmly, dy, m=3, lnR=-20):
    '''
    This function gets the pml cells in the y direction, and leaves the middle region 
    without the PML analytic continuation. 

    See _get_pml_x for more details
    '''
    x = jnp.arange(Nx)
    y = jnp.arange(Ny)
    
    X, Y = jnp.meshgrid(x,y, indexing='ij')
    w_pml = Npmly * dy
    sigma_max = -(m+1)*lnR / (2*w_pml) / omega
    
    mask1 = jnp.meshgrid(x, y < Npmly, indexing='ij')[1]
    mask2 = jnp.meshgrid(x, y >= Ny-Npmly, indexing='ij')[1]
    pml_y_Hz = jnp.where(mask1, 1.0 / (1.0 + 1j * sigma_max * ((Npmly-Y) / Npmly)**m), jnp.ones((Nx, Ny), dtype=complex))
    pml_y_Hz = jnp.where(mask2, 1.0 / (1.0 + 1j * sigma_max * ((Y-Ny+1+Npmly)/Npmly)**m), pml_y_Hz)

    pml_y_Ex = jnp.where(mask1, 1.0 / (1.0 + 1j * sigma_max * ((Npmly-Y-0.5) / Npmly)**m), jnp.ones((Nx, Ny), dtype=complex))
    pml_y_Ex = jnp.where(mask2, 1.0 / (1.0 + 1j * sigma_max * ((Y-Ny+1+Npmly+0.5)/Npmly)**m), pml_y_Ex)
    
    return pml_y_Hz, pml_y_Ex

@partial(jit, static_argnums=(1, 2, 3))
def _get_pml_x_wrapper(omega, Nx, Ny, Npmlx, dx, m=3, lnR=-20):
    if Npmlx == 0:
        return jnp.ones((Nx, Ny), dtype=complex), jnp.ones((Nx, Ny), dtype=complex)
    else:
        return _get_pml_x(omega, Nx, Ny, Npmlx, dx, m, lnR)

@partial(jit, static_argnums=(1, 2, 3))
def _get_pml_y_wrapper(omega, Nx, Ny, Npmly, dx, m=3, lnR=-20):
    if Npmly == 0:
        return jnp.ones((Nx, Ny), dtype=complex), jnp.ones((Nx, Ny), dtype=complex)
    else:
        return _get_pml_y(omega, Nx, Ny, Npmly, dx, m, lnR)

@partial(jit, static_argnums=(1,2,3,4))
def _build_TE_vac_A(omega, Nx, Ny, Npmlx, Npmly, dx, dy):
    """
    Construct TE FDFD system matrix A for vacuum. 
    The ordering of the indices goes (x,y,Hz), (x,y,Ex), (x,y,Ey), (x,y+1,Hz), (x,y+1,Ex) , ...
    Enfrocesa default periodic boundary conditions with no phase shift.
    Returns coordinates and data of A for use in sparse matrices.

    Parameters
    ----------
    omega : complex
        Angular frequency 
    Nx : int
        Size of domain in x
    Ny : int
        Size of domain in y
    Npmlx : int
        Number of pml pixels in x
    Npmly : int
        Number of pml pixels in y
    dx : float
        Grid spacing in x
    dy : float 
        Grid spacing in y

    Returns
    -------
    A_i : jnp.array
        Array corresponding to row indices of A
    A_j : jnp.array
        Array corresponding to column indices of A
    A_data : jnp.array
        Array corresponding to data of A
    """
    pml_x_Hz, pml_x_Ey = _get_pml_x_wrapper(omega, Nx, Ny, Npmlx, dx)
    pml_y_Hz, pml_y_Ex = _get_pml_y_wrapper(omega, Nx, Ny, Npmly, dy)

    CX, CY = jnp.meshgrid(jnp.arange(Nx), jnp.arange(Ny), indexing='ij')
    xyind = CX*Ny + CY
    Hzind = 3*xyind 

    i = Hzind.flatten()
    xp1yind = jnp.where(CX < Nx-1, (CX+1)*Ny + CY, CY).flatten()
    xyp1ind = jnp.where(CY < Ny-1, CX*Ny + CY+1, CX*Ny).flatten()
    xm1yind = jnp.where(CX > 0, (CX-1)*Ny + CY, (Nx-1)*Ny + CY).flatten()
    xym1ind = jnp.where(CY > 0, CX*Ny + CY-1, CX*Ny + Ny-1).flatten()

    A_i_1 = i
    A_j_1 = i
    A_data_1 = jnp.repeat(-1j*omega, len(i))

    jEx0 = 3*xym1ind + 1
    A_i_2 = i
    A_j_2 = jEx0
    A_data_2 = (pml_y_Hz/dy).flatten()

    jEx1 = i + 1
    A_i_3 = i
    A_j_3 = jEx1
    A_data_3 = (-pml_y_Hz/dy).flatten()

    jEy0 = i + 2
    A_i_4 = i
    A_j_4 = jEy0 
    A_data_4 = (-pml_x_Hz/dx).flatten()

    jEy1 = 3*xp1yind + 2
    A_i_5 = i
    A_j_5 = jEy1
    A_data_5 = (pml_x_Hz/dx).flatten()

    A_i_6 = i+1
    A_j_6 = i+1
    A_data_6 = jnp.repeat(1j*omega, len(i))

    jHz0 = Hzind.flatten()
    A_i_7 = i+1
    A_j_7 = jHz0
    A_data_7 = (-pml_y_Ex/dy).flatten()

    jHz1 = 3*xyp1ind 
    A_i_8 = i+1
    A_j_8 = jHz1
    A_data_8 = (pml_y_Ex/dy).flatten()

    A_i_9 = i+2
    A_j_9 = i+2
    A_data_9 = jnp.repeat(1j*omega, len(i))

    jHz0 = 3*xm1yind 
    A_i_10 = i+2
    A_j_10 = jHz0
    A_data_10 = (pml_x_Ey/dx).flatten()

    jHz1 = Hzind.flatten()
    A_i_11 = i+2
    A_j_11 = jHz1
    A_data_11 = (-pml_x_Ey/dx).flatten()

    A_i = jnp.concatenate((A_i_1, A_i_2, A_i_3, A_i_4, A_i_5, A_i_6, A_i_7, A_i_8, A_i_9, A_i_10, A_i_11))
    A_j = jnp.concatenate((A_j_1, A_j_2, A_j_3, A_j_4, A_j_5, A_j_6, A_j_7, A_j_8, A_j_9, A_j_10, A_j_11))
    A_data = jnp.concatenate((A_data_1, A_data_2, A_data_3, A_data_4, A_data_5, A_data_6, A_data_7, A_data_8, A_data_9, A_data_10, A_data_11))

    return A_i, A_j, A_data

@jit
def _get_diagA_from_chigrid(omega, chi_x, chi_y):
    '''
    The Maxwell operator has a diagonal component that corresponds to the susceptibility 
    distribution χ(r). This function constructs it.
    
    Parameters
    ----------
    omega : float 
        The angular frequency of harmonic oscillation 
    chi_x : jnp.array
        (Nx, Ny) size array corresponding to χ_x(r)
    chi_y : jnp.array
        (Nx, Ny) size array corresponding to χ_y(r)

    Returns
    -------
    x_ind : jnp.array
        Array corresponding to row indices of A
    y_ind : jnp.array
        Array corresponding to column indices of A
    data : jnp.array
        Array corresponding to data of A
    '''
    size = 3*chi_x.shape[0]*chi_x.shape[1]
    x_ind_1 = jnp.arange(1, size, 3, dtype=jnp.int32)
    x_ind_2 = jnp.arange(2, size, 3, dtype=jnp.int32)
    data = jnp.concatenate((1j*omega*chi_x.flatten(), 1j*omega*chi_y.flatten()))    
    x_ind = jnp.concatenate((x_ind_1, x_ind_2))
    return x_ind, x_ind, data

@partial(jit, static_argnums=(1,2,3,4))
def get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, cy, pol, amp=1.0, Qabs=jnp.inf, chigrid=None):
    '''
    Get the field of a dipole with x (pol = 1) or y (pol = 2) polarization. Solves the equation 
    -i \omega t (E) = [0           1/(1+chi(r)) curl] (E) + (-delta/(1+chi(r)))
                (H)   [-1/mu curl  0                ] (H)    0

    Parameters
    ----------
    wvlgth : float
        Wavelength
    Nx : int
        Size of domain in x 
    Ny : int 
        Size of domain in y 
    Npmlx : int 
        Number of pml pixels in x 
    Npmly : int 
        Number of pml pixels in y
    dx : float
        Grid spacing in x
    dy : float
        Grid spacing in y
    cx : int
        x coordinate of dipole
    cy : int
        y coordinate of dipole
    pol : int
        polarization, 1 for x, 2 for y
    amp : float
        amplitude of dipole
    Qabs : float
        Q factor of dipole
    chigrid : jnp.array
        (Nx, Ny) size array corresponding to χ_x(r) and χ_y(r)

    Returns
    -------
    Hzfield : jnp.array
        (Nx, Ny) size array corresponding to Hz field
    Exfield : jnp.array
        (Nx, Ny) size array corresponding to Ex field
    Eyfield : jnp.array
        (Nx, Ny) size array corresponding to Ey field
    '''

    omega = 2*jnp.pi/wvlgth * (1 + 1j/2/Qabs)
    total_Ai, total_Aj, total_Adata = _build_TE_vac_A(omega, Nx, Ny, Npmlx, Npmly, dx, dy)
    if not (chigrid is None):
        A_i2, A_j2, A_data2 = _get_diagA_from_chigrid(omega, chigrid, chigrid)
        total_Ai = jnp.concatenate((total_Ai, A_i2))
        total_Aj = jnp.concatenate((total_Aj, A_j2))
        total_Adata = jnp.concatenate((total_Adata, A_data2))

    xyind = cx*Ny + cy

    max_shape = Ny*Nx*3 
    b = jnp.zeros(max_shape, dtype=complex)
    b = b.at[3*xyind+pol].set(amp)

    nse = len(total_Adata)
    indices = jnp.column_stack((total_Ai, total_Aj))


    # TODO(alessio): It would be nice to extract indices and indptr without having to construct a BCOO 
    # There is a hidden method in experimental.sparse that can do this I think 
    # This is actually the bottleneck in this code for small system sizes. 
    # For bigger system sizes, this code is faster than scipy anyway

    # indices0, indptr0 = bcoo_to_bcsr(indices, shape=(max_shape, max_shape))
    A0 = jsparse.BCOO.sum_duplicates(jsparse.BCOO((total_Adata, indices), shape=(max_shape, max_shape)), nse=nse)
    A0 = jsparse.BCSR.from_bcoo(A0)
    tree, info = A0.tree_flatten()
    data, indices, indptr = tree     
    x = jsparse.linalg.spsolve(data, indices, indptr, b) 

    # Here is the scipy version to check the results 
    # x = spla.spsolve(sp.csr_matrix((total_Adata, (total_Ai, total_Aj)), dtype=complex), b) # This is the scipy equivalent, not differentiable. 

    Hzfield = jnp.reshape(x[::3], (Nx,Ny))
    Exfield = jnp.reshape(x[1::3], (Nx,Ny))
    Eyfield = jnp.reshape(x[2::3], (Nx,Ny))
    
    return Hzfield, Exfield, Eyfield

@partial(jit, static_argnums=(1,2,3,4))
def get_TE_linesource_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, pol, amp=1.0, Qabs=jnp.inf, chigrid=None):
    '''
    Convenient function to place a line source (plane when considering z direction) and solve for the field. 
    cx gives the location of the line which runs all along y
    Parameters and Returns: see function get_TE_dipole_field
    '''
    omega = 2*np.pi/wvlgth * (1 + 1j/2/Qabs)
    total_Ai, total_Aj, total_Adata = _build_TE_vac_A(omega, Nx, Ny, Npmlx, Npmly, dx, dy)

    if not (chigrid is None):
        A_i2, A_j2, A_data2 = _get_diagA_from_chigrid(omega, chigrid, chigrid)
        total_Ai = jnp.concatenate((total_Ai, A_i2))
        total_Aj = jnp.concatenate((total_Aj, A_j2))
        total_Adata = jnp.concatenate((total_Adata, A_data2))

    max_shape = Ny*Nx*3
    xyind = cx*Ny+jnp.arange(Ny)
    b = jnp.zeros(max_shape, dtype=complex)
    b = b.at[3*xyind+pol].set(amp)

    indices = jnp.column_stack((total_Ai, total_Aj))
    A0 = jsparse.BCOO((total_Adata, indices), shape=(max_shape, max_shape)).sum_duplicates(len(total_Adata))
    A0 = jsparse.BCSR.from_bcoo(A0)
    tree, info = A0.tree_flatten()
    data, indices, indptr = tree     
    x = jsparse.linalg.spsolve(data, indices, indptr, b) 

    # Scipy answer for comparison
    # x = spla.spsolve(sp.csr_matrix((total_Adata, (total_Ai, total_Aj)), dtype=complex), b)

    Hzfield = np.reshape(x[::3], (Nx,Ny))
    Exfield = np.reshape(x[1::3], (Nx,Ny))
    Eyfield = np.reshape(x[2::3], (Nx,Ny))
    
    return Hzfield, Exfield, Eyfield

def get_TE_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, source, pol, chigrid=None):
    '''
    Function to get the field from an arbitrary source distribution
    TODO(alessio): implement this function
    TODO(alessio): write get_TE_dipole_field and get_TE_linesource_field to just use this function 
    '''
    pass

def get_Yee_TE_masked_GreenFcn(wvlgth, Gx,Gy, Gmask, Npmlx, Npmly, dx,dy, Qabs=np.inf):
    """
    generate Green's function of a domain with shape specified by 2D boolean Gmask over a domain of size (Gx,Gy)
    Green's function basis ordering: ...(x,y,Ex)...(0,0,Ey),(0,1,Ey),...,(x,y,Ey),...
    """
    print("Warning, get_yee_TE_masked_GreenFcn does not use jax yet")
    gpwx = int(1.0/dx)
    gpwy = int(1.0/dy)
    Nx = 2*Gx-1 + gpwx//2 + 2*Npmlx
    Ny = 2*Gy-1 + gpwy//2 + 2*Npmly
    cx = Nx//2
    cy = Ny//2
    
    _, x_Exfield, x_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, cy, 1, amp=1, Qabs=Qabs)
    _, y_Exfield, y_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, cy, 2, amp=1, Qabs=Qabs)
    
    numCoord = np.sum(Gmask)
    G = np.zeros((2*numCoord,2*numCoord), dtype=complex)

    G_idx = np.argwhere(Gmask)
    
    for i,idx in enumerate(G_idx):
        ulx = cx - idx[0]
        uly = cy - idx[1]
        
        x_Ex = x_Exfield[ulx:ulx+Gx,uly:uly+Gy]
        x_Ey = x_Eyfield[ulx:ulx+Gx,uly:uly+Gy]
        y_Ex = y_Exfield[ulx:ulx+Gx,uly:uly+Gy]
        y_Ey = y_Eyfield[ulx:ulx+Gx,uly:uly+Gy]
        
        G[:numCoord,i] = x_Ex[Gmask]
        G[numCoord:,i] = x_Ey[Gmask]
        G[:numCoord,i+numCoord] = y_Ex[Gmask]
        G[numCoord:,i+numCoord] = y_Ey[Gmask]
        
    eta = 1.0 #dimensionless units
    k0 = 2*np.pi/wvlgth * (1+1j/2/Qabs) / 1
    Gfac = -1j*k0/eta
    G *= Gfac
    # print('check G reciprocity', np.linalg.norm(G-G.T))
    return G

def get_Yee_TE_Gddinv(wvlgth, dx, dy, Nx,Ny, Npmlx, Npmly, designMask, Qabs=np.inf, ordering='point'):
    print("warning: Gddinv does not use jax")
    omega = 2*np.pi/wvlgth * (1 + 1j/2/Qabs)
    k0 = omega / 1
    A_Yee = _build_TE_vac_A(omega, Nx, Ny, Npmlx, Npmly, dx, dy)
    
    full_designMask = np.zeros((3, Nx*Ny), dtype=bool)
    full_designMask[1,:] = designMask.flatten()
    full_designMask[2,:] = designMask.flatten()
    designInd = np.nonzero(full_designMask.T.flatten())[0] #transpose due to order in which pol. locations are labeled
    backgroundInd = np.nonzero(np.logical_not(full_designMask.T.flatten()))[0]
    
    A = (A_Yee[:,backgroundInd])[backgroundInd,:]
    B = (A_Yee[:,designInd])[backgroundInd,:]
    C = (A_Yee[:,backgroundInd])[designInd,:]
    D = (A_Yee[designInd,:])[:,designInd]
    
    AinvB = spla.spsolve(A, B)
    ETA_0 = 1.0
    Gfac = 1j*ETA_0 / k0 #note different Gfac compared with TM case due to directly extracting from A matrix
    Gddinv = (D - (C @ AinvB))*Gfac
    if ordering=='pol':
        num_des = np.sum(designMask)
        inds = np.arange(2*num_des)
        shuffle = np.zeros_like(inds, dtype=int)
        shuffle[:num_des] = inds[::2]
        shuffle[num_des:] = inds[1::2]
        Gddinv = Gddinv[shuffle,:][:,shuffle]
    return Gddinv, A_Yee
    
if __name__ == '__main__':
    import time 
    import Yee_TE_FDFD
    JIT_BEFORE = 1
    TEST_PML_FUNCTIONS = 1
    TEST_A_BUILD = 1
    TEST_DIAG_A_BUILD = 1
    TEST_TE_DIPOLE = 1
    TEST_TE_LINESOURCE = 1

    Nx, Ny = 40, 40
    Npmlx, Npmly = 10, 10
    print(f"Nx = {Nx}, Ny = {Ny}, Npmlx = {Npmlx}, Npmly = {Npmly}")

    if JIT_BEFORE:
        omega = 2*np.pi
        chigrid = jnp.ones((Nx, Ny), dtype=complex)
        Hz, Ey = _get_pml_x(omega, Nx, Ny, Npmlx, 1/200)
        yHz, yEx = _get_pml_y(omega, Nx, Ny, Npmly, 1/200)
        A_i, A_j, A_data = _build_TE_vac_A(omega, Nx, Ny, Npmlx, Npmly, 1/200, 1/200)
        x_ind, y_ind, _ = _get_diagA_from_chigrid(omega, chigrid, chigrid)
        Hzfield, Exfield, Eyfield = get_TE_dipole_field(1, Nx, Ny, Npmlx, Npmly, 1/200, 1/200, Nx//2, Ny//2, 1, amp=1, Qabs=jnp.inf, chigrid=chigrid)
        
        Hxfield, Exfield, Eyfield = get_TE_linesource_field(1, Nx, Ny, Npmlx, Npmly, 1/200, 1/200, Nx//2, 1, amp=1, Qabs=jnp.inf, chigrid=chigrid)

    if TEST_PML_FUNCTIONS:
        print('Testing get PML functions')
        import Yee_TE_FDFD 
        t1 = time.time()
        xHz, xEy = _get_pml_x(omega, Nx, Ny, Npmlx, 1/200)
        yHz, yEx = _get_pml_y(omega, Nx, Ny, Npmly, 1/200)
        t2 = time.time()
        xHz2, xEy2 = Yee_TE_FDFD.get_pml_x(omega, Nx, Ny, Npmlx, 1/200)
        yHz2, yEx2 = Yee_TE_FDFD.get_pml_y(omega, Nx, Ny, Npmly, 1/200)
        t3 = time.time()
        print('Time to get_pml_x/y (JIT): ', t2-t1)
        print("Time to get_pml_x/y      : ", time.time() - t2)

        assert np.allclose(xHz, xHz2, rtol=1e-6)
        assert np.allclose(xEy, xEy2, rtol=1e-6)
        assert np.allclose(yHz, yHz2, rtol=1e-6)
        assert np.allclose(yEx, yEx2, rtol=1e-6)

    if TEST_A_BUILD:
        print('Testing A operator builder')
        t1 = time.time()
        A_i, A_j, A_data = _build_TE_vac_A(omega, Nx, Ny, Npmlx, Npmly, 1/200, 1/200)
        t2 = time.time()
        A2 = Yee_TE_FDFD.build_TE_vac_A(omega, Nx, Ny, Npmlx, Npmly, 1/200, 1/200)
        print('Time to build A (JIT): ', t2-t1)
        print("Time to build A      : ", time.time() - t2)

        A = sp.coo_matrix((A_data, (A_i,A_j)), shape=(3*Nx*Ny,3*Nx*Ny)).tocsc()
        if Nx < 50:
            assert np.allclose(A.todense(), A2.todense(), atol=1e-6)

    if TEST_DIAG_A_BUILD:
        print("Testing diagA builder")
        t1 = time.time()
        x_ind, y_ind, A_data2 = _get_diagA_from_chigrid(1, chigrid, chigrid)
        t2 = time.time()
        diagA2 = Yee_TE_FDFD.get_diagA_from_chigrid(1, chigrid, chigrid)
        print('Time to build diagA (JIT): ', t2-t1)
        print("Time to build diagA      : ", time.time() - t2)

        diagA = sp.coo_matrix((A_data2, (x_ind, y_ind)), shape=(3*Nx*Ny,3*Nx*Ny)).tocsc()
        if Nx < 50:
            assert np.allclose(diagA.todense(), diagA2.todense(), atol=1e-6)

    if TEST_TE_DIPOLE:
        print('Testing differentiable TE mode solver')
        t1 = time.time()
        Hzfield, Exfield, Eyfield = get_TE_dipole_field(1, Nx, Ny, Npmlx, Npmly, 1/200, 1/200, Nx//2, Ny//2, 1, amp=1, Qabs=jnp.inf, chigrid=chigrid)
        t2 = time.time()
        Hzfield2, Exfield2, Eyfield2 = Yee_TE_FDFD.get_TE_dipole_field(1, Nx, Ny, Npmlx, Npmly, 1/200, 1/200, Nx//2, Ny//2, 1, amp=1, Qabs=jnp.inf, chigrid=chigrid)
        print('Time to get TE dipole field (JIT): ', t2-t1)
        print("Time to get TE dipole field      : ", time.time() - t2)

        assert np.allclose(Hzfield, Hzfield2, atol=1e-5)
        assert np.allclose(Exfield, Exfield2, atol=1e-5)
        assert np.allclose(Eyfield, Eyfield2, atol=1e-5)

    if TEST_TE_LINESOURCE:
        print('Testing differentiable TE mode solver')
        t1 = time.time()
        Hzfield, Exfield, Eyfield = get_TE_linesource_field(1, Nx, Ny, Npmlx, Npmly, 1/200, 1/200, Nx//2, 1, amp=1, Qabs=jnp.inf, chigrid=chigrid)
        t2 = time.time()
        Hzfield2, Exfield2, Eyfield2 = Yee_TE_FDFD.get_TE_linesource_field(1, Nx, Ny, Npmlx, Npmly, 1/200, 1/200, Nx//2, 1, amp=1, Qabs=jnp.inf, chigrid=chigrid)
        print('Time to get TE linesource field (JIT): ', t2-t1)
        print("Time to get TE linesource field      : ", time.time() - t2)

        assert np.allclose(Hzfield, Hzfield2, atol=1e-5)
        assert np.allclose(Exfield, Exfield2, atol=1e-5)
        assert np.allclose(Eyfield, Eyfield2, atol=1e-5)

    print("All tests passed.")
    
    print("Testing gradient")
    fixed_size_getTEdipolefield = lambda cgrid : jnp.real(jnp.sum(get_TE_dipole_field(1, Nx, Ny, 1, 1, 1/200, 1/200, Nx, Ny//2, 1, amp=1, Qabs=jnp.inf, chigrid=cgrid, A=(A_i, A_j, A_data))[0]))
    gradfunc = grad(fixed_size_getTEdipolefield)
    print("Gradient calculation did not fail.")

