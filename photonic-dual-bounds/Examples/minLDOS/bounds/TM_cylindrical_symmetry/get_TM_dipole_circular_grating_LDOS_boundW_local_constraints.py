import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

import time,sys
sys.path.append('../../') #load package directory into system path so the script can access the package

from dualbound.Maxwell.TM_FDFD import get_TM_dipole_field, get_Gddinv
from dualbound.Maxwell.TM_radial_FDFD import get_radial_MaxwellOp, get_radial_Gddinv

from dualbound.Lagrangian.spatialProjopt_Zops_Msparse import get_Msparse_gradZTT, get_Msparse_gradZTS_S, check_Msparse_spatialProj_Lags_validity, check_Msparse_spatialProj_incLags_validity

from get_Msparse_bounds import get_Msparse_bound

"""
radial discretization scheme following
A Note on Finite Difference Discretizations for Poisson Equation on a Disk
by Ming-Chi Lai
"""

def get_TM_dipole_circular_grating_LDOS_global_bound_local_constraints(chi, wvlgth, r_inner, r_outer, pml_sep, pml_thick, gpr, Qabs=np.inf, justAsym=False, opttol=1e-2, fakeSratio=1e-2, iter_period=20, numSubRegions=0):
    '''
    Evaluates limit of enhancement to dipole radiation power in 1D for a TM dipole
    at the center of a circular symmetric design of inner radius r_inner and
    outer radius r_outer.
    '''
    dr = 1.0/gpr

    N_r_inner = int(np.round(r_inner / dr))
    N_r_outer = int(np.round(r_outer / dr))

    N_pml_sep = int(np.round(pml_sep / dr))
    N_pml = int(np.round(pml_thick / dr))

    r_gridnum = N_r_outer + N_pml_sep + N_pml
    r_max = dr * (r_gridnum + 0.5) #+1/2 due to particulars of the radial discretization scheme
    
    omega = (2*np.pi/wvlgth) * (1 + 1j/2/Qabs)

    #############get operators and fields ##########
    M, r_grid = get_radial_MaxwellOp(omega, r_max, r_gridnum, N_pml)
    print('dr', dr, r_grid[1]-r_grid[0], 'first r_grid point', r_grid[0]) #check r_grid is constructed as expected
    des_mask = np.zeros(r_gridnum, dtype=bool)
    des_mask[N_r_inner+1:N_r_outer+1] = True
    Ginv = get_radial_Gddinv(omega, M, des_mask)

    #Set up source
    Jvac = np.zeros(r_gridnum)
    Jvac[0] = 1 / (2*np.pi*r_grid[0]*dr)
    Evac = spla.spsolve(M, 1j*omega*Jvac)
    
    #Ginv, Evac now is in the discretized basis with support value 1 at a given grid point
    print('num vacuum ldos', -0.5*np.real(Evac[0]), flush=True)
    wtil = (2*np.pi/wvlgth) * (1.0 + 1j/2/Qabs)
    k0r = np.real(wtil)
    k0i = np.imag(wtil)
    Ntil = k0r/abs(wtil)**2
    print('exact vacuum ldos', 1/(4*np.pi*Ntil)*np.arctan2(k0r, k0i))
    #we now convert Ginv, Evac to the normalized basis with support value sqrt(1/(2*np.pi*r_i*dr)) at grid point i
    basis_weight = np.sqrt(1.0/2/np.pi/dr/r_grid)
    print('Ginv shape', Ginv.shape, 'basis_weight shape', basis_weight.shape)
    Evac /= basis_weight
    
    Ginv = sp.diags(1.0/basis_weight[des_mask], format='csc') @ Ginv @ sp.diags(basis_weight[des_mask], format='csc')
    S1 = Evac[des_mask]

    UM = (Ginv.conj().T @ Ginv) / np.conj(chi) - Ginv
    
    #############get projection operators#############
    GinvdagPdaglist = [Ginv.conj().T.tocsc()]
    UPlist = [UM.tocsc()]
    
    pixelsDes = int(sum(des_mask))
#     numSubRegions = 10
    for i in range(numSubRegions):
        subRegionGpr = pixelsDes//numSubRegions
        Proj_mat = np.zeros(pixelsDes, dtype=int)
        indL = subRegionGpr*i
        indR = subRegionGpr*(i+1)
        Proj_mat[indL:indR] = 1
        Proj_mat = sp.diags(Proj_mat, format="csc")
        GinvdagPdag = Ginv.conj().T @ Proj_mat
        UMP = (Ginv.conj().T @ GinvdagPdag.conj().T)/np.conj(chi) - GinvdagPdag.conj().T
        #print('GinvdagPdag format', GinvdagPdag.format)
        #print('UMP format', UMP.format, flush=True)
        GinvdagPdaglist.append(GinvdagPdag.tocsc())
        UPlist.append(UMP.tocsc())
        
        
    include = np.array([True] * (numSubRegions+1) * 2 )
    
    if numSubRegions > 0:
        #since we include global constraints, leave out last subregion so that dual Hessian is not singular
        include[-1] = False
        include[-2] = False
    
    print('len of include:', len(include), flush=True)
    print('len of UPlist:', len(UPlist), flush=True)
    print('len of GinvdagPdaglist:', len(GinvdagPdaglist), flush=True)
    
    # if justAsym:
    #     for i in range(numRegions+1):
    #         include[2*i] = False

    #############set up optimization#############
    wtil = (2*np.pi/wvlgth) * (1.0 + 1j/2/Qabs)
    k0r = np.real(wtil)
    k0i = np.imag(wtil)
    Ntil = k0r/abs(wtil)**2
    O_lin = -0.5 * (1j/2) * np.conj(omega) * (-Ginv.conj().T @ np.conj(S1)) / np.conj(wtil*Ntil) #ATTENTION
    O_quad = sp.csc_matrix(Ginv.shape)

    print('entering get_Msparse_bound', flush=True)
    
    optLags, optgrad, optdual, optobj = get_Msparse_bound(S1, O_lin, O_quad, GinvdagPdaglist, UPlist, include)
    
    rho0 = 1/(4*np.pi*Ntil)*np.arctan2(k0r, k0i)
    enh = -optdual / rho0 #calculate enhancement factor of scattered LDOS relative to single frequency vacuum ldos which in the 2D TM case is omega/8

    return optdual, enh
