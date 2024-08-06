#when calling methods from this script, make sure to load the path to the entire package in the top level script

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

def get_TM_dipole_circular_grating_LDOS_global_bound(chi, wvlgth, r_inner, r_outer, pml_sep, pml_thick, gpr, Qabs=np.inf, justAsym=False, opttol=1e-2, fakeSratio=1e-2, iter_period=20):

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

    Jvac = np.zeros(r_gridnum)
    Jvac[0] = 1 / (2*np.pi*r_grid[0]*dr)
    Evac = spla.spsolve(M, 1j*omega*Jvac)
    
    #Ginv, Evac now is in the discretized basis with support value 1 at a given grid point
    print('vacuum ldos', -0.5*np.real(Evac[0]), flush=True)
    #we now convert Ginv, Evac to the normalized basis with support value sqrt(1/(2*np.pi*r_i*dr)) at grid point i
    basis_weight = np.sqrt(1.0/2/np.pi/dr/r_grid)
    print('Ginv shape', Ginv.shape, 'basis_weight shape', basis_weight.shape)
    Evac /= basis_weight
    Ginv = sp.diags(1.0/basis_weight[des_mask], format='csc') @ Ginv @ sp.diags(basis_weight[des_mask], format='csc')
    S1 = Evac[des_mask]

    UM = (Ginv.conj().T @ Ginv) / np.conj(chi) - Ginv

    GinvdagPdaglist = [Ginv.conj().T.tocsc()]
    UPlist = [UM.tocsc()]


    wtil = (2*np.pi/wvlgth) * (1.0 + 1j/2/Qabs)
    k0r = np.real(wtil)
    k0i = np.imag(wtil)
    Ntil = k0r/abs(wtil)**2
    O_lin = -0.5 * (1j/2) * np.conj(omega) * (-Ginv.conj().T @ np.conj(S1)) / np.conj(wtil*Ntil) #ATTENTION
    O_quad = sp.csc_matrix(Ginv.shape)

    if justAsym:
        include = np.array([False, True])
    else:
        include = np.array([True,True])

    print('entering get_Msparse_bound', flush=True)
    
    optLags, optgrad, optdual, optobj = get_Msparse_bound(S1, O_lin, O_quad, GinvdagPdaglist, UPlist, include)
    
    rho0 = 1/(4*np.pi*Ntil)*np.arctan2(k0r, k0i)
    enh = -optdual / rho0 #calculate enhancement factor of scattered LDOS relative to single frequency vacuum ldos which in the 2D TM case is omega/8

    return optdual, enh