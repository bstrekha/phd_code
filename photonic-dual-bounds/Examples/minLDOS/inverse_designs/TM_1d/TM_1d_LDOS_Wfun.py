import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import time,sys
sys.path.append('../../')


def TM_1d_LDOS_Wfun(dof, dofgrad, z_grid, des_mask, sL, sR, omega, epsdiff, Mvac, opt_data, Qsrc_vac_ldos=0.0):
    """
    inverse design objective for LDOS enhancement of 1D TM dipolar source
    Qsrc_vac_ldos is the vacuum ldos evaluated at the given source bandwidth and discretization
    """

    dz = z_grid[1] - z_grid[0] #assume uniform grid
    J = np.zeros_like(z_grid)
    J[sL:sR] = 1.0 / dz / (sR - sL) #define source with unit amplitude

    epsdiff_grid = np.zeros_like(z_grid, dtype=complex)
    epsdiff_grid[des_mask] += dof * epsdiff
    M = Mvac - sp.diags(omega**2 * epsdiff_grid, shape=Mvac.shape, format='csc')

    E = spla.spsolve(M, 1j*omega*J)
    
    k0r = np.real(omega)
    k0i = np.imag(omega)
    wtil = k0r + 1j*k0i
    Ntil = k0r/(k0r**2 + k0i**2)

    ldos = 0.5*np.real(np.vdot(J, dz*E)/(wtil*Ntil))

    opt_data['count'] += 1
    printIterPeriod = 1000
    if (opt_data['count'] % printIterPeriod == 0):
        print('at iteration #', opt_data['count'], 'the scattered ldos value is', ldos-Qsrc_vac_ldos, 'with enhancement', (ldos-Qsrc_vac_ldos)/abs(Qsrc_vac_ldos), flush=True)

    if opt_data['count'] % opt_data['output_base'] == 0:
        np.savetxt(opt_data['name']+'_dof'+str(opt_data['count'])+'.txt', dof)
    
    if len(dofgrad)>0:
        dofgrad[:] = -0.5*np.real(epsdiff * (E*dz*E) * 1j * omega/(wtil*Ntil))[des_mask] #add in discretization factor

    return ldos - Qsrc_vac_ldos

