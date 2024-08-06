import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import time,sys
sys.path.append('../../')

from dualbound.Maxwell.TM_radial_FDFD import get_radial_MaxwellOp


def TM_radial_LDOS_Wfun(dof, dofgrad, r_grid, des_mask, omega, epsdiff, Mvac, opt_data, Qsrc_vac_ldos=0.0):
    """
    inverse design objective for LDOS enhancement of 2D TM dipolar source
    at the origin of a cylindrically symmetric bullseye grating design
    following the Lai scheme for polar coordinate finite differences
    Qsrc_vac_ldos is the vacuum ldos evaluated at the given source bandwidth and discretization
    """

    dr = r_grid[1] - r_grid[0] #assume uniform grid
    J = np.zeros_like(r_grid)
    J[0] = 1 / (2*np.pi*r_grid[0]*dr)

    epsdiff_grid = np.zeros_like(r_grid, dtype=complex)
    epsdiff_grid[des_mask] += dof * epsdiff
    M = Mvac - sp.diags(omega**2 * epsdiff_grid, shape=Mvac.shape, format='csc')

    E = spla.spsolve(M, 1j*omega*J)
    
    k0r = np.real(omega)
    k0i = np.imag(omega)
    wtil = k0r + 1j*k0i
    Ntil = k0r/(k0r**2 + k0i**2)

    ldos = 0.5*np.real(np.vdot(J, 2*np.pi*dr*r_grid*E)/(wtil*Ntil)) #modified dot product for radial case

    opt_data['count'] += 1
    
    print('at iteration #', opt_data['count'], 'the scattered ldos value is', ldos-Qsrc_vac_ldos, 'with enhancement', (ldos-Qsrc_vac_ldos)/abs(Qsrc_vac_ldos), flush=True)

    if opt_data['count'] % opt_data['output_base'] == 0:
        np.savetxt(opt_data['name']+'_dof'+str(opt_data['count'])+'.txt', dof)
    
    if len(dofgrad)>0:
        dofgrad[:] = -0.5*np.real(epsdiff * (E*2*np.pi*dr*r_grid*E) * 1j * omega/(wtil*Ntil))[des_mask] #add in discretization factor

    return ldos - Qsrc_vac_ldos

