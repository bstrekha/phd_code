import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import time,sys
sys.path.append('../../')

from dualbound.Maxwell.TM_radial_FDFD import get_radial_MaxwellOp

def TM_radial_LDOS(dof, r_grid, des_mask, omega, epsdiff, Mvac):
    """
    LDOS of 2D TM dipolar source
    at the origin of a cylindrically symmetric bullseye grating design
    following the Lai scheme for polar coordinate finite differences
    """

    dr = r_grid[1] - r_grid[0] #assume uniform grid
    J = np.zeros_like(r_grid)
    J[0] = 1 / (2*np.pi*r_grid[0]*dr)

    epsdiff_grid = np.zeros_like(r_grid, dtype=complex)
    epsdiff_grid[des_mask] += dof * epsdiff
    M = Mvac - sp.diags(omega**2 * epsdiff_grid, shape=Mvac.shape, format='csc')

    E = spla.spsolve(M, 1j*omega*J)

    ldos = -0.5*np.real(np.vdot(J, 2*np.pi*dr*r_grid*E)) #modified dot product for radial case
    #ldosw0 = 2*np.pi/8
    return ldos

