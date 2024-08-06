import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

import time,sys,argparse
sys.path.append('../../')
from dualbound.Maxwell.TM_1d_FDFD import build_TM_1d_vac_A
import nlopt

parser = argparse.ArgumentParser()
parser.add_argument('-wavelength',action='store',type=float,default=1.0)

parser.add_argument('-pow10Qsrc_start',action='store',type=float,default=0)
parser.add_argument('-pow10Qsrc_end',action='store',type=float,default=6)
parser.add_argument('-pow10Qsrc_num',action='store',type=int,default=13)
parser.add_argument('-includeQinf',action='store',type=int,default=-1)

parser.add_argument('-ReChi',action='store',type=float,default=4.0)
parser.add_argument('-ImChi',action='store',type=float,default=0.1)
parser.add_argument('-gpr',action='store',type=int,default=1200)
parser.add_argument('-omega_pts',action='store',type=int,default=1001)

parser.add_argument('-zMin',action='store',type=float,default=0.0)
parser.add_argument('-zMax',action='store',type=float,default=2.0)
parser.add_argument('-pml_sep',action='store',type=float,default=3.0)
parser.add_argument('-pml_thick',action='store',type=float,default=3.0)

parser.add_argument('-job',action='store',type=int,default=1) # 0 objective evaluation -1 gradient check 1 optimization
parser.add_argument('-init_type',action='store',type=str,default='file')
parser.add_argument('-init_file',action='store',type=str,default='DATA/TMcyl_chi4+1e-1j_R20_gpr1200_Qsrc1.0e+06_optdof.txt')
parser.add_argument('-output_base',action='store',type=int,default=10)
parser.add_argument('-xtol_rel',action='store',type=float,default=1e-3)
parser.add_argument('-maxeval', action='store',type=int,default=1000)
parser.add_argument('-name',action='store',type=str,default='DATA/TMcyl_chi4+1e-1j_R20_gpr1200_Qsrc1.0e+06_optdof_spectrum')

args,unknown = parser.parse_known_args()
print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

chi = args.ReChi + 1j*args.ImChi
dz = 1.0/args.gpr

N_domain = int(np.round((args.zMax - args.zMin) / dz))
N_pml_sep = int(np.round(args.pml_sep / dz))
N_pml = int(np.round(args.pml_thick / dz))

N_gridnum = N_domain + 2*N_pml_sep + 2*N_pml

z_grid = np.arange(N_gridnum)/args.gpr

omega0 = 2*np.pi/args.wavelength

des_mask = np.zeros(N_gridnum, dtype=bool)
des_mask[N_pml+N_pml_sep:N_pml+N_pml_sep+N_domain] = True

J = np.zeros(N_gridnum)
sR = N_gridnum //2
sL = sR - 1
sR = sR + 1
J[sL:sR] = 1.0 / dz / (sR - sL)

############initialize dofs##########################
ndof = int(np.sum(des_mask))
if args.init_type=='vac':
    designdof = np.zeros(ndof)
if args.init_type=='slab':
    designdof = np.ones(ndof)
if args.init_type=='half':
    designdof = 0.5*np.ones(ndof)
if args.init_type=='rand':
    designdof = np.random.rand(ndof)
if args.init_type=='phc':
    width = round(1/np.sqrt(1 + np.real(chi))/4.0*args.gpr)
    NR = N_domain
    NL = 0
    designdof = np.zeros(ndof)
    if (args.zMax - args.zMin) >= 4.0:
        aPer = (1 + np.sqrt(1 + np.real(chi)))/np.sqrt(1 + np.real(chi))/4.0
        aPixels = round(aPer*args.gpr)
        
        ind = N_domain//2 + round(args.gpr//8)
        while ind + width < NR:
            designdof[ind:ind+width] = 1.0
            ind += aPixels
            
        ind = N_domain//2 - round(args.gpr//8)
        while ind - width > NL:
            designdof[ind-width:ind] = 1.0
            ind -= aPixels
if args.init_type=='file':
    designdof = np.loadtxt(args.init_file)

omegas = (np.linspace(-1e6/2/10**(6),1e6/2/10**(6), int(args.omega_pts))+1)*2*np.pi

ldos_list = []
    
for w in omegas:
    Qsrc_omega = w
    Qsrc_Mvac = -build_TM_1d_vac_A(Qsrc_omega, N_gridnum, N_pml, dz)
    Qsrc_Mvac = Qsrc_Mvac.tocsr()
    chi_grid = np.zeros_like(z_grid, dtype=complex)
    chi_grid[des_mask] += designdof * chi
    M = Qsrc_Mvac - sp.diags(Qsrc_omega**2 * chi_grid, shape=Qsrc_Mvac.shape, format='csc')
    Qsrc_Evac = spla.spsolve(M, 1j*Qsrc_omega*J)
   
    k0r = np.real(Qsrc_omega)
    k0i = np.imag(Qsrc_omega)
    wtil = k0r + 1j*k0i
    Ntil = k0r/(k0r**2 + k0i**2)
    
    cur_ldos = -0.5*np.real(np.vdot(J*dz, Qsrc_Evac)/(wtil*Ntil))
    vac_ldos0 = 1.0/4
    ldos_list.append(cur_ldos / vac_ldos0)
    print('ldos/ldos0', cur_ldos / vac_ldos0, flush=True)

np.save(args.name, np.array(ldos_list))