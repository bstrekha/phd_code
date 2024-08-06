import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

import time,sys,argparse
sys.path.append('../../')
from dualbound.Maxwell.TM_radial_FDFD import get_radial_MaxwellOp
import nlopt

from TM_radial_LDOS import TM_radial_LDOS

parser = argparse.ArgumentParser()
parser.add_argument('-wavelength',action='store',type=float,default=1.0)

parser.add_argument('-pow10Qsrc_start',action='store',type=float,default=0)
parser.add_argument('-pow10Qsrc_end',action='store',type=float,default=6)
parser.add_argument('-pow10Qsrc_num',action='store',type=int,default=13)
parser.add_argument('-includeQinf',action='store',type=int,default=-1)

parser.add_argument('-ReChi',action='store',type=float,default=4.0)
parser.add_argument('-ImChi',action='store',type=float,default=1.0)
parser.add_argument('-gpr',action='store',type=int,default=800)

parser.add_argument('-r_inner',action='store',type=float,default=0.0)
parser.add_argument('-r_outer',action='store',type=float,default=50.0)
parser.add_argument('-pml_sep',action='store',type=float,default=.5)
parser.add_argument('-pml_thick',action='store',type=float,default=2.0)

parser.add_argument('-job',action='store',type=int,default=1) # 0 objective evaluation -1 gradient check 1 optimization
parser.add_argument('-init_type',action='store',type=str,default='file')
parser.add_argument('-init_file',action='store',type=str,default='DATA/TMcyl_chi4+1j_R50_gpr800_Qsrc1.0e+06_optdof.txt')
parser.add_argument('-output_base',action='store',type=int,default=10)
parser.add_argument('-xtol_rel',action='store',type=float,default=1e-3)
parser.add_argument('-maxeval', action='store',type=int,default=1000)
parser.add_argument('-name',action='store',type=str,default='DATA/TMcyl_chi4+1j_R50_gpr800_Qsrc1.0e+06_optdof_spectrum.txt')

args,unknown = parser.parse_known_args()
print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

chi = args.ReChi + 1j*args.ImChi
dr = 1.0/args.gpr

N_r_inner = int(np.round(args.r_inner / dr))
N_r_outer = int(np.round(args.r_outer / dr))

N_pml_sep = int(np.round(args.pml_sep / dr))
N_pml = int(np.round(args.pml_thick / dr))

r_gridnum = N_r_outer + N_pml_sep + N_pml
r_max = dr * (r_gridnum + 0.5) #+1/2 due to particulars of the radial discretization scheme

omega0 = 2*np.pi/args.wavelength

des_mask = np.zeros(r_gridnum, dtype=bool)
des_mask[N_r_inner+1:N_r_outer+1] = True

vac_ldos = omega0 / 8 #single frequency vacuum ldos, for use as a unit

J = np.zeros(r_gridnum)
J[0] = 1 / (np.pi*dr**2)

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
if args.init_type=='file':
    designdof = np.loadtxt(args.init_file)

omegas = (np.linspace(-1e6/2/10**(6),1e6/2/10**(6),2001)+1)*2*np.pi

ldos_list = []
    
for w in omegas:
    Qsrc_omega = w

    Qsrc_Mvac, r_grid = get_radial_MaxwellOp(Qsrc_omega, r_max, r_gridnum, N_pml)
    Qsrc_Evac = spla.spsolve(Qsrc_Mvac, 1j*Qsrc_omega*J)
   
    k0r = np.real(Qsrc_omega)
    k0i = np.imag(Qsrc_omega)
    wtil = k0r + 1j*k0i
    Ntil = k0r/(k0r**2 + k0i**2)
    
    cur_ldos = TM_radial_LDOS(designdof, r_grid, des_mask, w, chi, Qsrc_Mvac)
    vac_ldos0 = 2*np.pi/8
    ldos_list.append(cur_ldos / vac_ldos0)
    print('ldos/ldos0', cur_ldos / vac_ldos0)

np.save('DATA/TMcyl_chi4+1j_R50_gpr800_Qsrc1.0e+06_optdof_spectrum', np.array(ldos_list))

