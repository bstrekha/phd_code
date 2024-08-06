import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

import time,sys,argparse
sys.path.append('../../')
from dualbound.Maxwell.TM_radial_FDFD import get_radial_MaxwellOp
import nlopt

from TM_radial_LDOS_Wfun import TM_radial_LDOS_Wfun

parser = argparse.ArgumentParser()
parser.add_argument('-wavelength',action='store',type=float,default=1.0)

parser.add_argument('-pow10Qsrc_start',action='store',type=float,default=-1)
parser.add_argument('-pow10Qsrc_end',action='store',type=float,default=6)
parser.add_argument('-pow10Qsrc_num',action='store',type=int,default=15)
parser.add_argument('-includeQinf',action='store',type=int,default=-1)

parser.add_argument('-ReChi',action='store',type=float,default=4.0)
parser.add_argument('-ImChi',action='store',type=float,default=0.1)
parser.add_argument('-gpr',action='store',type=int,default=1200)

parser.add_argument('-r_inner',action='store',type=float,default=0.0)
parser.add_argument('-r_outer',action='store',type=float,default=50.0)
parser.add_argument('-pml_sep',action='store',type=float,default=.5)
parser.add_argument('-pml_thick',action='store',type=float,default=2.0)

parser.add_argument('-job',action='store',type=int,default=1) # 0 objective evaluation -1 gradient check 1 optimization
parser.add_argument('-init_type',action='store',type=str,default='vac')
parser.add_argument('-init_file',action='store',type=str,default='test.txt')
parser.add_argument('-output_base',action='store',type=int,default=6000)
parser.add_argument('-xtol_rel',action='store',type=float,default=1e-4)
parser.add_argument('-xtol_abs',action='store',type=float,default=1e-15)
parser.add_argument('-maxeval', action='store',type=int,default=10000)
parser.add_argument('-name',action='store',type=str,default='DATA/TMcyl_chi4+1e-1j_R50_gpr1000')

args,unknown = parser.parse_known_args()
print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

opt_data = {'count':0, 'output_base':args.output_base, 'name':args.name}

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

#vac_ldos = omega0 / 8 #single frequency vacuum ldos, for use as a unit

vac_ldos = omega0 / 8 #single frequency vacuum ldos, for use as a unit
opt_data['vac_ldos'] = vac_ldos

J = np.zeros(r_gridnum)
J[0] = 1.0 / (np.pi*dr**2)

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

Qsrclist = 10.0**np.linspace(args.pow10Qsrc_start, args.pow10Qsrc_end, args.pow10Qsrc_num)
if args.includeQinf>0:
    Qsrclist = Qsrclist.tolist() + [np.inf]

finished_Qsrc_list = []
sca_ldos_list = []
sca_enh_list = []
    
for Qsrc in Qsrclist:
    print('at Qsrc', Qsrc)
    opt_data['count'] = 0 #refresh the iteration count

    if Qsrc<1e16:
        Qsrc_omega = omega0 * (1+1j/2/Qsrc)
        opt_data['name'] = args.name + f'_Qsrc{Qsrc:.1e}'
    else:
        Qsrc_omega = omega0
        opt_data['name'] = args.name + '_QsrcINF'

    Qsrc_Mvac, r_grid = get_radial_MaxwellOp(Qsrc_omega, r_max, r_gridnum, N_pml)
    Qsrc_Evac = spla.spsolve(Qsrc_Mvac, 1j*Qsrc_omega*J)
   
    k0r = np.real(Qsrc_omega)
    k0i = np.imag(Qsrc_omega)
    wtil = k0r + 1j*k0i
    Ntil = k0r/(k0r**2 + k0i**2)
    
    Qsrc_vac_ldos = 0.5 * np.real(Qsrc_Evac[0]/(wtil*Ntil)) #minus sign difference
    exact_Qsrc_vac_ldos = 1/(4*np.pi*Ntil)*np.arctan2(k0r,k0i) 
    print('exact avg vac ldos', vac_ldos, 'Qsrc vac ldos', abs(Qsrc_vac_ldos))

    optfunc = lambda dof, grad: TM_radial_LDOS_Wfun(dof, grad, r_grid, des_mask, Qsrc_omega, chi, Qsrc_Mvac, opt_data, -exact_Qsrc_vac_ldos)

    print('ndof: ', ndof,flush=True)
    lb = np.zeros(ndof)
    ub = np.ones(ndof)

    opt = nlopt.opt(nlopt.LD_MMA, ndof)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    #opt.set_xtol_rel(args.xtol_rel)
    opt.set_xtol_rel(args.xtol_abs)
    opt.set_maxeval(args.maxeval)
    opt.set_max_objective(optfunc)

    designdof = opt.optimize(designdof)
    max_sca_ldos = opt.last_optimum_value()
    max_sca_enh = max_sca_ldos / exact_Qsrc_vac_ldos

    print(f'Qsrc{Qsrc:.1e} best scattered LDOS and enchancement found via topology optimization', max_sca_ldos, max_sca_enh)

    np.savetxt(opt_data['name'] + '_optdof.txt', designdof)
    #np.savetxt(args.init_file, designdof)

    finished_Qsrc_list.append(Qsrc)
    sca_ldos_list.append(max_sca_ldos)
    sca_enh_list.append(max_sca_enh)

    np.save(args.name+'_Qsrc.npy', np.array(finished_Qsrc_list))
    np.save(args.name+'_LDOS_sca.npy', np.array(sca_ldos_list))
    np.save(args.name+'_LDOS_sca_enh.npy', np.array(sca_enh_list))

    np.savetxt(args.name+'_Qsrc.txt', np.array(finished_Qsrc_list))
    np.savetxt(args.name+'_LDOS_sca.txt', np.array(sca_ldos_list))
    np.savetxt(args.name+'_LDOS_sca_enh.txt', np.array(sca_enh_list))

