import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

import time,sys,argparse
sys.path.append('../../')
from dualbound.Maxwell.TM_1d_FDFD import build_TM_1d_vac_A
import nlopt

from TM_1d_LDOS_Wfun import TM_1d_LDOS_Wfun

parser = argparse.ArgumentParser()
parser.add_argument('-wavelength',action='store',type=float,default=1.0)

parser.add_argument('-pow10Qsrc_start',action='store',type=float,default=5)
parser.add_argument('-pow10Qsrc_end',action='store',type=float,default=6)
parser.add_argument('-pow10Qsrc_num',action='store',type=int,default=13)
parser.add_argument('-includeQinf',action='store',type=int,default=-1)
parser.add_argument('-dwfactor2',action='store',type=int,default=1)

parser.add_argument('-ReChi',action='store',type=float,default=4.0)
parser.add_argument('-ImChi',action='store',type=float,default=0.1)
parser.add_argument('-gpr',action='store',type=int,default=500)

parser.add_argument('-zMin',action='store',type=float,default=0.0)
parser.add_argument('-zMax',action='store',type=float,default=2.0)
parser.add_argument('-pml_sep',action='store',type=float,default=3.0)
parser.add_argument('-pml_thick',action='store',type=float,default=3.0)

parser.add_argument('-job',action='store',type=int,default=1) # 0 objective evaluation -1 gradient check 1 optimization
parser.add_argument('-init_type',action='store',type=str,default='vac')
parser.add_argument('-init_file',action='store',type=str,default='test.txt')
parser.add_argument('-output_base',action='store',type=int,default=10)
parser.add_argument('-xtol_rel',action='store',type=float,default=1e-4)
parser.add_argument('-xtol_abs',action='store',type=float,default=1e-15)
parser.add_argument('-maxeval', action='store',type=int,default=100)
parser.add_argument('-name',action='store',type=str,default='DATA/test')

args,unknown = parser.parse_known_args()
print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

opt_data = {'count':0, 'output_base':args.output_base, 'name':args.name}

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

vac_ldos = 1.0 / 4 #single frequency vacuum ldos, for use as a unit
opt_data['vac_ldos'] = vac_ldos

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

# plt.plot(np.arange(len(designdof))/args.gpr - len(designdof)/args.gpr/2, designdof)
# plt.savefig('test_initdof.pdf', bbox_inches='tight')

Qsrclist = 10.0**np.linspace(args.pow10Qsrc_start, args.pow10Qsrc_end, args.pow10Qsrc_num)
if int(args.dwfactor2) == 1:
    Qsrclist = Qsrclist/2
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

    Qsrc_Mvac = -build_TM_1d_vac_A(Qsrc_omega, N_gridnum, N_pml, dz)
    Qsrc_Mvac = Qsrc_Mvac.tocsr()
#     chi_grid = np.zeros_like(z_grid, dtype=complex)
#     chi_grid[des_mask] += designdof * chi
#     M = Qsrc_Mvac - sp.diags(Qsrc_omegaomega**2 * chi_grid, shape=Qsrc_Mvac.shape, format='csc')
#     Qsrc_Evac = spla.spsolve(M, 1j*Qsrc_omega*J)
#     k0r = np.real(Qsrc_omega)
#     k0i = np.imag(Qsrc_omega)
#     wtil = k0r + 1j*k0i
#     Ntil = k0r/(k0r**2 + k0i**2)
    
#     Qsrc_vac_ldos = -0.5*np.real(np.vdot(J*dz, Qsrc_Evac)/(wtil*Ntil))
#     exact_vac_ldos = 1.0/4
#     print('exact avg vac ldos', exact_vac_ldos, 'Qsrc vac ldos', abs(Qsrc_vac_ldos))

    exact_vac_ldos = 1.0/4
    optfunc = lambda dof, grad: TM_1d_LDOS_Wfun(dof, grad, z_grid, des_mask, sL, sR, Qsrc_omega, chi, Qsrc_Mvac, opt_data, -exact_vac_ldos)

    lb = np.zeros(ndof)
    ub = np.ones(ndof)

    opt = nlopt.opt(nlopt.LD_MMA, ndof)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    opt.set_xtol_abs(args.xtol_abs)
    opt.set_maxeval(args.maxeval)
    opt.set_max_objective(optfunc)

    designdof = opt.optimize(designdof)
    max_sca_ldos = opt.last_optimum_value()
    max_sca_enh = max_sca_ldos / exact_vac_ldos

    print(f'Qsrc{Qsrc:.1e} best scattered LDOS and enchancement found via topology optimization', max_sca_ldos, max_sca_enh)

    np.savetxt(opt_data['name'] + '_optdof.txt', designdof)
    np.savetxt(args.init_file, designdof)

    finished_Qsrc_list.append(Qsrc)
    sca_ldos_list.append(max_sca_ldos)
    sca_enh_list.append(max_sca_enh)

    np.save(args.name+'_Qsrc.npy', np.array(finished_Qsrc_list))
    np.save(args.name+'_LDOS_sca.npy', np.array(sca_ldos_list))
    np.save(args.name+'_LDOS_sca_enh.npy', np.array(sca_enh_list))

    np.savetxt(args.name+'_Qsrc.txt', np.array(finished_Qsrc_list))
    np.savetxt(args.name+'_LDOS_sca.txt', np.array(sca_ldos_list))
    np.savetxt(args.name+'_LDOS_sca_enh.txt', np.array(sca_enh_list))

