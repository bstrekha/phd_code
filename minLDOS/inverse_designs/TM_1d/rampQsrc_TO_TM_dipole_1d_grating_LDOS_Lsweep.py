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

parser.add_argument('-Qsrc',action='store',type=float,default=1000000)

parser.add_argument('-ReChi',action='store',type=float,default=4.0)
parser.add_argument('-ImChi',action='store',type=float,default=0.1)
parser.add_argument('-gpr',action='store',type=int,default=1000)

parser.add_argument('-L_start',action='store',type=float,default=1.0)
parser.add_argument('-L_end',action='store',type=float,default=80.0)
parser.add_argument('-L_pts',action='store',type=float,default=30)

parser.add_argument('-pml_sep',action='store',type=float,default=1.0)
parser.add_argument('-pml_thick',action='store',type=float,default=3.0)

parser.add_argument('-job',action='store',type=int,default=1) # 0 objective evaluation -1 gradient check 1 optimization
parser.add_argument('-init_type',action='store',type=str,default='vac')
parser.add_argument('-init_file',action='store',type=str,default='test.txt')
parser.add_argument('-output_base',action='store',type=int,default=10)
parser.add_argument('-xtol_rel',action='store',type=float,default=1e-3)
parser.add_argument('-xtol_abs',action='store',type=float,default=1e-15)
parser.add_argument('-maxeval', action='store',type=int,default=100)
parser.add_argument('-name',action='store',type=str,default='DATA/test')

args,unknown = parser.parse_known_args()
print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

opt_data = {'count':0, 'output_base':args.output_base, 'name':args.name}

Llist = np.geomspace(args.L_start, args.L_end, int(args.L_pts))

finished_L_list = []
sca_ldos_list = []
sca_enh_list = []

for i in range(len(Llist)):
    chi = args.ReChi + 1j*args.ImChi
    dz = 1.0/args.gpr
    
    zMin = 0
    zMax = Llist[i]

    N_domain = int(np.round((zMax - zMin) / dz))
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
    
    Qsrc = args.Qsrc
    print('at L', zMax)
    opt_data['count'] = 0 #refresh the iteration count

    if Qsrc<1e16:
        Qsrc_omega = omega0 * (1+1j/2/Qsrc)
        opt_data['name'] = args.name + f'_Qsrc{Qsrc:.1e}'
    else:
        Qsrc_omega = omega0
        opt_data['name'] = args.name + '_QsrcINF'

    Qsrc_Mvac = -build_TM_1d_vac_A(Qsrc_omega, N_gridnum, N_pml, dz)
    Qsrc_Mvac = Qsrc_Mvac.tocsr()
    
    exact_vac_ldos = 1.0/4
    optfunc = lambda dof, grad: TM_1d_LDOS_Wfun(dof, grad, z_grid, des_mask, sL, sR, Qsrc_omega, chi, Qsrc_Mvac, opt_data, -exact_vac_ldos)

    lb = np.zeros(ndof)
    ub = np.ones(ndof)

    opt = nlopt.opt(nlopt.LD_MMA, ndof)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    #opt.set_xtol_rel(args.xtol_rel)
    opt.set_xtol_abs(args.xtol_abs)
    opt.set_maxeval(args.maxeval)
    opt.set_max_objective(optfunc)

    designdof = opt.optimize(designdof)
    max_sca_ldos = opt.last_optimum_value()
    max_sca_enh = max_sca_ldos / exact_vac_ldos

    print(f'L{zMax:.1e} best scattered LDOS and enchancement found via topology optimization', max_sca_ldos, max_sca_enh)

#     np.savetxt(opt_data['name'] + '_optdof.txt', designdof)
    #np.savetxt(args.init_file, designdof)

    finished_L_list.append(zMax)
    sca_ldos_list.append(max_sca_ldos)
    sca_enh_list.append(max_sca_enh)

#     np.save(args.name+'_L.npy', np.array(finished_L_list))
#     np.save(args.name+'_LDOS_sca.npy', np.array(sca_ldos_list))
#     np.save(args.name+'_LDOS_sca_enh.npy', np.array(sca_enh_list))

    np.savetxt(args.name+'_L.txt', np.array(finished_L_list))
    np.savetxt(args.name+'_LDOS_sca.txt', np.array(sca_ldos_list))
    np.savetxt(args.name+'_LDOS_sca_enh.txt', np.array(sca_enh_list))