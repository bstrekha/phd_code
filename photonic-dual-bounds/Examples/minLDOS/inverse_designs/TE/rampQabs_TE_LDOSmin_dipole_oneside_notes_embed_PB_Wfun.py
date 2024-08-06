import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time,sys,argparse

sys.path.append('../photonic-bounds')
from dualbound.Maxwell.Yee_TE_FDFD_master import get_TE_dipole_field
from objective_TE_LDOS_PB_Wfun import designdof_ldos_objective
import ceviche
from ceviche.constants import C_0, ETA_0

import nlopt

parser = argparse.ArgumentParser()
parser.add_argument('-wavelength',action='store',type=float,default=1.0)

parser.add_argument('-pow10Qabs_start',action='store',type=float,default=2.0)
parser.add_argument('-pow10Qabs_end',action='store',type=float,default=6.0)
parser.add_argument('-pow10Qabs_num',action='store',type=int,default=5)
parser.add_argument('-Qinfonly',action='store',type=int,default=0)

parser.add_argument('-ReChi',action='store',type=float,default=2.0)
parser.add_argument('-ImChi',action='store',type=float,default=1e-2)
parser.add_argument('-gpr',action='store',type=int,default=20)

###design area size, design area is rectangular with central rectangular hole where the dipole lives###
parser.add_argument('-design_x',action='store',type=float,default=1.0)
parser.add_argument('-design_y',action='store',type=float,default=1.0)

parser.add_argument('-vacuum_x',action='store',type=float,default=0.2)
parser.add_argument('-vacuum_y',action='store',type=float,default=0.2)

parser.add_argument('-emitter_x',action='store',type=float,default=0.05)
parser.add_argument('-emitter_y',action='store',type=float,default=0.05)
parser.add_argument('-pol',action='store',type=int,default=2) #pol=1 hor pol, pol=2 ver pol

parser.add_argument('-dist_x',action='store',type=float,default=0.1)

#separation between pml inner boundary and source walls
parser.add_argument('-pml_sep',action='store',type=float,default=0.5)
parser.add_argument('-pml_thick',action='store',type=float,default=0.5)

parser.add_argument('-init_type',action='store',type=str,default='vac')
parser.add_argument('-init_file',action='store',type=str,default='test.txt')
parser.add_argument('-maxeval',action='store',type=int,default=10000)
parser.add_argument('-output_base',action='store',type=int,default=10)
parser.add_argument('-name',action='store',type=str,default='test')


args,unknown = parser.parse_known_args()
print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

opt_data = {'count':0, 'output_base':args.output_base, 'name':args.name}

k = 2*np.pi/args.wavelength
omega = k
dl = 1.0/args.gpr
design_vals = [args.design_x]
pol = args.pol
for design_i in range(len(design_vals)):
    Mx = int(np.round(design_vals[design_i]/dl))
    My = int(np.round(design_vals[design_i]/dl))
    print('design_x and design_y are:',design_vals[design_i])
    if design_i == 0:
        Mx0 = int(np.round(1.0/dl))
        My0 = int(np.round(1.0/dl))
    else:
        Mx0 = int(np.round(design_vals[design_i-1]/dl))
        My0 = int(np.round(design_vals[design_i-1]/dl))
    Npml = int(np.round(args.pml_thick/dl))
    if design_vals[design_i] < 10:
        Npmlsep = int(np.round(0.5/dl))
        print('pml_sep is',0.5,flush=True)
    else:
        Npmlsep = int(np.round(1.0/dl))
        print('pml_sep is',1.0,flush=True)
    Emitterx = int(np.round(args.emitter_x / dl))
    Emittery = int(np.round(args.emitter_y / dl))
    Vacuumx = int(np.round(args.vacuum_x / dl)) #if you want a vacuum cavity in the center of design domain
    Vacuumy = int(np.round(args.vacuum_y / dl))
#     Vacuumx = Emitterx
#     Vacuumy = Emittery
    Distx = int(np.round(args.dist_x / dl))

    Nx = Mx + 2*(Npmlsep+Npml)+Distx
    Ny = My + 2*(Npmlsep+Npml)

    design_mask = np.zeros((Nx,Ny), dtype=np.bool)
    design_mask[Npml+Npmlsep+Distx:Npml+Npmlsep+Distx+Mx , Npml+Npmlsep:Npml+Npmlsep+My] = True

    if Distx==0:#this is case where dipole at center of design domain and one may want a cavity
        cavxL = Npml+Npmlsep+Distx+Mx//2 - Vacuumx//2
        cavxR = Npml+Npmlsep+Distx+Mx//2 + Vacuumx//2 + 1 #make cavity width symmetrical
        cavyL = Npml+Npmlsep+My//2 - Vacuumy//2
        cavyR = Npml+Npmlsep+My//2 + Vacuumy//2 + 1 #make cavity height symmetrical 
        print('cavxL:', cavxL, flush=True)
        print('cavxR:', cavxR, flush=True)
        print('cavyL:', cavyL, flush=True)
        print('cavyR:', cavyR, flush=True)
        design_mask[cavxL:cavxR, cavyL:cavyR] = False

    chi = args.ReChi + 1j*args.ImChi
    epsval = 1.0 + chi
    print('epsval', epsval, flush=True)

    epsVac = np.ones((Nx,Ny), dtype=np.complex)
    #define the emitter/source x,y left and right indices
    if Distx==0: #cavity case
        exL = Npml+Npmlsep+Distx+Mx//2 - Emitterx//2
        exR = Npml+Npmlsep+Distx+Mx//2 + Emitterx//2 #+ 1 #make source width symmetrical
        eyL = Npml+Npmlsep+My//2 - Emittery//2
        eyR = Npml+Npmlsep+My//2 + Emittery//2 #+ 1 #make source height symmetrical
    else: #half-space case
        exL = Npml+Npmlsep - Emittery//2
        exR = Npml+Npmlsep + Emitterx//2 #+ 1 #make source width symmetrical
        eyL = Npml+Npmlsep + My//2 - Emittery//2
        eyR = Npml+Npmlsep + My//2 + Emittery//2 #+ 1 #make source height symmetrical
    print('exL:', exL, flush=True)
    print('exR:', exR, flush=True)
    print('eyL:', eyL, flush=True)
    print('eyR:', eyR, flush=True)
    #set dipole source
    emitter_mask = np.zeros((Nx,Ny), dtype=np.bool)
    emitter_mask[exL:exR, eyL:eyR] = True

    source = np.zeros((Nx,Ny), dtype=np.complex)
    sourceXpixels = exR - exL
    sourceYpixels = eyR - eyL
    sourceAmp = 1.0 / (sourceXpixels*dl*sourceYpixels*dl)
    source[emitter_mask] = sourceAmp
    
    #check configuration
    config = np.zeros((Nx,Ny))
    config[design_mask] = 1.0
    config[emitter_mask] = 2.0
    plt.imshow(config)
    plt.savefig(args.name+str(design_vals[design_i])+'_check_config.png')

    _, vac_fieldx, vac_fieldy = get_TE_dipole_field(args.wavelength, Nx, Ny, Npml, Npml, dl, dl, exL, exR, eyL, eyR, pol, amp=sourceAmp, chigrid=epsVac-1, omega_Qabs=omega)

    if pol == 1:
        num_vac_ldos = -np.real(np.sum(np.conj(source)*vac_fieldx)) * 0.5 * dl**2
    else:
        num_vac_ldos = -np.real(np.sum(np.conj(source)*vac_fieldy)) * 0.5 * dl**2
    exact_vac_ldos = omega/16
    opt_data['vac_ldos'] = exact_vac_ldos
    print('numerical center freq vacuum LDOS', num_vac_ldos)
    print('exact center freq vacuum LDOS', exact_vac_ldos)

    ndof = np.sum(design_mask)
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
        
#     if args.init_type!='file':
#         dofNx = int(np.sqrt(len(designdof)))
#         designdof = designdof.reshape((dofNx,dofNx))
#         designdof[dofNx//2-12:dofNx//2+12, dofNx//2-12:dofNx//2+12] = False #put no material on sources, by default
#         designdof = designdof.flatten()
        
    Qabslist = 10.0**np.linspace(args.pow10Qabs_start, args.pow10Qabs_end, args.pow10Qabs_num)
    Qinfonly = args.Qinfonly
    if Qinfonly == 1:
        Qabslist = [np.inf]
    else:
        Qabslist = Qabslist.tolist() + [np.inf]
    
    for Qabs in Qabslist:
#         if Qabs < 10:
#             designdof = np.zeros(ndof) #for small Q (large bandwidth), vac init usually does better than rand
#             #at all other Q, start rand init
        print('at Qabs', Qabs)
        opt_data['count'] = 0 #refresh the iteration count
    
        if Qabs<1e16:
            omega_Qabs = omega * (1+1j/2/Qabs)
            opt_data['name'] = args.name + f'_Qabs{Qabs:.1e}'
        else:
            omega_Qabs = omega
            opt_data['name'] = args.name + '_Qinf'

        vac_ldos_Q = np.arctan(2*Qabs)*omega*(1 + (1.0/2/Qabs)**2)/(8*np.pi)
        opt_data['vac_ldos_Q'] = vac_ldos_Q
        optfunc = lambda dof, grad: designdof_ldos_objective(dof, grad, epsval, design_mask, dl, source, opt_data, args.wavelength, Nx, Ny, Npml, Npml, dl, dl, exL, exR, eyL, eyR, pol, sourceAmp, vac_ldos_Q, omega_Qabs, Qabs)

        lb = np.zeros(ndof)
        ub = np.ones(ndof)

        opt = nlopt.opt(nlopt.LD_MMA, int(ndof))
        #opt = nlopt.opt(nlopt.LD_LBFGS, int(ndof))
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        
        opt.set_xtol_rel(1e-8)
        opt.set_maxeval(args.maxeval)

        opt.set_min_objective(optfunc)
        
        designdof = opt.optimize(designdof.flatten())
        min_ldos = opt.last_optimum_value()
        min_enh = min_ldos / vac_ldos_Q

        print(f'Qabs{Qabs:.1e} best LDOS and suppression found via topology optimization', min_ldos, min_enh)
        np.savetxt(opt_data['name'] + '_L' + str(design_vals[design_i])+ '_optdof.txt', designdof)
        np.savetxt(args.init_file, designdof)

        opt_design = np.zeros((Nx,Ny))
        opt_design[design_mask] = designdof
        
        np.savetxt(opt_data['name'] + '_L' + str(design_vals[design_i])+ '_fulldesigndof_optdof.txt', opt_design[Npml+Npmlsep+Distx:Npml+Npmlsep+Distx+Mx,Npml+Npmlsep:Npml+Npmlsep+My])
        
        plt.figure()
        plt.imshow(np.reshape(opt_design[Npml+Npmlsep+Distx:Npml+Npmlsep+Distx+Mx,Npml+Npmlsep:Npml+Npmlsep+My], (Mx,My)))
        plt.savefig(opt_data['name']+str(design_vals[design_i])+'_opt_design.png')
