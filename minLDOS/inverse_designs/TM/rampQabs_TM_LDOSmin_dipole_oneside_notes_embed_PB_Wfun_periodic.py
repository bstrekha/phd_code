import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time, sys, argparse
import nlopt
import ceviche
from ceviche.constants import C_0, ETA_0
from objective_TM_LDOS_PB_Wfun import designdof_ldos_objective_periodic, get_supercell_dof

parser = argparse.ArgumentParser()
parser.add_argument('-wavelength',action='store',type=float,default=1.0)

parser.add_argument('-pow10Qabs_start',action='store',type=float,default=2.0)
parser.add_argument('-pow10Qabs_end',action='store',type=float,default=6.0)
parser.add_argument('-pow10Qabs_num',action='store',type=int,default=5)
parser.add_argument('-dwfactor2',action='store',type=int,default=-1)

parser.add_argument('-ReChi',action='store',type=float,default=2.0)
parser.add_argument('-ImChi',action='store',type=float,default=1e-2)
parser.add_argument('-gpr',action='store',type=int,default=20)

###design area size, design area is rectangular with central rectangular hole where the dipole lives###
parser.add_argument('-design_x',action='store',type=float,default=1.0)
parser.add_argument('-design_y',action='store',type=float,default=1.0)
parser.add_argument('-geometry', action='store', type=str, default='Cavity')
parser.add_argument('-periodic', action='store', type=int, default=0) #default not periodic (1 is periodic)
parser.add_argument('-wavelengthPerCellX', action='store', type=float, default=1)
parser.add_argument('-wavelengthPerCellY', action='store', type=float, default=1)
parser.add_argument('-numCellsX', action='store', type=int, default=2) #how many cells into supercell
parser.add_argument('-numCellsY', action='store', type=int, default=2) #how many cells into supercell

parser.add_argument('-vacuum_x',action='store',type=float,default=0.2)
parser.add_argument('-vacuum_y',action='store',type=float,default=0.2)

parser.add_argument('-emitter_x',action='store',type=float,default=0.05)
parser.add_argument('-emitter_y',action='store',type=float,default=0.05)

parser.add_argument('-dist_x',action='store',type=float,default=0.02)

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
omega = C_0 * k
omega0 = C_0 * k
dl = 1.0/args.gpr
#design_vals = [1.0,1.26,1.59,2.0,2.45,3.0,4.0,5.0]
design_vals = [args.design_x]
for design_i in range(len(design_vals)):
#     print('design_y is',design_vals[design_i])
    numCellsX = int(args.numCellsX)
    numCellsY = int(args.numCellsY)
    Mx0 = int(np.round(args.wavelengthPerCellX/dl))
    My0 = int(np.round(args.wavelengthPerCellY/dl))
    Mx = Mx0*numCellsX
    My = My0*numCellsY
    
    Npml = int(np.round(args.pml_thick/dl))
    if design_vals[design_i] < 10:
        Npmlsep = int(np.round(0.5/dl))
        print('pml_sep is',0.5,flush=True)
    else:
        Npmlsep = int(np.round(1.0/dl))
        print('pml_sep is',1.0,flush=True)
    Emitterx = int(np.round(args.emitter_x / dl))
    Emittery = int(np.round(args.emitter_y / dl))
    Vacuumx = int(np.round(args.vacuum_x / dl))
    Vacuumy = int(np.round(args.vacuum_y / dl))
    Distx = int(np.round(args.dist_x / dl))

    Nx = Mx + 2*(Npmlsep+Npml)+Distx
    Ny = My + 2*(Npmlsep+Npml)

    design_mask = np.zeros((Nx,Ny), dtype=bool)
    desxL = Npml+Npmlsep+Distx
    desxR = Npml+Npmlsep+Distx+Mx
    desyL = Npml+Npmlsep
    desyR = Npml+Npmlsep+My
    design_mask[desxL:desxR, desyL:desyR] = True
    design_mask[Npml+Npmlsep+Distx+(Mx-Vacuumx)//2:Npml+Npmlsep+Distx+(Mx-Vacuumx)//2+Vacuumx,Npml+Npmlsep+(My-Vacuumy)//2:Npml+Npmlsep+(My-Vacuumy)//2+Vacuumy] = False
    
    chi = args.ReChi - 1j*args.ImChi #ceviche has +iwt time convention
    epsval = 1.0 + chi
    print('epsval', epsval, flush=True)
    if args.geometry.lower()[0] == 'c':
        print('cavity optimization', flush=True)
        emitter_mask = np.zeros((Nx,Ny), dtype=bool)
        emitter_mask[Npml+Npmlsep+(Mx-Emitterx)//2:Npml+Npmlsep+(Mx-Emitterx)//2+Emitterx,Npml+Npmlsep+(My-Emittery)//2:Npml+Npmlsep+(My-Emittery)//2+Emittery] = True
    else:
        print('half-space optimization', flush=True)
        emitter_mask = np.zeros((Nx,Ny), dtype=bool)
        emitter_mask[Npml+Npmlsep:Npml+Npmlsep+Emitterx,Npml+Npmlsep+(My-Emittery)//2:Npml+Npmlsep+(My-Emittery)//2+Emittery] = True

        
    cx = Npml+Npmlsep+(Mx-Emitterx)//2
    cy = Npml+Npmlsep+(My-Emittery)//2
    #set TM dipole source
    source = np.zeros((Nx,Ny), dtype=complex)
    source[emitter_mask] = 1.0 / (Emitterx*dl*Emittery*dl)
    #calculate field for source in vacuum
    epsVac = np.ones((Nx,Ny), dtype=complex)
    sim_vac = ceviche.fdfd_ez(omega, dl, epsVac, [Npml,Npml])
    _,_,vac_field = sim_vac.solve(source)
    
    #visualization
    """
    print('Nx', Nx, 'Ny', Ny, 'Mx', Mx, 'My', My)
    print('vac dipole center field', Ez[Npml+Npmlsep+(Mx-Emitterx)//2, Npml+Npmlsep+(My-Emittery)//2])
    print('vac ldos', np.sum(np.real(source*Ez))*dl**2/2.0)
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(Ez), cmap='RdBu', norm=DivergingNorm(vcenter=0.0))
    ax2.imshow(np.imag(Ez), cmap='RdBu', norm=DivergingNorm(vcenter=0.0))
    plt.savefig('test_TM_dipole_cplxfreq_source.png')
    """
    
    #calculate ldos numerically and compare to known result
    num_vac_ldos = np.real(np.sum(np.conj(source)*vac_field) / omega0 / omega0 * (omega0**2+(omega0**2)*(1/2./np.inf)**2)) * 0.5 * dl**2
    exact_vac_ldos = 1/4/np.pi/omega0*(omega0**2)*np.arctan(np.inf)*np.sqrt(1.25663706e-6/ 8.85418782e-12)*np.sqrt(1.25663706e-6 * 8.85418782e-12)
    opt_data['vac_ldos'] = exact_vac_ldos #vac lDOS in single freq limit at center freq
    print('numerical center freq vacuum LDOS', num_vac_ldos)
    print('exact center freq vacuum LDOS', exact_vac_ldos)
    
    #visually check configuration
    checkConfig = True
    if checkConfig:
        config = np.zeros((Nx,Ny))
        config[design_mask] = 1.0
        config[emitter_mask] = 2.0
        plt.imshow(config)
        plt.savefig(args.name+str(design_vals[design_i])+'_check_config.png')

    #set up optimization now
    ndof = Mx0*My0
    if args.init_type=='vac':
        designdof = np.zeros(ndof)
    if args.init_type=='slab':
        designdof = np.ones(ndof)
    if args.init_type=='half':
        designdof = 0.5*np.ones(ndof)
    if args.init_type=='rand':
        designdof = np.random.rand(ndof)

    #Qabslist = 10.0**np.linspace(args.pow10Qabs_start, args.pow10Qabs_end, args.pow10Qabs_num)
    Qabslist = np.logspace(args.pow10Qabs_start, args.pow10Qabs_end, args.pow10Qabs_num)
    if int(args.dwfactor2) == 1:
        Qabslist = Qabslist/2.0
    Qabslist = Qabslist.tolist() + [np.inf]
    
    for Qabs in Qabslist:
        print('at Qabs', Qabs)
        opt_data['count'] = 0 #refresh the iteration count
    
        if Qabs<1e16:
            omega_Qabs = omega * (1-1j/2/Qabs)
            opt_data['name'] = args.name + f'_Qabs{Qabs:.1e}'
        else:
            omega_Qabs = omega
            opt_data['name'] = args.name + '_Qinf'

        period_xPixels = int(args.gpr)
        period_yPixels = int(args.gpr)
        optfunc = lambda dof, grad: designdof_ldos_objective_periodic(dof, grad, args.wavelength, epsval, design_mask, dl, source, omega0, Qabs, epsVac, Npml, opt_data, cx, cy, desxL, desyL, numCellsX, numCellsY, Mx0, My0)

        lb = np.zeros(ndof)
        ub = np.ones(ndof)

        opt = nlopt.opt(nlopt.LD_MMA, int(ndof))
        #opt = nlopt.opt(nlopt.LD_LBFGS, int(ndof))
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)

        if Qabs < 1e4:
            opt.set_xtol_rel(1e-8)
        else:
            opt.set_xtol_rel(1e-11)
        opt.set_maxeval(args.maxeval)
        opt.set_min_objective(optfunc)
        vac_ldos_Q = 1/4/np.pi*omega0*(1 +(1/2./Qabs)**2)*np.arctan(2*Qabs)*np.sqrt(1.25663706e-6/ 8.85418782e-12)*np.sqrt(1.25663706e-6 * 8.85418782e-12)
        opt_data['vac_ldos_Q'] = vac_ldos_Q
        
        designdof = opt.optimize(designdof.flatten())
        min_ldos = opt.last_optimum_value()
        min_enh = min_ldos / vac_ldos_Q
        print('avg vacuum LDOS', vac_ldos_Q)

        print(f'Qabs{Qabs:.1e} best LDOS and enhancement found via topology optimization', min_ldos, min_enh)
        
        opt_design = np.zeros((Mx0*numCellsX, My0*numCellsY))
        get_supercell_dof(opt_design, designdof, design_mask, 0, 0, numCellsX, numCellsY, Mx0, My0)
        
        np.savetxt(opt_data['name'] + '_L' + str(design_vals[design_i])+ '_optdof.txt', opt_design.flatten())
        np.savetxt(args.init_file, opt_design.flatten())
    
        saveOptPDF = False
        if saveOptPDF:
            plt.figure()
            plt.imshow(opt_design, cmap='Greys')
            plt.savefig(opt_data['name']+str(design_vals[design_i])+'_opt_design.pdf', bbox_inches='tight')
    

