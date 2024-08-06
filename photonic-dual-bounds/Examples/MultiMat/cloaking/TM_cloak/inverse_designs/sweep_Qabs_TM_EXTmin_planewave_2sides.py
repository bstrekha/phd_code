import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import time, sys, argparse
sys.path.append('../../../../../')
from dualbound.Maxwell import TM_FDFD as TM

import nlopt
from objective_TM_ext_2sides import designdof_ext_objective
from objective_TM_ext_2sides import ext_objective

parser = argparse.ArgumentParser()
parser.add_argument('-wavelength',action='store',type=float,default=1.0)

parser.add_argument('-pow10Qabs_start',action='store',type=float,default=1.0)
parser.add_argument('-pow10Qabs_end',action='store',type=float,default=7.0)
parser.add_argument('-pow10Qabs_num',action='store',type=int,default=13)
parser.add_argument('-dwfactor2',action='store',type=int,default=0) #1 = True, 0 = False for diving Qabs list by 2
parser.add_argument('-Qabs', action='store', type=float, default=np.inf)

parser.add_argument('-chifRe', action='store', type=float, default=2.0)
parser.add_argument('-chifIm', action='store', type=float, default=0.01)
parser.add_argument('-chidRe', action='store', type=float, default=4.0)
parser.add_argument('-chidIm', action='store', type=float, default=0.0001)

parser.add_argument('-des_region', action='store', type=str, default='circle') #options are 'rect' and 'circle'
parser.add_argument('-design_x', action='store', type=float, default=2.0)
parser.add_argument('-design_y', action='store', type=float, default=2.0)
parser.add_argument('-des_param1', action='store', type=float, default=0.5) #inner radius of circle design region
parser.add_argument('-des_param2', action='store', type=float, default=1.0) #outer radius of circle design region
parser.add_argument('-gpr', action='store', type=int, default=50)

#separation between pml inner boundary and source walls
parser.add_argument('-pml_sep',action='store',type=float,default=0.5)
parser.add_argument('-pml_thick',action='store',type=float,default=0.5)

parser.add_argument('-init_type',action='store',type=str,default='ones')
parser.add_argument('-init_file',action='store',type=str,default='') #name of init numpy file
parser.add_argument('-maxeval',action='store',type=int,default=120)
parser.add_argument('-xtol_abs',action='store',type=float,default=1e-15)
parser.add_argument('-output_base',action='store',type=int,default=60)
parser.add_argument('-name',action='store',type=str,default='test')

args,unknown = parser.parse_known_args()
print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

opt_data = {'count':0, 'output_base':args.output_base, 'name':args.name, 'background_term':0}

wv = args.wavelength
do_checks = True

des_region = args.des_region
assert(des_region in ['rect','circle','circlerand'])
des_params = [args.des_param1, args.des_param2]
chif = args.chifRe + 1j*args.chifIm
chid = args.chidRe + 1j*args.chidIm
pml_thick = args.pml_thick
pml_sep = args.pml_sep
design_x = args.design_x
design_y = args.design_y

for design_i in range(1):
    dl = 1.0/args.gpr
    Mx = int(np.round(design_x/dl))
    My = int(np.round(design_y/dl))
    Npml = int(np.round(pml_thick/dl))
    Npmlsep = int(np.round(pml_sep/dl))
    nonpmlNx = Mx + 2*Npmlsep
    nonpmlNy = My + 2*Npmlsep
    Nx = nonpmlNx + 2*Npml
    Ny = nonpmlNy + 2*Npml
    
    #define the masks
    background_mask = np.zeros((nonpmlNx,nonpmlNy), dtype=bool) #the fixed object, which is considered to be part of background
    if des_region == 'rect':
        #written this way for symmetry
        xwidth = int(des_params[0]/dl)
        ywidth = int(des_params[1]/dl)
        assert(xwidth <= Mx)
        assert(ywidth <= My)
        background_mask[Npmlsep + Mx//2 - xwidth//2:Npmlsep + Mx//2 + xwidth//2, Npmlsep + My//2 - ywidth//2:Npmlsep + My//2 + ywidth//2] = True
    elif des_region == 'circle':
        originx = Npmlsep + Mx//2
        originy = Npmlsep + My//2
        radius = des_params[0]
        assert(radius >= 0)
        assert(radius <= min(design_x/2, design_y/2))
        radiusN = int(radius/dl)
        for i in range(radiusN):
            for j in range(radiusN):
                if np.sqrt(i**2 + j**2) <= radiusN:
                    background_mask[originx+i, originy+j] = True
                    background_mask[originx+i, originy-j] = True
                    background_mask[originx-i, originy+j] = True
                    background_mask[originx-i, originy-j] = True
    elif des_region == 'circlerand':
        originx = Npmlsep+Mx//2
        originy = Npmlsep+My//2
        radius = des_params[0]
        assert(radius >= 0)
        assert(radius <= min(design_x/2, design_y/2))
        radiusN = int(radius/dl)
        for i in range(radiusN):
            for j in range(radiusN):
                if np.sqrt(i**2 + j**2) <= radiusN:
                    background_mask[originx+i, originy+j] = True
                    background_mask[originx+i, originy-j] = True
                    background_mask[originx-i, originy+j] = True
                    background_mask[originx-i, originy-j] = True
        #pick a random background that fits within the background region
        background_mask = background_mask.flatten()
        ids = np.where(background_mask)
        ids = ids[0]
        for i in range(len(ids)):
            background_mask[ids[i]] = np.random.rand() > 0.5
        background_mask = np.reshape(background_mask, (nonpmlNx,nonpmlNy))
    
    #background_mask = np.zeros((nonpmlNx,nonpmlNy), dtype=bool) #can check with only design region (no fixed object)
    if do_checks:
        config2 = np.zeros((nonpmlNx,nonpmlNy))
        config2[background_mask] = 2.0
        plt.figure()
        plt.imshow(config2)
        plt.savefig('check_background.png')
        
    design_mask = np.zeros((nonpmlNx,nonpmlNy), dtype=bool)

    if des_region == 'rect':
        design_mask[Npmlsep:Npmlsep+Mx, Npmlsep:Npmlsep+My] = True
        xwidth = int(des_params[0]/dl)
        ywidth = int(des_params[1]/dl)
        assert(xwidth <= Mx)
        assert(ywidth <= My)
        background_mask[Npmlsep + Mx//2 - xwidth//2:Npmlsep + Mx//2 + xwidth//2, Npmlsep + My//2 - ywidth//2:Npmlsep + My//2 + ywidth//2] = True
        design_mask[background_mask] = False
    elif des_region == 'circle':
        originx = Npmlsep+Mx//2
        originy = Npmlsep+My//2
        radius = des_params[1]
        assert(radius >= 0)
        assert(radius <= min(design_x/2, design_y/2))
        radiusN = int(radius/dl)
        for i in range(radiusN):
            for j in range(radiusN):
                if np.sqrt(i**2 + j**2) <= radiusN:
                    design_mask[originx+i, originy+j] = True
                    design_mask[originx+i, originy-j] = True
                    design_mask[originx-i, originy+j] = True
                    design_mask[originx-i, originy-j] = True
        design_mask[background_mask] = False
    elif des_region == 'circlerand':
        originx = Npmlsep+Mx//2
        originy = Npmlsep+My//2
        radius = des_params[1]
        assert(radius >= 0)
        assert(radius <= min(design_x/2, design_y/2))
        radiusN = int(radius/dl)
        for i in range(radiusN):
            for j in range(radiusN):
                if np.sqrt(i**2 + j**2) <= radiusN:
                    design_mask[originx+i, originy+j] = True
                    design_mask[originx+i, originy-j] = True
                    design_mask[originx-i, originy+j] = True
                    design_mask[originx-i, originy-j] = True
        design_mask[background_mask] = False
        #pick a random design that fits within the design region
        design_mask = design_mask.flatten()
        des_ids = np.where(design_mask)
        des_ids = des_ids[0]
        for i in range(len(des_ids)):
            design_mask[des_ids[i]] = np.random.rand() > 0.5
        design_mask = np.reshape(design_mask, (nonpmlNx,nonpmlNy))
        
    if do_checks:
        config = np.zeros((nonpmlNx,nonpmlNy))
        config[design_mask] = 1.0
        plt.figure()
        plt.imshow(config)
        plt.savefig('check_design.png')
    
    if do_checks:
        config3 = np.zeros((nonpmlNx,nonpmlNy))
        config3[design_mask] = 1.0
        config3[background_mask] += 2.0
        plt.figure()
        plt.imshow(config3)
        plt.savefig('check_design+background.png')
    
    np.save(opt_data['name'] + '_L' + str(design_x)+ '_backgrounddof', background_mask[Npmlsep:Npmlsep+Mx, Npmlsep:Npmlsep+My]*1.0)
    
    background_chi = background_mask * chif #change from boolean mask of fixed object to complex numeric array with chi value
    Z = 1.0 # dimensionless units
    C_0 = 1.0
    eps0 = 1/Z/C_0
    
    big_background_chi = np.zeros((Nx, Ny), dtype=complex) #background chi over grid including pml region
    big_background_chi[Npml:-Npml, Npml:-Npml] = background_chi[:,:]
        
    ####Now do optimization inverse design
    ndof = np.sum(design_mask)
    if args.init_type=='vac':
        designdof = np.zeros(ndof)
    if args.init_type=='ones':
        designdof = np.ones(ndof)
    if args.init_type=='half':
        designdof = np.ones(ndof)/2.0
    if args.init_type=='rand':
        designdof = np.random.rand(ndof)
       
    inv_des_list = []
    Qabs_list = np.logspace(args.pow10Qabs_start, args.pow10Qabs_end, args.pow10Qabs_num)
    if int(args.dwfactor2) == 1:
        Qabs_list = Qabs_list/2.0
    Qabs_list = Qabs_list.tolist() + [np.inf]
    
    for Qabs in Qabs_list:
        if Qabs <= 6000: #reset design for large bandwidths
            ndof = np.sum(design_mask)
            if args.init_type=='vac':
                designdof = np.zeros(ndof)
            if args.init_type=='ones':
                designdof = np.ones(ndof)
            if args.init_type=='half':
                designdof = np.ones(ndof)/2.0
            if args.init_type=='rand':
                designdof = np.random.rand(ndof)
                
        #update omegas due to new Q value
        print('at Qabs', Qabs)
        opt_data['count'] = 0 #refresh the iteration count
        
        omega0 = 2*np.pi/wv
        omega = 2*np.pi/wv*(1 + 1j/2/Qabs)
        opt_data['name'] = args.name + f'_Qabs{Qabs:.1e}'
        if Qabs >= 1e16:
            omega = omega0
            opt_data['name'] = args.name + '_Qinf'
            
        #generate an initial plane wave from x side
        cx = Npml + Npmlsep//2 #position of current sheet
        #plane wave in vacuum
        Ei_cx = TM.get_TM_linesource_field(wv, dl, Nx, Ny, cx, Npml, bloch_x=0.0, bloch_y=0.0, amp=1.0, Qabs=Qabs, chigrid=None)
         
        #each Qabs has separate background term
        G0inv = TM.get_TM_MaxwellOp(wv, dl, Nx, Ny, Npml, bloch_x=0, bloch_y=0, Qabs=Qabs)
        Gfinv = G0inv + TM.get_diagM_from_chigrid(omega, big_background_chi)

        #we need |S1> = G_{0}^{-1} G_{f} V_{f} |E^i>
        S1_cx = big_background_chi*Ei_cx
        S1_cx = S1_cx.flatten()
        S1_cx = spla.spsolve(Gfinv, S1_cx)
        S1_cx = G0inv @ S1_cx
        EiS1_cx = np.vdot(Ei_cx, S1_cx) * dl**2
        background_term_cx = np.imag(EiS1_cx*omega/2/Z)
        print('background_term_cx: ', background_term_cx, flush=True)
        
        #generate an initial plane wave from y side
        cy = Npml + Npmlsep//2 #position of current sheet
        #plane wave in vacuum
        Ei_cy = TM.get_TM_linesource_field_cy(wv, dl, Nx, Ny, cy, Npml, bloch_x=0.0, bloch_y=0.0, amp=1.0, Qabs=Qabs, chigrid=None)
        
        #we need |S1> = G_{0}^{-1} G_{f} V_{f} |E^i>
        S1_cy = big_background_chi*Ei_cy
        S1_cy = S1_cy.flatten()
        S1_cy = spla.spsolve(Gfinv, S1_cy)
        S1_cy = G0inv @ S1_cy
        EiS1_cy = np.vdot(Ei_cy, S1_cy) * dl**2
        background_term_cy = np.imag(EiS1_cy*omega/2/Z)
        print('background_term_cy: ', background_term_cy, flush=True)
        
        background_term = background_term_cx + background_term_cy
        opt_data['background_term'] = background_term
        
        test_background_term = ext_objective(np.zeros(ndof), chif, background_mask, chid, design_mask, Npml, dl, cx, cy, omega0, Qabs)
        print('optfun test (equal to background_term?): ', test_background_term, flush=True)
        print('|test_background_term - background_term|/|background_term|', abs(test_background_term - opt_data['background_term'])/abs(opt_data['background_term']))

        co_term = ext_objective(np.ones(ndof), chif, background_mask, chid, design_mask, Npml, dl, cx, cy, omega0, Qabs)
        print('initial cloak+object: ', co_term, flush=True)
        
        optfun = lambda dof, dofgrad: designdof_ext_objective(dof, dofgrad, chif, background_mask, chid, design_mask, Npml, dl, cx, cy, omega0, Qabs, opt_data)

        lb = np.zeros(ndof)
        ub = np.ones(ndof)

        opt = nlopt.opt(nlopt.LD_MMA, int(ndof))
        #opt = nlopt.opt(nlopt.LD_LBFGS, int(ndof))
        #dof should be between 0 and 1. 0 means no chid at that pixel. 1 means chid at that pixel. allow grayscale.
        opt.set_lower_bounds(lb) 
        opt.set_upper_bounds(ub)

        opt.set_xtol_abs(args.xtol_abs)
        opt.set_maxeval(args.maxeval)
        opt.set_min_objective(optfun)

        designdof = opt.optimize(designdof) #optimize over designdof
        min_ext = opt.last_optimum_value() #get objective at such designof
        min_enh = min_ext / background_term
        print(f'Qabs{Qabs:.1e} best Pext and enhancement found via topology optimization', min_ext, min_enh)
        #print the binarized performance
        bin_obj = ext_objective(np.round(designdof), chif, background_mask, chid, design_mask, Npml, dl, cx, cy, omega0, Qabs)
        print(f'Qabs{Qabs:.1e} binarized opt result Pext and enhancement found via topology optimization', bin_obj, bin_obj/background_term)
        
        #save final opt design
        fulldesigndof = np.zeros_like(design_mask) * 1.0
        fulldesigndof[design_mask] = designdof[:]
        np.save(opt_data['name'] + '_L' + str(design_x) + '_optdof', fulldesigndof[Npmlsep:Npmlsep+Mx, Npmlsep:Npmlsep+My])
        if args.init_file != '':
            np.save(args.init_file, fulldesigndof[design_mask].flatten()) #update init file
            
        inv_des_list.append(min_enh)
        np.save(args.name+'_invdes_list', np.array(inv_des_list))
        np.save(args.name+'_Qabs', np.array(Qabs_list[0:len(inv_des_list)]))