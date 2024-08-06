import numpy as np
import csv
import sys, argparse
import os 

sys.path.append('../../../../../')
current_path = os.path.dirname(os.path.realpath(__file__))

from Examples.MultiMat.cloaking.TM_cloak.bounds.sp_all_cloak_bound_2sides import sp_get_bandwidth_cloak_bound_2sides as get_bound

parser = argparse.ArgumentParser()

parser.add_argument('-chifRe', action='store', type=float, default=2.0)
parser.add_argument('-chifIm', action='store', type=float, default=0.01)
parser.add_argument('-chidRe', action='store', type=float, default=4.0)
parser.add_argument('-chidIm', action='store', type=float, default=0.0001)
parser.add_argument('-wavelengthList', nargs='*', action='store', type=float, default=[1.0])
parser.add_argument('-nmat', action='store', type=int, default=1)
parser.add_argument('-cons', nargs=2, action='store', type=int, default=[1, 1])
parser.add_argument('-des_region', action='store', type=str, default='circle') #options are 'rect' and 'circle'
parser.add_argument('-design_x', action='store', type=float, default=1.0)
parser.add_argument('-design_y', action='store', type=float, default=1.0)
parser.add_argument('-des_param1', action='store', type=float, default=0.25) #inner radius of circle design region
parser.add_argument('-des_param2', action='store', type=float, default=0.5) #outer radius of circle design region
parser.add_argument('-gpr', action='store', type=int, default=30)
parser.add_argument('-obj', action='store', type=str, default='EXT')
parser.add_argument('-nprojx',action='store',type=int,default=1)
parser.add_argument('-nprojy',action='store',type=int,default=1)
parser.add_argument('-all_local_constraints',action='store',type=int,default=0) #1=True, 0=False and use nprojx, nprojy
parser.add_argument('-save',action='store',type=int,default=1)
parser.add_argument('-opttol', action='store', type=float, default=1e-3)
parser.add_argument('-Qabs', action='store', type=float, default=np.inf)
parser.add_argument('-fakeSratio', action='store', type=float, default=1e-3)
parser.add_argument('-iter_period', action='store', type=int, default=100)
parser.add_argument('-opttype', action='store', type=str, default='bfgs')
parser.add_argument('-pml_sep',action='store',type=float,default=0.5)
parser.add_argument('-pml_thick',action='store',type=float,default=0.5)
parser.add_argument('-do_checks',action='store',type=bool,default=True)
parser.add_argument('-name',action='store',type=str,default='test')
parser.add_argument('-sweep_type', action='store', type=str, default='Q')
parser.add_argument('-normalize_result', action='store', type=int, default=1) #0 = False, 1 = True

parser.add_argument('-pow10Qabs_start',action='store',type=float,default=2.0)
parser.add_argument('-pow10Qabs_end',action='store',type=float,default=7.0)
parser.add_argument('-pow10Qabs_num',action='store',type=int,default=11)
parser.add_argument('-dwfactor2',action='store',type=int,default=1) #1 = True, 0 = False for diving Qabs list by 2

parser.add_argument('-maxratioRouterRinner',action='store',type=float,default=8.0)
parser.add_argument('-numRsweep',action='store',type=int,default=9)

parser.add_argument('-Rinner_start',action='store',type=float,default=0.25)
parser.add_argument('-Rinner_end',action='store',type=float,default=2.0)

parser.add_argument('-chifRe_start',action='store',type=float,default=1.0)
parser.add_argument('-chifRe_end',action='store',type=float,default=15.0)
parser.add_argument('-chifRe_num',action='store',type=int,default=22)

parser.add_argument('-chifIm_start',action='store',type=float,default=0.0001)
parser.add_argument('-chifIm_end',action='store',type=float,default=1.0)
parser.add_argument('-chifIm_num',action='store',type=int,default=22)

parser.add_argument('-chidRe_start',action='store',type=float,default=1.0)
parser.add_argument('-chidRe_end',action='store',type=float,default=15.0)
parser.add_argument('-chidRe_num',action='store',type=int,default=22)

parser.add_argument('-chidIm_start',action='store',type=float,default=0.0001)
parser.add_argument('-chidIm_end',action='store',type=float,default=1.0)
parser.add_argument('-chidIm_num',action='store',type=int,default=22)


args,unknown = parser.parse_known_args()

print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

init_lags = None

bounds_list = []

if args.all_local_constraints == 1:
    local_nprojx = np.inf
    local_nprojy = np.inf
else:
    local_nprojx = args.nprojx
    local_nprojy = args.nprojy
    
if args.normalize_result == 1:
    normalize_result = True
else:
    normalize_result = False

global_lags = None

if args.sweep_type == 'Q':
    Qabs_list = np.logspace(args.pow10Qabs_start, args.pow10Qabs_end, args.pow10Qabs_num)
    if int(args.dwfactor2) == 1:
        Qabs_list *= 1/2.0
    Qabs_list = Qabs_list.tolist()
    #if args.nprojx > 1 or args.nprojy > 1 or args.all_local_constraints == 1:
        #first get global lags
#         for Qabs in Qabs_list[0:1]:
#             bound, global_lags, violations, conv, nprojx, nprojy = get_bound(args.chifRe + 1j*args.chifIm, args.chidRe + 1j*args.chidIm, wavelengthList, args.nmat, args.cons, args.des_region, \
#             [args.des_param1, args.des_param2], args.obj, args.design_x, args.design_y, args.gpr, 1, 1, args.pml_sep, args.pml_thick, Qabs=Qabs, opttol=args.opttol, \
#         	fakeSratio=args.fakeSratio, iter_period=args.iter_period, opttype=args.opttype, init_lags=init_lags, do_checks=args.do_checks, normalize_result=normalize_result)
            
    #now can do general runs if 
    wavelengthList = [1.0, 1.0]
    for Qabs in Qabs_list:
        if args.nprojx > 1 or args.nprojy > 1 or args.all_local_constraints == 1:
            bound, global_lags, violations, conv, nprojx, nprojy = get_bound(args.chifRe + 1j*args.chifIm, args.chidRe + 1j*args.chidIm, wavelengthList, args.nmat, args.cons, args.des_region, \
                [args.des_param1, args.des_param2], args.obj, args.design_x, args.design_y, args.gpr, 1, 1, args.pml_sep, args.pml_thick, Qabs=Qabs, opttol=args.opttol, \
                fakeSratio=args.fakeSratio, iter_period=args.iter_period, opttype=args.opttype, init_lags=init_lags, do_checks=args.do_checks, normalize_result=normalize_result)
        #now use global run for local run
        bound, lags, violations, conv, nprojx, nprojy = get_bound(args.chifRe + 1j*args.chifIm, args.chidRe + 1j*args.chidIm, wavelengthList, args.nmat, args.cons, args.des_region, \
            [args.des_param1, args.des_param2], args.obj, args.design_x, args.design_y, args.gpr, local_nprojx, local_nprojy, args.pml_sep, args.pml_thick, Qabs=Qabs, opttol=args.opttol, \
        	fakeSratio=args.fakeSratio, iter_period=args.iter_period, opttype=args.opttype, init_lags=init_lags, do_checks=args.do_checks, init_global_lags=global_lags, normalize_result=normalize_result)
        
        bounds_list.append(bound)
        print(f'At Qabs {Qabs} Pext/Pext0 bound = {bound}', flush=True)
        np.save(args.name+'_bounds_vs_Q', np.array(bounds_list))
        np.save(args.name+'_Qs', np.array(Qabs_list[0:len(bounds_list)]))
        
if args.sweep_type == 'Router':
    Qabs_val = args.Qabs
    if Qabs_val > 1e15:
        Qabs_val = np.inf
        
    R_list = args.des_param1*np.geomspace(1.0, args.maxratioRouterRinner, args.numRsweep)
    R_list = R_list[1:] #used when using np.geomspace(1.0, args.maxratioRouterRinner, args.numRsweep)
    print('R_list: ', R_list, flush=True)
    for Router in R_list:
        print(f'starting Router = {Router} run', flush=True)
        design_x = Router*2
        design_y = Router*2
        global_lags = None
        gpr_val = args.gpr
#         if Router/args.des_param1 < 2.0:
#             gpr_val *= 2
        if args.nprojx > 1 or args.nprojy > 1 or args.all_local_constraints == 1:
            bound, global_lags, violations, conv, nprojx, nprojy = get_bound(args.chifRe + 1j*args.chifIm, args.chidRe + 1j*args.chidIm, wavelengthList, args.nmat, args.cons, args.des_region, \
            [args.des_param1, Router], args.obj, design_x, design_y, gpr_val, 1, 1, args.pml_sep, args.pml_thick, Qabs=Qabs_val, opttol=args.opttol, \
        	fakeSratio=args.fakeSratio, iter_period=args.iter_period, opttype=args.opttype, init_lags=init_lags, do_checks=args.do_checks, init_global_lags=global_lags, normalize_result=normalize_result)
            
        #now do use potential global one for potential local one
        bound, lags, violations, conv, nprojx, nprojy = get_bound(args.chifRe + 1j*args.chifIm, args.chidRe + 1j*args.chidIm, wavelengthList, args.nmat, args.cons, args.des_region, \
            [args.des_param1, Router], args.obj, design_x, design_y, gpr_val, local_nprojx, local_nprojy, args.pml_sep, args.pml_thick, Qabs=Qabs_val, opttol=args.opttol, \
        	fakeSratio=args.fakeSratio, iter_period=args.iter_period, opttype=args.opttype, init_lags=init_lags, do_checks=args.do_checks, init_global_lags=global_lags, normalize_result=normalize_result)
        
        bounds_list.append(bound)
        print(f'At Router {Router} Pext/Pext0 bound = {bound}', flush=True)
        np.save(args.name+'_bounds_vs_Router', np.array(bounds_list))
        np.save(args.name+'_Router', np.array(R_list[0:len(bounds_list)]))
        
if args.sweep_type == 'Rinner':
    R_list = np.geomspace(args.Rinner_start, args.Rinner_end, args.numRsweep)
    print('R_list: ', R_list, flush=True)
    for Rinner in R_list:
        print(f'starting Rinner = {Rinner} run', flush=True)
        Router = 2*Rinner
        design_x = Router*2
        design_y = Router*2
        bound, lags, violations, conv, nprojx, nprojy = get_bound(args.chifRe + 1j*args.chifIm, args.chidRe + 1j*args.chidIm, wavelengthList, args.nmat, args.cons, args.des_region, \
            [Rinner, Router], args.obj, design_x, design_y, args.gpr, local_nprojx, local_nprojy, args.pml_sep, args.pml_thick, Qabs=args.Qabs, opttol=args.opttol, \
        	fakeSratio=args.fakeSratio, iter_period=args.iter_period, opttype=args.opttype, init_lags=init_lags, do_checks=args.do_checks, normalize_result=normalize_result)
        
        bounds_list.append(bound)
        print(f'At Rinner {Rinner} Pext/Pext0 bound = {bound}', flush=True)
        np.save(args.name+'_bounds_vs_Rinner', np.array(bounds_list))
        np.save(args.name+'_Rinner', np.array(R_list[0:len(bounds_list)]))
        
if args.sweep_type == 'chifRe':
    chifRe_list = np.geomspace(args.chifRe_start, args.chifRe_end, args.chifRe_num)
    print('chifRe_list: ', chifRe_list, flush=True)
    for chifRe in chifRe_list:
        print(f'starting chifRe = {chifRe} run', flush=True)
        bound, lags, violations, conv, nprojx, nprojy = get_bound(chifRe + 1j*args.chifIm, args.chidRe + 1j*args.chidIm, wavelengthList, args.nmat, args.cons, args.des_region, \
            [args.des_param1, args.des_param2], args.obj, args.design_x, args.design_y, args.gpr, local_nprojx, local_nprojy, args.pml_sep, args.pml_thick, Qabs=args.Qabs, opttol=args.opttol, \
        	fakeSratio=args.fakeSratio, iter_period=args.iter_period, opttype=args.opttype, init_lags=init_lags, do_checks=args.do_checks, normalize_result=normalize_result)
        
        bounds_list.append(bound)
        print(f'At chifRe {chifRe} Pext/Pext0 bound = {bound}', flush=True)
        np.save(args.name+'_bounds_vs_chifRe', np.array(bounds_list))
        np.save(args.name+'_chifRe', np.array(chifRe_list[0:len(bounds_list)]))
        
if args.sweep_type == 'chifIm':
    chifIm_list = np.geomspace(args.chifIm_start, args.chifIm_end, args.chifIm_num)
    print('chifIm_list: ', chifIm_list, flush=True)
    for chifIm in chifIm_list:
        print(f'starting chifIm = {chifIm} run', flush=True)
        bound, lags, violations, conv, nprojx, nprojy = get_bound(args.chifRe + 1j*chifIm, args.chidRe + 1j*args.chidIm, wavelengthList, args.nmat, args.cons, args.des_region, \
            [args.des_param1, args.des_param2], args.obj, args.design_x, args.design_y, args.gpr, local_nprojx, local_nprojy, args.pml_sep, args.pml_thick, Qabs=args.Qabs, opttol=args.opttol, \
        	fakeSratio=args.fakeSratio, iter_period=args.iter_period, opttype=args.opttype, init_lags=init_lags, do_checks=args.do_checks, normalize_result=normalize_result)
        
        bounds_list.append(bound)
        print(f'At chifIm {chifIm} Pext/Pext0 bound = {bound}', flush=True)
        np.save(args.name+'_bounds_vs_chifIm', np.array(bounds_list))
        np.save(args.name+'_chifIm', np.array(chifIm_list[0:len(bounds_list)]))
         
if args.sweep_type == 'chidRe':
    chidRe_list = np.geomspace(args.chidRe_start, args.chidRe_end, args.chidRe_num)
    print('chidRe_list: ', chidRe_list, flush=True)
    for chidRe in chidRe_list:
        print(f'starting chidRe = {chidRe} run', flush=True)
        bound, lags, violations, conv, nprojx, nprojy = get_bound(args.chifRe + 1j*args.chifIm, chidRe + 1j*args.chidIm, wavelengthList, args.nmat, args.cons, args.des_region, \
            [args.des_param1, args.des_param2], args.obj, args.design_x, args.design_y, args.gpr, local_nprojx, local_nprojy, args.pml_sep, args.pml_thick, Qabs=args.Qabs, opttol=args.opttol, \
        	fakeSratio=args.fakeSratio, iter_period=args.iter_period, opttype=args.opttype, init_lags=init_lags, do_checks=args.do_checks, normalize_result=normalize_result)
        
        bounds_list.append(bound)
        print(f'At chidRe {chidRe} Pext/Pext0 bound = {bound}', flush=True)
        np.save(args.name+'_bounds_vs_chidRe', np.array(bounds_list))
        np.save(args.name+'_chidRe', np.array(chidRe_list[0:len(bounds_list)]))
        
if args.sweep_type == 'chidIm':
    chidIm_list = np.geomspace(args.chidIm_start, args.chidIm_end, args.chidIm_num)
    print('chidIm_list: ', chidIm_list, flush=True)
    for chidIm in chidIm_list:
        print(f'starting chidIm = {chidIm} run', flush=True)
        bound, lags, violations, conv, nprojx, nprojy = get_bound(args.chifRe + 1j*args.chifIm, args.chidRe + 1j*chidIm, wavelengthList, args.nmat, args.cons, args.des_region, \
            [args.des_param1, args.des_param2], args.obj, args.design_x, args.design_y, args.gpr, local_nprojx, local_nprojy, args.pml_sep, args.pml_thick, Qabs=args.Qabs, opttol=args.opttol, \
        	fakeSratio=args.fakeSratio, iter_period=args.iter_period, opttype=args.opttype, init_lags=init_lags, do_checks=args.do_checks, normalize_result=normalize_result)
        
        bounds_list.append(bound)
        print(f'At chidIm {chidIm} Pext/Pext0 bound = {bound}', flush=True)
        np.save(args.name+'_bounds_vs_chidIm', np.array(bounds_list))
        np.save(args.name+'_chidIm', np.array(chidIm_list[0:len(bounds_list)]))