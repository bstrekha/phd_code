import numpy as np
import csv
import sys,argparse
import os 
import pandas as pd 
import time

sys.path.append('../../../../../')
current_path = os.path.dirname(os.path.realpath(__file__))

from Examples.MultiMat.cloaking.TM_cloak.bounds.sp_all_cloak_bound_2sides import sp_get_bandwidth_cloak_bound_2sides as get_bound

parser = argparse.ArgumentParser()

parser.add_argument('-chifRe', action='store', type=float, default=2.0)
parser.add_argument('-chifIm', action='store', type=float, default=0.01)
parser.add_argument('-chidRe', action='store', type=float, default=4.0)
parser.add_argument('-chidIm', action='store', type=float, default=0.0001)
parser.add_argument('-wavelength', action='store', type=float, default=1.0)
parser.add_argument('-des_region', action='store', type=str, default='circle') #options are 'rect' and 'circle'
parser.add_argument('-design_x', action='store', type=float, default=1.0)
parser.add_argument('-design_y', action='store', type=float, default=1.0)
parser.add_argument('-nprojx',action='store',type=int,default=1)
parser.add_argument('-nprojy',action='store',type=int,default=1)
#parser.add_argument('-des_params', nargs='*', action='store', type=float, default=[0.25,0.5]) #inner radius and outer radius of circle design region
parser.add_argument('-des_param0', action='store', type=float, default=0.25) #inner radius of circle design region
parser.add_argument('-des_param1', action='store', type=float, default=0.5) #outer radius of circle design region
parser.add_argument('-gpr', action='store', type=int, default=20)
parser.add_argument('-obj', action='store', type=str, default='EXT')
parser.add_argument('-save',action='store',type=int,default=1)
parser.add_argument('-opttol', action='store', type=float, default=1e-4)
parser.add_argument('-Qabs', action='store', type=float, default=np.inf)
parser.add_argument('-fakeSratio', action='store', type=float, default=1e-3)
parser.add_argument('-iter_period', action='store', type=int, default=80)
parser.add_argument('-opttype', action='store', type=str, default='bfgs')
parser.add_argument('-pml_sep',action='store',type=float,default=0.5)
parser.add_argument('-pml_thick',action='store',type=float,default=0.5)
parser.add_argument('-do_checks',action='store',type=bool,default=False)
parser.add_argument('-normalize_result', action='store', type=int, default=1) #0 = False, 1 = True
parser.add_argument('-nmat', action='store', type=int, default=1)
parser.add_argument('-cons', nargs=2, action='store', type=int, default=[1, 1])
parser.add_argument('-angle_y', action='store', type=float, default=0.0)

args,unknown = parser.parse_known_args()
print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

save_func = None
des_params = [args.des_param0, args.des_param1]
wavelengthList = [1.0, 1.0]

t1 = time.time() 

bound, lags, violations, conv, nprojx, nprojy = get_bound(args.chifRe + 1j*args.chifIm, args.chidRe + 1j*args.chidIm, wavelengthList, args.nmat, args.cons, args.des_region, \
            [args.des_param0, args.des_param1], args.obj, args.design_x, args.design_y, args.gpr, args.nprojx, args.nprojy, args.pml_sep, args.pml_thick, Qabs=args.Qabs, opttol=args.opttol, \
        	fakeSratio=args.fakeSratio, iter_period=args.iter_period, opttype=args.opttype, do_checks=args.do_checks, normalize_result=args.normalize_result, angle_y=args.angle_y)

print(f'Time taken: {time.time() - t1}')

print('Pext/Pext0 bound: ')
print(bound)

#save the bound calculated in the current single run to an csv file
if args.save:
    chif, chid = args.chifRe + 1j*args.chifIm, args.chidRe + 1j*args.chidIm
    to_save = np.array([args.obj, args.des_region, args.wavelength, args.Qabs, chif, chid, 
          des_params, np.round(args.design_x, 10), np.round(args.design_y, 10), args.nprojx, args.nprojy, args.gpr, 
          args.pml_sep, args.pml_thick, args.opttol, args.fakeSratio, args.iter_period, bound, args.angle_y], dtype=object)
    with open('results/parallel_2sides_angle_cloak_results.csv', 'ab') as f:
        np.savetxt(f, to_save, fmt='%s;', newline='', delimiter=';')
        f.write(b"\n")    
    


