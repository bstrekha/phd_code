import numpy as np
import csv
import sys,argparse
import os 
import pandas as pd 
import time

sys.path.append('../../../../../')
current_path = os.path.dirname(os.path.realpath(__file__))

# from Examples.MultiMat.cloaking.TM_cloak.bounds.sp_cloak_bound import sp_get_bandwidth_cloak_bound as get_bound
from Examples.MultiMat.cloaking.TM_cloak.bounds.sp_all_cloak_bound_gcd import sp_get_bandwidth_cloak_bound as get_bound

parser = argparse.ArgumentParser()

parser.add_argument('-chifRe', action='store', type=float, default=2.0)
parser.add_argument('-chifIm', action='store', type=float, default=0.01)
parser.add_argument('-chidRe', action='store', type=float, default=4.0)
parser.add_argument('-chidIm', action='store', type=float, default=0.0001)
parser.add_argument('-wavelength', action='store', type=float, default=1.0)
parser.add_argument('-des_region', action='store', type=str, default='circle') #options are 'rect' and 'circle'
parser.add_argument('-design_x', action='store', type=float, default=1.0)
parser.add_argument('-design_y', action='store', type=float, default=1.0)
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
parser.add_argument('-Pnum',action='store',type=int,default=10) # 5-15, the higher the faster convergence to local, but slower. If gcd_maxiter is high but you're still not converging to the local constraints value, Pnum should be higher. 
parser.add_argument('-gcd_maxiter',action='store',type=int,default=20) # between 10 and ~50, higher the better
parser.add_argument('-pml_sep',action='store',type=float,default=0.5)
parser.add_argument('-pml_thick',action='store',type=float,default=0.5)
parser.add_argument('-do_checks',action='store',type=bool,default=False)

args,unknown = parser.parse_known_args()
print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

save_func = None
des_params = [args.des_param0, args.des_param1]
if args.save:
    from Examples.saving_tools import hash_array

    def Plist_to_veclist(Plist):
        return [list(np.diag(P.todense())) for P in Plist]

    chif, chid = args.chifRe + 1j*args.chifIm, args.chidRe + 1j*args.chidIm

    def save_func(iternum, optdual, dualconst, optLags, optgrad, Plist, background_term):
        vecs_san = np.array(Plist_to_veclist(Plist))
        vecs_hash = hash_array(vecs_san)
        lags_hash = hash_array(optLags)
        print('hashes:', vecs_hash, lags_hash)
        bound = (background_term - optdual)/background_term

        to_save = np.array([args.obj, args.des_region, args.wavelength, args.Qabs, chif, chid, 
              des_params, np.round(args.design_x, 10), np.round(args.design_y, 10), args.gpr, args.Pnum, iternum, 
              optdual, dualconst, background_term, args.pml_sep, args.pml_thick, args.opttol, args.fakeSratio, args.iter_period, bound,
              lags_hash, vecs_hash], dtype=object)
        with open('results/gcd_cloak_results.csv', 'ab') as f:
            np.savetxt(f, to_save, fmt='%s;', newline='', delimiter=';')
            f.write(b"\n")
#         np.save(f'results/_vecs/{vecs_hash}.npy', vecs_san)
#         np.save(f'results/_lags/{lags_hash}.npy', optLags)

opt_params = {'opttol':args.opttol, 'fakeSratio': args.fakeSratio, 
              'iter_period': args.iter_period, 'Pnum': args.Pnum, 
              'gcd_maxiter': args.gcd_maxiter, 'save_func': save_func}

t1 = time.time() 
optdual, background_term = get_bound(args.chifRe + 1j*args.chifIm, args.chidRe + 1j*args.chidIm, args.wavelength, args.des_region, \
    des_params, args.obj, args.design_x, args.design_y, args.gpr, args.pml_sep, args.pml_thick, opt_params, Qabs=args.Qabs, \
	do_checks=args.do_checks)
print(f'Time taken: {time.time() - t1}')

print('Pext/Pext0 bound: ')
print((background_term-optdual)/background_term)


