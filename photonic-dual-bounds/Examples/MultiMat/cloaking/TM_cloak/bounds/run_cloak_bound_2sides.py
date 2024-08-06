import numpy as np
import csv
import sys,argparse
import os 
import pandas as pd 
import time

sys.path.append('../../../../../')
current_path = os.path.dirname(os.path.realpath(__file__))

# from Examples.MultiMat.cloaking.TM_cloak.bounds.sp_cloak_bound import sp_get_bandwidth_cloak_bound as get_bound
from Examples.MultiMat.cloaking.TM_cloak.bounds.sp_all_cloak_bound_2sides import sp_get_bandwidth_cloak_bound_2sides as get_bound

parser = argparse.ArgumentParser()

parser.add_argument('-chifRe', action='store', type=float, default=-10.0)
parser.add_argument('-chifIm', action='store', type=float, default=0.1)
parser.add_argument('-chidRe', action='store', type=float, default=12.0)
parser.add_argument('-chidIm', action='store', type=float, default=0.01)
parser.add_argument('-des', action='store', type=float, default=1.0)
parser.add_argument('-wavelengthList', nargs='*', action='store', type=float, default=[1.0, 1.0])
parser.add_argument('-nmat', action='store', type=int, default=1)
parser.add_argument('-cons', nargs=2, action='store', type=int, default=[1, 1])
parser.add_argument('-des_region', action='store', type=str, default='circle') #options are 'rect' and 'circle'
parser.add_argument('-design_x', action='store', type=float, default=0.7)
parser.add_argument('-design_y', action='store', type=float, default=0.7)
#parser.add_argument('-des_params', nargs='*', action='store', type=float, default=[0.25,0.5]) #inner radius and outer radius of circle design region
parser.add_argument('-des_param0', action='store', type=float, default=0.25) #inner radius of circle design region
parser.add_argument('-des_param1', action='store', type=float, default=0.35) #outer radius of circle design region
parser.add_argument('-gpr', action='store', type=int, default=30)
parser.add_argument('-obj', action='store', type=str, default='EXT')
parser.add_argument('-nprojx',action='store',type=int,default=1)
parser.add_argument('-nprojy',action='store',type=int,default=1)
parser.add_argument('-save',action='store',type=int,default=0)
parser.add_argument('-opttol', action='store', type=float, default=1e-4)
parser.add_argument('-Qabs', action='store', type=float, default=50)
parser.add_argument('-fakeSratio', action='store', type=float, default=1e-3)
parser.add_argument('-iter_period', action='store', type=int, default=80)
parser.add_argument('-opttype', action='store', type=str, default='newton')
parser.add_argument('-pml_sep',action='store',type=float,default=0.5)
parser.add_argument('-pml_thick',action='store',type=float,default=0.5)
parser.add_argument('-do_checks',action='store',type=bool,default=True)

args,unknown = parser.parse_known_args()

print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

init_lags = None

# if (args.nprojx > 1):
#     res = pd.read_csv('results/sp_do_results.csv', sep=';', index_col=False)
#     consstring = '[' + ', '.join([str(x) for x in args.cons]) + ']'
#     wvstring = '[' + ', '.join([str(x) for x in args.wavelengthList]) + ']'
#     res = res[(res['obj'] == args.obj) & (res['des_region'] == args.des_region) & (res['nmat'] == args.nmat) & (res['wavelengths'] == wvstring) &
#               (res['des_params'] == args.des_params) & (res['gpr'] == args.gpr) & (res['nprojx'] == 1.0) &
#               (res['nprojy'] == 1.0) & (res['dx'] == args.design_x) & (res['dy'] == args.design_y) & 
#               (res['cons'] == consstring)]

#     try:
#         res = res[res['bound'] == res['bound'].min()]
#         print(f"current bound: {res['bound'].values}")
#         lags = np.array([eval(y) for y in res['lags'].values[0][1:-1].replace(',', '').split()])
#         init_lags = lags
#     except:
#         print(f"no previous run found for these parameters.")
#         exit()

des_params = [args.des_param0, args.des_param1]
t1 = time.time() 
wavelengthList = [1.0, 1.0]
bound, lags, violations, conv, nprojx, nprojy = get_bound(args.chifRe + 1j*args.chifIm, args.chidRe + 1j*args.chidIm, wavelengthList, args.nmat, args.cons, args.des_region, \
    des_params, args.obj, args.design_x, args.design_y, args.gpr, args.nprojx, args.nprojy, args.pml_sep, args.pml_thick, Qabs=args.Qabs, opttol=args.opttol, \
	fakeSratio=args.fakeSratio, iter_period=args.iter_period, opttype=args.opttype, init_lags=init_lags, do_checks=args.do_checks)
print(f'Time taken: {time.time() - t1}')

print('Pext/Pext0 bound: ')
print(bound)

# if args.save:
#     nsource = len(args.wavelengthList)
#     nmat = args.nmat
#     wv = args.wavelengthList
#     wv = wv[0]
#     chif = args.chifRe + 1j*args.chifIm
#     chid = args.chidRe + 1j*args.chidIm

#     to_save = np.array([args.obj, args.des_region, wv, args.Qabs, chif, chid, \
#               args.des_params, np.round(args.design_x, 10), np.round(args.design_y, 10), args.gpr, nprojx, nprojy, \
#               args.pml_sep, args.pml_thick, args.opttol, args.fakeSratio, args.iter_period, args.opttype, bound, lags.tolist(), violations.tolist(), conv], dtype=object)

#     with open('results/sp_do_results.csv', 'ab') as f:
#         np.savetxt(f, to_save, fmt='%s;', newline='', delimiter=';')
#         f.write(b"\n")

