import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time,sys,argparse
from get_spectrum_cloak_performance_TM_EXTmin_planewave import get_spectrum_of_cloakobj

parser = argparse.ArgumentParser()
parser.add_argument('-wavelength', action='store', type=float, default=1.0)

parser.add_argument('-Qabs', action='store', type=float, default=np.inf)
parser.add_argument('-omega_pts', action='store', type=int, default=501)

parser.add_argument('-chifRe', action='store', type=float, default=2.0)
parser.add_argument('-chifIm', action='store', type=float, default=0.01)
parser.add_argument('-chidRe', action='store', type=float, default=4.0)
parser.add_argument('-chidIm', action='store', type=float, default=0.0001)

parser.add_argument('-design_x', action='store', type=float, default=2.0)
parser.add_argument('-design_y', action='store', type=float, default=2.0)
parser.add_argument('-gpr', action='store', type=int, default=50)

parser.add_argument('-pml_sep', action='store', type=float, default=0.5)
parser.add_argument('-include_cloak', action='store', type=int, default=1)

parser.add_argument('-design_file', action='store', type=str, default='test')
parser.add_argument('-background_file', action='store', type=str, default='test')
parser.add_argument('-save_name', action='store', type=str, default='test')

args,unknown = parser.parse_known_args()
print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

omegas = (np.linspace(-1e6/2/10**(6),1e6/2/10**(6), int(args.omega_pts))+1)*2*np.pi

obj_list = []
    
for w in omegas:
    obj = get_spectrum_of_cloakobj(np.inf, args.chifRe + 1j*args.chifIm, args.chidRe + 1j*args.chidIm, args.design_x, args.design_y, args.gpr, args.design_file, args.background_file, args.save_name, pml_sep=1.0, binarize_des=False, omega0=w)
    obj_list.append(obj)
    print('P_ext/P_ext0: ', obj, flush=True)

np.save(args.save_name + '_spectrum', np.array(obj_list))
np.save(args.save_name + '_spectrum_omegas', omegas)