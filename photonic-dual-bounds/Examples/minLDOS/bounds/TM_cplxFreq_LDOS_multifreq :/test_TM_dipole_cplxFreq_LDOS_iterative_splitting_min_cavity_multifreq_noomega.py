import numpy as np

import time,sys,argparse

sys.path.append('../../')

from get_TM_dipole_cplxFreq_LDOS_min_cavity_multifreq_noomega import get_TM_dipole_oneside_ldos_Msparse_iterative_splitting


parser = argparse.ArgumentParser()

parser.add_argument('-wavelength', action='store', type=float, default=1.0)
parser.add_argument('-Qabs', action='store', type=float, default=100)
parser.add_argument('-Qabstol', action='store', type=float, default=100)
parser.add_argument('-ReChi', action='store', type=float, default=2.0)
parser.add_argument('-ImChi', action='store', type=float, default=0)

parser.add_argument('-gpr', action='store', type=int, default=20)
parser.add_argument('-design_x', action='store', type=float, default=1.0)
parser.add_argument('-design_y', action='store', type=float, default=1.0)
parser.add_argument('-vacuum_x', action='store', type=float, default=1.0)
parser.add_argument('-vacuum_y', action='store', type=float, default=1.0)
parser.add_argument('-emitter_x', action='store', type=float, default=1.0)
parser.add_argument('-emitter_y', action='store', type=float, default=1.0)
parser.add_argument('-dist', action='store', type=float, default=0.5)
parser.add_argument('-Num_Poles', action='store', type=int, default=1)
parser.add_argument('-geometry', action='store', type=str, default='Cavity')

parser.add_argument('-pml_sep',action='store',type=float,default=0.5)
parser.add_argument('-pml_thick',action='store',type=float,default=0.5)

parser.add_argument('-alg',action='store',type=str,default='Newton')
parser.add_argument('-lag_input',action='store',type=str,default='Lags_input.txt')

args,unknown = parser.parse_known_args()
print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

chi = args.ReChi + 1j*args.ImChi

Prad_enh = get_TM_dipole_oneside_ldos_Msparse_iterative_splitting(chi, args.wavelength, args.Qabs, args.design_x, args.design_y, args.vacuum_x, args.vacuum_y, args.emitter_x, args.emitter_y, args.dist, args.pml_sep, args.pml_thick, args.gpr, alg=args.alg, Lag_input=args.lag_input, Qabstol=args.Qabstol, Geometry=args.geometry, Num_Poles=args.Num_Poles)

print('bound for Prad enhancement', Prad_enh)
