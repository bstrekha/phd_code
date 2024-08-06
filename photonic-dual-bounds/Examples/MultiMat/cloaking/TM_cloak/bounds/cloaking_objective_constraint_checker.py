import sys,argparse
sys.path.append('../../../../../')
import numpy as np
import matplotlib.pyplot as plt
from Examples.MultiMat.cloaking.TM_cloak.bounds.sp_cloak_bound import sp_get_bandwidth_cloak_bound as get_bound
from dualbound.Maxwell import TM_FDFD as TM
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import dualbound.Lagrangian.multimat.zops_multimat_sp as zmultimat

from objective_TM_ext import ext_objective

parser = argparse.ArgumentParser()

parser.add_argument('-ReChiList', nargs='*', action='store', type=float, default=[4.0]) # Format: S1M1, S1M2, ... , S1Mn, S2M1, ... , SnsMn
parser.add_argument('-ImChiList', nargs='*', action='store', type=float, default=[1e-1])
parser.add_argument('-chifRe', action='store', type=float, default=4.0)
parser.add_argument('-chifIm', action='store', type=float, default=0.001)
parser.add_argument('-chidRe', action='store', type=float, default=2.0)
parser.add_argument('-chidIm', action='store', type=float, default=0.001)
parser.add_argument('-des', action='store', type=float, default=1.0)
parser.add_argument('-wavelengthList', nargs='*', action='store', type=float, default=[1.0])
parser.add_argument('-nmat', action='store', type=int, default=1)
parser.add_argument('-cons', nargs=2, action='store', type=int, default=[1, 1])
parser.add_argument('-des_region', action='store', type=str, default='circlerand') #options are 'rect' and 'circle' (and 'circlerand', intended for rand tests)
parser.add_argument('-design_x', action='store', type=float, default=1.0)
parser.add_argument('-design_y', action='store', type=float, default=1.0)
parser.add_argument('-des_params', nargs='*', action='store', type=float, default=[0.25,0.5])
parser.add_argument('-gpr', action='store', type=int, default=50)
parser.add_argument('-obj', action='store', type=str, default='EXT')
parser.add_argument('-nprojx',action='store',type=int,default=1)
parser.add_argument('-nprojy',action='store',type=int,default=1)
parser.add_argument('-save',action='store',type=int,default=1)
parser.add_argument('-opttol', action='store', type=float, default=1e-4)
parser.add_argument('-Qabs', action='store', type=float, default=5e3)
parser.add_argument('-fakeSratio', action='store', type=float, default=1e-3)
parser.add_argument('-iter_period', action='store', type=int, default=80)
parser.add_argument('-opttype', action='store', type=str, default='newton')

parser.add_argument('-pml_sep',action='store',type=float,default=0.5)
parser.add_argument('-pml_thick',action='store',type=float,default=0.5)

args,unknown = parser.parse_known_args()
assert(args.nmat*len(args.wavelengthList) == len(args.ReChiList))

print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

chilist = np.array(args.ReChiList) + np.array(args.ImChiList)*1j
print(f"Number of materials: {args.nmat}")

init_lags = None

design_mask, background_mask, nonpmlNx, nonpmlNy, Npml, Npmlsep, dl = get_bound(args.chifRe + 1j*args.chifIm, args.chidRe + 1j*args.chidIm, args.wavelengthList, args.nmat, args.cons, args.des_region, \
    args.des_params, args.obj, args.design_x, args.design_y, args.gpr, args.nprojx, args.nprojy, args.pml_sep, args.pml_thick, Qabs=args.Qabs, opttol=args.opttol, \
	fakeSratio=args.fakeSratio, iter_period=args.iter_period, opttype=args.opttype, init_lags=init_lags, TESTS={'just_mask': True})

Nx = nonpmlNx + 2*Npml
Ny = nonpmlNy + 2*Npml
wv = 1.0
Qabs = args.Qabs
omega0 = 2*np.pi/wv
omega = 2*np.pi/wv * (1 + 1j/2/Qabs)
cx = Npml + Npmlsep//2 #position of current sheet
chif = args.chifRe + 1j*args.chifIm
chid = args.chidRe + 1j*args.chidIm
#chif = 5*np.random.rand() + 1j*np.random.rand()
#chid = 10*np.random.rand() + 1j*np.random.rand()
Ei = TM.get_TM_linesource_field(wv, dl, Nx, Ny, cx, Npml, bloch_x=0.0, bloch_y=0.0, amp=1.0, Qabs=Qabs, chigrid=None) #plane wave in vacuum
print('calculated Ei', flush=True)
do_checks = True

################# calculate the fixed vectors #################
Z = 1.0 # dimensionless units
C_0 = 1.0
eps0 = 1/Z/C_0

if do_checks:
    config = np.zeros((nonpmlNx,nonpmlNy))
    config[design_mask] = 1.0
    plt.imshow(config)
    plt.savefig('check_design.png')

if do_checks:
    config3 = np.zeros((nonpmlNx,nonpmlNy))
    config3[design_mask] = 1.0
    config3[background_mask] += 2.0
    plt.imshow(config3)
    plt.savefig('check_design+background.png')

background_chi = background_mask * chif

print(f"Making Gdd for lambda={wv}", flush=True)
Gfddinv, _ = TM.get_Gddinv(wv, dl, nonpmlNx, nonpmlNy, (Npml, Npml), design_mask, Qabs=Qabs, chigrid=background_chi)
N = Gfddinv.shape[0] #number of grid points in design region

big_background_chi = np.zeros((Nx, Ny), dtype=complex) #background chi over grid including pml region
big_background_chi[Npml:-Npml, Npml:-Npml] = background_chi[:,:]
    
big_design_mask = np.zeros((Nx, Ny), dtype=bool)
big_design_mask[Npml:-Npml, Npml:-Npml] = design_mask[:,:] #design mask over grid including pml region

#generate an initial plane wave.
cx = Npml + Npmlsep//2 #position of current sheet
Ei = TM.get_TM_linesource_field(wv, dl, Nx, Ny, cx, Npml, bloch_x=0.0, bloch_y=0.0, amp=1.0, Qabs=Qabs, chigrid=None) #plane wave in vacuum

if do_checks:
    #should look like planewave
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(Ei), cmap='RdBu')
    ax2.imshow(np.imag(Ei), cmap='RdBu')
    plt.savefig('test_TM_planewave_Ei.png')

G0inv = TM.get_TM_MaxwellOp(wv, dl, Nx, Ny, Npml, bloch_x=0, bloch_y=0, Qabs=Qabs)
Gfinv = G0inv + TM.get_diagM_from_chigrid(omega, big_background_chi)

#we need |S1> = G_{0}^{-1} G_{f} V_{f} |E^i>
S1 = big_background_chi*Ei
S1 = S1.flatten()
S1 = spla.spsolve(Gfinv, S1)
S1 = G0inv @ S1
EiS1 = np.vdot(Ei, S1) * dl**2
background_term = np.imag(EiS1*omega/2/Z)
print('background_term: ', background_term)

ndof = np.sum(design_mask.flatten())
co_term = ext_objective(np.ones(ndof), chif, background_mask, chid, design_mask, Npml, dl, cx, omega0, Qabs)
print('initial cloak+object: ', co_term, flush=True)

#we need |S3> = G_{f,dd}^{-1} G_{f} G_{0}^{-1} |E^{i*}>
S3 = G0inv @ np.conj(Ei.flatten())
S3 = spla.spsolve(Gfinv, S3)
#Gddinv, _ = TM.get_Gddinv(wv, dl, Nx, Ny, Npml, design_mask, bloch_x=0, bloch_y=0, Qabs=Qabs)
S3 = np.reshape(S3, (Nx, Ny))
S3 = S3[big_design_mask]
S3 = Gfddinv @ S3.flatten()

#we need |S2> = |S3^*>
S2 = np.conj(S3)

#we need |E_bowtie> = G_f G_{0}^{-1}|E^i>
#E_bowtie = G0inv @ (Ei.flatten())
#E_bowtie = spla.spsolve(Gfinv, E_bowtie)
test_Ebowtie = True
if test_Ebowtie:
    Ebowtie = G0inv @ (Ei.flatten())
    Ebowtie = spla.spsolve(Gfinv, Ebowtie) #this gives the same as E_bowtie
E_bowtie = TM.get_TM_linesource_field(wv, dl, Nx, Ny, cx, Npml, bloch_x=0.0, bloch_y=0.0, amp=1.0, Qabs=Qabs, chigrid=big_background_chi)
E_bowtie = E_bowtie.flatten()
#compare the two methods. If they agree, suggests that Gfinv is correct.
if do_checks:
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(np.reshape(E_bowtie, (Nx,Ny))), cmap='RdBu')
    ax2.imshow(np.imag(np.reshape(E_bowtie, (Nx,Ny))), cmap='RdBu')
    plt.savefig('test_E_bowtie.png')
    
if do_checks and test_Ebowtie:
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(np.reshape(Ebowtie, (Nx,Ny))), cmap='RdBu')
    ax2.imshow(np.imag(np.reshape(Ebowtie, (Nx,Ny))), cmap='RdBu')
    plt.savefig('test_Ebowtie.png')
    
if do_checks and test_Ebowtie:
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(np.reshape(Ebowtie - E_bowtie, (Nx,Ny))), cmap='RdBu')
    ax2.imshow(np.imag(np.reshape(Ebowtie - E_bowtie, (Nx,Ny))), cmap='RdBu')
    plt.savefig('test_E_bowtieminusEbowtie.png')
    print('done vector calculations')

if do_checks and test_Ebowtie:
    print('|E_bowtie - Ebowtie|/|E_bowtie| = ', np.linalg.norm(E_bowtie - Ebowtie)/np.linalg.norm(E_bowtie))
    
E_bowtie = E_bowtie[big_design_mask.flatten()]

tot_chi = big_background_chi + chid*big_design_mask
E_tot = TM.get_TM_linesource_field(wv, dl, Nx, Ny, cx, Npml, bloch_x=0.0, bloch_y=0.0, amp=1.0, Qabs=Qabs, chigrid=tot_chi)

vec_T_bowtie = (chid*big_design_mask)*E_tot
vec_T_bowtie = vec_T_bowtie[big_design_mask]
vec_T_bowtie = vec_T_bowtie.flatten()

U = Gfddinv.conjugate().T @ (1/chid.conjugate() * Gfddinv) - Gfddinv

tilde_vec_T_bowtie =  spla.spsolve(Gfddinv, vec_T_bowtie)

########## check constraint
constraint = np.vdot(E_bowtie, vec_T_bowtie) - (np.conjugate(tilde_vec_T_bowtie) @ U @ tilde_vec_T_bowtie)

print('re_constraint_explicit: ', np.real(constraint), flush=True)
print('im_constraint_explicit: ', np.imag(constraint), flush=True)

########## check gradZTT
Plist = [np.eye(N, dtype=complex)]
nmat = 1
nsource = 1
npixels = len(Plist)
Ginvlist = [Gfddinv]
nm = nmat
total_numlag = int(npixels*(2*(nsource)**2 + 2*(nm*(nm-1)/2)*(nsource*(nsource+1)/2)))
gradZTT = zmultimat.get_gradZTT(Plist, nmat, nsource, np.array([[chid]]), Ginvlist)

########## check gradZTS
gradZTS_S = zmultimat.get_gradZTS_S(E_bowtie, Plist, Ginvlist, nm, nsource)

print('constraint_linear_ZTS: ', 2*np.conjugate(gradZTS_S[0]) @ tilde_vec_T_bowtie)
print('constraint_linear_explicit: ', np.vdot(E_bowtie, vec_T_bowtie))

print('re_constraint_quadratic_ZTT: ', np.conjugate(tilde_vec_T_bowtie) @ gradZTT[0] @ tilde_vec_T_bowtie)
print('re_constraint_quadratic_explicit: ', (np.conjugate(tilde_vec_T_bowtie) @ (U + U.conjugate().T)/2 @ tilde_vec_T_bowtie))

print('im_constraint_quadratic_ZTT: ', np.conjugate(tilde_vec_T_bowtie) @ gradZTT[1] @ tilde_vec_T_bowtie)
print('im_constraint_quadratic_explicit: ', (np.conjugate(tilde_vec_T_bowtie) @ (U - U.conjugate().T)/2j @ tilde_vec_T_bowtie), flush=True)

###lastly, test objective function with this 
#first test with only first object
background_term = np.imag(EiS1*omega/2/Z)
print(f'background_term: {background_term}')

ndof = np.sum(design_mask.flatten())
background_term_obj = ext_objective(np.zeros(ndof), chif, background_mask, chid, design_mask, Npml, dl, cx, omega0, Qabs)
print(f'background_term_obj: {background_term_obj}')

#now test with cloak

ndof = np.sum(design_mask.flatten())
tot_obj = ext_objective(np.ones(ndof), chif, background_mask, chid, design_mask, Npml, dl, cx, omega0, Qabs)
print(f'tot_obj: {tot_obj}')

#S2
S2term = np.imag(np.vdot(S2, tilde_vec_T_bowtie) * omega/2/Z) * dl**2
print(f'obj_S2_way: ', background_term + S2term)

#Lastly, check O_lin_S. If this passes, bound code is highly like to be correct (i.e. it calculates what I think it calculates)
print(f'obj_S2_term: {S2term}')

O_lin_S = (np.conj(omega)/2/Z) * S2 * (-1/2j) * dl**2
Oterm = np.vdot(tilde_vec_T_bowtie, O_lin_S)
print('<tildeTbowtie|O_lin_S> term',Oterm)
print('2 * real part of that:', 2*np.real(Oterm))
print('ratio', S2term/(2*np.real(Oterm)))


