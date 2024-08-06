import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time, sys, argparse
sys.path.append('../photonic-bounds')
import nlopt
import ceviche
from ceviche.constants import C_0, ETA_0
from objective_TE_LDOS_PB_Wfun import ldos_tot_objective

parser = argparse.ArgumentParser()
parser.add_argument('-wavelength',action='store',type=float,default=1.0)
parser.add_argument('-num_omegas',action='store',type=int,default=1001)

parser.add_argument('-ReChi',action='store',type=float,default=2.0)
parser.add_argument('-ImChi',action='store',type=float,default=1e-2)
parser.add_argument('-gpr',action='store',type=int,default=20)

###design area size, design area is rectangular with central rectangular hole where the dipole lives###
parser.add_argument('-design_x',action='store',type=float,default=1.0)
parser.add_argument('-design_y',action='store',type=float,default=1.0)
parser.add_argument('-geometry', action='store', type=str, default='Cavity')

parser.add_argument('-vacuum_x',action='store',type=float,default=0.2)
parser.add_argument('-vacuum_y',action='store',type=float,default=0.2)

parser.add_argument('-emitter_x',action='store',type=float,default=0.05)
parser.add_argument('-emitter_y',action='store',type=float,default=0.05)
parser.add_argument('-pol',action='store',type=int,default=2)

parser.add_argument('-dist_x',action='store',type=float,default=0.02)

#separation between pml inner boundary and source walls
parser.add_argument('-pml_sep',action='store',type=float,default=0.5)
parser.add_argument('-pml_thick',action='store',type=float,default=0.5)

parser.add_argument('-design_file',action='store',type=str,default='test.txt')

args,unknown = parser.parse_known_args()

k = 2*np.pi/args.wavelength
omega0 = k
dl = 1.0/args.gpr
design_L = args.design_x
Mx = int(np.round(design_L/dl))
My = int(np.round(design_L/dl))
if design_L == 0:
    Mx0 = int(np.round(1.0/dl))
    My0 = int(np.round(1.0/dl))
else:
    Mx0 = int(np.round(design_L/dl))
    My0 = int(np.round(design_L/dl))
Npml = int(np.round(args.pml_thick/dl))
if design_L < 10:
    Npmlsep = int(np.round(0.5/dl))
else:
    Npmlsep = int(np.round(1.0/dl))
Emitterx = int(np.round(args.emitter_x / dl))
Emittery = int(np.round(args.emitter_y / dl))
Vacuumx = int(np.round(args.vacuum_x / dl))
Vacuumy = int(np.round(args.vacuum_y / dl))
Distx = int(np.round(args.dist_x / dl))
pol = args.pol

Nx = Mx + 2*(Npmlsep+Npml)+Distx
Ny = My + 2*(Npmlsep+Npml)

design_mask = np.zeros((Nx,Ny), dtype=np.bool)
design_mask[Npml+Npmlsep+Distx:Npml+Npmlsep+Distx+Mx , Npml+Npmlsep:Npml+Npmlsep+My] = True

if Distx==0:#this is case where dipole at center of design domain and one may want a cavity
    design_mask[Npml+Npmlsep+Distx+(Mx-Vacuumx)//2:Npml+Npmlsep+Distx+(Mx-Vacuumx)//2+Vacuumx,Npml+Npmlsep+(My-Vacuumy)//2:Npml+Npmlsep+(My-Vacuumy)//2+Vacuumy] = False

chi = args.ReChi + 1j*args.ImChi
epsval = 1.0 + chi

epsVac = np.ones((Nx,Ny), dtype=np.complex)
#define the emitter/source x,y left and right indices
if Distx==0: #cavity case
    exL = Npml+Npmlsep+Distx+Mx//2 - Emitterx//2
    exR = Npml+Npmlsep+Distx+Mx//2 + Emitterx//2 + 1 #make source width symmetrical
    eyL = Npml+Npmlsep+My//2 - Emittery//2
    eyR = Npml+Npmlsep+My//2 + Emittery//2 + 1 #make source width symmetrical
else: #half-space case
    exL = Npml+Npmlsep - Emittery//2
    exR = Npml+Npmlsep + Emitterx//2 + 1 #make source width symmetrical
    eyL = Npml+Npmlsep + My//2 - Emittery//2
    eyR = Npml+Npmlsep + My//2 + Emittery//2 + 1 #make source width symmetrical
#set dipole source
emitter_mask = np.zeros((Nx,Ny), dtype=np.bool)
emitter_mask[exL:exR, eyL:eyR] = True

source = np.zeros((Nx,Ny), dtype=np.complex)
sourceXpixels = exR - exL
sourceYpixels = eyR - eyL
source[emitter_mask] = 1.0 / (sourceXpixels*dl*sourceYpixels*dl)

#now get spectrum
design_dof = np.loadtxt(args.design_file)

full_dof = np.zeros((Nx,Ny)) 
full_dof[design_mask] = design_dof[:]
num_omegas = args.num_omegas
omega_list = (np.linspace(-1e6/2/10**(6),1e6/2/10**(6), num_omegas) + 1)*omega0
rho_vac_0 = omega0/16
for i in range(len(omega_list)):
    cur_ldos,_,_ = ldos_tot_objective(full_dof.flatten(), epsval, design_mask, dl, source, args.wavelength, Nx, Ny, Npml, Npml, dl, dl, exL, exR, eyL, eyR, pol, 1.0/dl/dl/sourceXpixels/sourceYpixels, omega_list[i], np.inf)
    print(cur_ldos/rho_vac_0, flush=True)
