import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time, sys, argparse
import nlopt
import ceviche
from ceviche.constants import C_0, ETA_0
from objective_TM_LDOS_PB_Wfun import ldos_tot_objective

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

parser.add_argument('-dist_x',action='store',type=float,default=0.02)

#separation between pml inner boundary and source walls
parser.add_argument('-pml_sep',action='store',type=float,default=0.5)
parser.add_argument('-pml_thick',action='store',type=float,default=0.5)

parser.add_argument('-design_file',action='store',type=str,default='test.txt')

args,unknown = parser.parse_known_args()

k = 2*np.pi/args.wavelength
omega0 = C_0 * k
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

Nx = Mx + 2*(Npmlsep+Npml)+Distx
Ny = My + 2*(Npmlsep+Npml)

design_mask = np.zeros((Nx,Ny), dtype=bool)
design_mask[Npml+Npmlsep+Distx:Npml+Npmlsep+Distx+Mx , Npml+Npmlsep:Npml+Npmlsep+My] = True
design_mask[Npml+Npmlsep+Distx+(Mx-Vacuumx)//2:Npml+Npmlsep+Distx+(Mx-Vacuumx)//2+Vacuumx,Npml+Npmlsep+(My-Vacuumy)//2:Npml+Npmlsep+(My-Vacuumy)//2+Vacuumy] = False

chi = args.ReChi - 1j*args.ImChi #ceviche has +iwt time convention
epsval = 1.0 + chi
if args.geometry.lower()[0] == 'c':
    emitter_mask = np.zeros((Nx,Ny), dtype=bool)
    emitter_mask[Npml+Npmlsep+(Mx-Emitterx)//2:Npml+Npmlsep+(Mx-Emitterx)//2+Emitterx,Npml+Npmlsep+(My-Emittery)//2:Npml+Npmlsep+(My-Emittery)//2+Emittery] = True
else:
    emitter_mask = np.zeros((Nx,Ny), dtype=bool)
    emitter_mask[Npml+Npmlsep:Npml+Npmlsep+Emitterx,Npml+Npmlsep+(My-Emittery)//2:Npml+Npmlsep+(My-Emittery)//2+Emittery] = True

#set TM dipole source
source = np.zeros((Nx,Ny), dtype=complex)
source[emitter_mask] = 1.0 / (Emitterx*dl*Emittery*dl)

#now get spectrum
design_dof = np.loadtxt(args.design_file)

full_dof = np.zeros((Nx,Ny)) 
full_dof[design_mask] = design_dof[:]
num_omegas = args.num_omegas
omega_list = (np.linspace(-1e6/2/10**(6),1e6/2/10**(6), num_omegas) + 1)*omega0
rho_vac_0 = omega0/8*np.sqrt(1.25663706e-6/ 8.85418782e-12)*np.sqrt(1.25663706e-6 * 8.85418782e-12)
for i in range(len(omega_list)):
    print(ldos_tot_objective(full_dof.flatten(), epsval, design_mask, dl, source, omega_list[i], np.inf, Npml)/rho_vac_0, flush=True)
