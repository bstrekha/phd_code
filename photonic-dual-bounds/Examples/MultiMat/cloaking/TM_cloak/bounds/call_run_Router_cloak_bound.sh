#!/bin/bash
#SBATCH --job-name=sweep_Router_gpr25_Rinner0d5_chif-10+1e-1j_chid8+1e-1j
#SBATCH --output=OUTPUT/sweep_Router_gpr25_Rinner0d5_chif-10+1e-1j_chid8+1e-1j.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=36000
#SBATCH --error=ERRORS/sweep_Router_gpr25_Rinner0d5_chif-10+1e-1j_chid8+1e-1j.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=run_sweep_cloak_bound.py

wavelength=1.0

chifRe=-10.0
chifIm=0.1
chidRe=8.0
chidIm=0.1

#Qabs=np.inf by default in run_sweep_cloak_bound.py

des_region='circle'
des_param1=0.5
sweep_type='Router'
maxratioRouterRinner=5.0
numRsweep=16
gpr=25
pml_sep=0.50
pml_thick=0.50

opttol=0.00001
fakeSratio=0.05

all_local_constraints=0 #1 means yes

name='results/sweep_Router_gpr25_Rinner0d5_chif-10+1e-1j_chid8+1e-1j'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -chifRe $chifRe -chifIm $chifIm -chidRe $chidRe -chidIm $chidIm -des_region $des_region -des_param1 $des_param1 -maxratioRouterRinner $maxratioRouterRinner -numRsweep $numRsweep -gpr $gpr -pml_sep $pml_sep -pml_think $pml_thick -name $name -opttol $opttol -fakeSratio $fakeSratio -all_local_constraints $all_local_constraints -sweep_type $sweep_type