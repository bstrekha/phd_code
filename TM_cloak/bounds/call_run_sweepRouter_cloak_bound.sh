#!/bin/bash
#SBATCH --job-name=sweepRouter_gpr_30_chif4+1e-3j_chid5+1e-3j_Rinner0d25
#SBATCH --output=OUTPUT/sweepRouter_gpr_30_chif4+1e-3j_chid5+1e-3j_Rinner0d25.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=24000
#SBATCH --error=ERRORS/sweepRouter_gpr_30_chif4+1e-3j_chid5+1e-3j_Rinner0d25.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=run_sweep_cloak_bound.py

wavelength=1.0
#Qabs=1000

chifRe=4.0
chifIm=0.001
chidRe=5.0
chidIm=0.001

des_region='circle'
des_param1=0.25
maxratioRouterRinner=15.0
numRsweep=31
gpr=30
pml_sep=0.50
pml_thick=0.50

all_local_constraints=0 #1 means yes

sweep_type='Router'

name='results/sweepRouter_gpr_30_chif4+1e-3j_chid5+1e-3j_Rinner0d25'

do_checks=False

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -chifRe $chifRe -chifIm $chifIm -chidRe $chidRe -chidIm $chidIm -des_region $des_region -des_param1 $des_param1 -maxratioRouterRinner $maxratioRouterRinner -numRsweep $numRsweep -gpr $gpr -pml_sep $pml_sep -pml_think $pml_thick -name $name -all_local_constraints $all_local_constraints -sweep_type $sweep_type -do_checks $do_checks

