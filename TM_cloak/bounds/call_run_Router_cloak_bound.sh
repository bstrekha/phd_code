#!/bin/bash
#SBATCH --job-name=sweep_Router_gpr30_chif-4+1j_chid5+1e-3j
#SBATCH --output=OUTPUT/sweep_Router_gpr30_chif-4+1j_chid5+1e-3j.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=24000
#SBATCH --error=ERRORS/sweep_Router_gpr30_chif-4+1j_chid5+1e-3j.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=run_sweep_cloak_bound.py

wavelength=1.0

chifRe=-4.0
chifIm=1.0
chidRe=5.0
chidIm=0.001

#Qabs=np.inf by default in run_sweep_cloak_bound.py

des_region='circle'
des_param1=0.25
sweep_type='Router'
maxratioRouterRinner=10.0
numRsweep=16
gpr=30
pml_sep=0.50
pml_thick=0.50

name='results/sweep_Router_gpr30_chif-4+1j_chid5+1e-3j'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -chifRe $chifRe -chifIm $chifIm -chidRe $chidRe -chidIm $chidIm -des_region $des_region -design_x $design_x -design_y $design_y -des_param1 $des_param1 -sweep_type $sweep_type -maxratioRouterRinner $maxratioRouterRinner -numRsweep $numRsweep -gpr $gpr -pml_sep $pml_sep -pml_think $pml_thick -name $name

