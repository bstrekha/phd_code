#!/bin/bash
#SBATCH --job-name=sweepRinner_gpr_30_chif-4+1e-1j_chid5+1e-1j_Rinnerstart0d25_Rinnerend4d0
#SBATCH --output=OUTPUT/sweepRinner_gpr_30_chif-4+1e-1j_chid5+1e-1j_Rinnerstart0d25_Rinnerend4d0.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=24000
#SBATCH --error=ERRORS/sweepRinner_gpr_30_chif-4+1e-1j_chid5+1e-1j_Rinnerstart0d25_Rinnerend4d0.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=run_sweep_cloak_bound.py

wavelength=1.0

chifRe=-4.0
chifIm=0.1
chidRe=5.0
chidIm=0.1

des_region='circle'
gpr=30
pml_sep=0.50
pml_thick=0.50

all_local_constraints=0 #1 means yes

sweep_type='Rinner'

Rinner_start=0.25
Rinner_end=4.0
numRsweep=21

name='results/sweepRinner_gpr_30_chif-4+1e-1j_chid5+1e-1j_Rinnerstart0d25_Rinnerend4d0'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -chifRe $chifRe -chifIm $chifIm -chidRe $chidRe -chidIm $chidIm -des_region $des_region -numRsweep $numRsweep -gpr $gpr -pml_sep $pml_sep -pml_think $pml_thick -name $name -all_local_constraints $all_local_constraints -sweep_type $sweep_type -Rinner_start $Rinner_start -Rinner_end $Rinner_end

