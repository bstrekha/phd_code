#!/bin/bash
#SBATCH --job-name=sweepchifRe_chifIm1e-2j_chid-4+1j_Rinner0d25_Router0d5
#SBATCH --output=OUTPUT/sweepchifRe_chifIm1e-2j_chid-4+1j_Rinner0d25_Router0d5.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=10000
#SBATCH --error=ERRORS/sweepchifRe_chifIm1e-2j_chid-4+1j_Rinner0d25_Router0d5.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=run_sweep_cloak_bound.py

wavelength=1.0

chifIm=0.01
chidRe=-4.0
chidIm=1.0

des_region='circle'
design_x=1.0
design_y=1.0
des_param1=0.25
des_param2=0.5
gpr=50
pml_sep=0.50
pml_thick=0.50

name='results/sweepchifRe_chifIm1e-2j_chid-4+1j_Rinner0d25_Router0d5'

sweep_type='chifRe'
chifRe_start=100.0
chifRe_end=0.01
chifRe_num=31

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -chifIm $chifIm -chidRe $chidRe -chidIm $chidIm -des_region $des_region -design_x $design_x -design_y $design_y -des_param1 $des_param1 -des_param2 $des_param2 -gpr $gpr -pml_sep $pml_sep -pml_think $pml_thick -name $name -sweep_type $sweep_type -chifRe_start $chifRe_start -chifRe_end $chifRe_end -chifRe_num $chifRe_num

