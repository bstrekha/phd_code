#!/bin/bash
#SBATCH --job-name=sweepchidRe_grp40_chif4+1e-2j_chidIm1e-2_Rinner0d25_Router0d5
#SBATCH --output=OUTPUT/sweepchidRe_grp40_chif4+1e-2j_chidIm1e-2_Rinner0d25_Router0d5.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=12000
#SBATCH --error=ERRORS/sweepchidRe_grp40_chif4+1e-2j_chidIm1e-2_Rinner0d25_Router0d5.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=run_sweep_cloak_bound.py

wavelength=1.0

chifRe=4.0
chifIm=0.01
chidIm=0.01

des_region='circle'
design_x=1.0
design_y=1.0
des_param1=0.25
des_param2=0.50
gpr=40
pml_sep=0.50
pml_thick=0.50

name='results/sweepchidRe_grp40_chif4+1e-2j_chidIm1e-2_Rinner0d25_Router0d5'

sweep_type='chidRe'
chidRe_start=0.001
chidRe_end=100.0
chidRe_num=51

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -chifRe $chifRe -chifIm $chifIm -chidIm $chidIm -des_region $des_region -design_x $design_x -design_y $design_y -des_param1 $des_param1 -des_param2 $des_param2 -gpr $gpr -pml_sep $pml_sep -pml_think $pml_thick -name $name -sweep_type $sweep_type -chidRe_start $chidRe_start -chidRe_end $chidRe_end -chidRe_num $chidRe_num

