#!/bin/bash
#SBATCH --job-name=sweepchifIm_global_gpr25_chifRe-4_chid3+1e-2j_Rinner0d25_Router0d75
#SBATCH --output=OUTPUT/sweepchifIm_global_gpr25_chifRe-4_chid3+1e-2j_Rinner0d25_Router0d75.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=12000
#SBATCH --error=ERRORS/sweepchifIm_global_gpr25_chifRe-4_chid3+1e-2j_Rinner0d25_Router0d75.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=run_sweep_cloak_bound.py

wavelength=1.0

chifRe=-4.0
chidRe=3.0
chidIm=0.01

nprojx=1
nprojy=1
all_local_constraints=0 #1 means yes

des_region='circle'
design_x=1.5
design_y=1.5
des_param1=0.25
des_param2=0.75
gpr=25
pml_sep=0.50
pml_thick=0.50

name='results/sweepchifIm_global_gpr25_chifRe-4_chid3+1e-2j_Rinner0d25_Router0d75'

sweep_type='chifIm'
chifIm_start=1.0
chifIm_end=0.00001
chifIm_num=31

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -chifRe $chifRe -chidRe $chidRe -chidIm $chidIm -nprojx $nprojx -nprojy $nprojy -all_local_constraints $all_local_constraints -des_region $des_region -design_x $design_x -design_y $design_y -des_param1 $des_param1 -des_param2 $des_param2 -gpr $gpr -pml_sep $pml_sep -pml_think $pml_thick -name $name -sweep_type $sweep_type -chifIm_start $chifIm_start -chifIm_end $chifIm_end -chifIm_num $chifIm_num

