#!/bin/bash
#SBATCH --job-name=test_2sides
#SBATCH --output=OUTPUT/test_2sides.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:01:00
#SBATCH --mem-per-cpu=10000
#SBATCH --error=ERRORS/test_2sides.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=run_cloak_bound_2sides.py

wavelength=1.0
#Qabs=1000

chifRe=4.0
chifIm=0.001
chidRe=5.0
chidIm=0.001

nprojx=1
nprojy=1

des_region='circle'
design_x=1.5
design_y=1.5
des_param0=0.25
des_param1=0.75
gpr=25
pml_sep=0.50
pml_thick=0.50

name='results/test_2sides'

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -chifRe $chifRe -chifIm $chifIm -chidRe $chidRe -chidIm $chidIm -nprojx $nprojx -nprojy $nprojy -des_region $des_region -design_x $design_x -design_y $design_y -des_param0 $des_param0 -des_param1 $des_param1 -gpr $gpr -pml_sep $pml_sep -pml_think $pml_thick -name $name

