#!/bin/bash
#SBATCH --job-name=1by1_chif4+1e-3j_chid2+1e-3j
#SBATCH --output=OUTPUT/1by1_chif4+1e-3j_chid2+1e-3j.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=10000
#SBATCH --error=ERRORS/1by1_chif4+1e-3j_chid2+1e-3j.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=run_sweep_cloak_bound.py

wavelength=1.0
pow10Qabs_start=2
pow10Qabs_end=7
pow10Qabs_num=11
dwfactor2=1 #1=true

chifRe=4.0
chifIm=0.001
chidRe=2.0
chidIm=0.001

des_region='circle'
design_x=1.0
design_y=1.0
des_param1=0.25
des_param2=0.5
gpr=50
pml_sep=0.50
pml_thick=0.50

name='results/cloak_1by1_chif4+1e-3j_chid2+1e-3j'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -pow10Qabs_start $pow10Qabs_start -pow10Qabs_end $pow10Qabs_end -pow10Qabs_num $pow10Qabs_num -dwfactor2 $dwfactor2 -chifRe $chifRe -chifIm $chifIm -chidRe $chidRe -chidIm $chidIm -des_region $des_region -design_x $design_x -design_y $design_y -des_param1 $des_param1 -des_param2 $des_param2 -gpr $gpr -pml_sep $pml_sep -pml_think $pml_thick -name $name

