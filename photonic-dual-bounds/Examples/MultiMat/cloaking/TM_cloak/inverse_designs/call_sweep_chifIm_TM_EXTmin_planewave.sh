#!/bin/bash
#SBATCH --job-name=cloak_sweepchifIm_gpr25_chif4_chidRe5+1e-2j_Rinner0d25_Router0d75
#SBATCH --output=OUTPUT/cloak_sweepchifIm_gpr25_chif4_chidRe5+1e-2j_Rinner0d25_Router0d75.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=12000
#SBATCH --error=ERRORS/cloak_sweepchifIm_gpr25_chif4_chidRe5+1e-2j_Rinner0d25_Router0d75.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=sweep_chifIm_TM_EXTmin_planewave.py

wavelength=1.0

chifRe=4.0
chidRe=5.0
chidIm=0.01

chifIm_start=1.0
chifIm_end=0.00001
chifIm_num=11

des_region='circle'
design_x=1.5
design_y=1.5
des_param1=0.25
des_param2=0.75
gpr=25
pml_sep=0.50
pml_thick=0.50

init_type='vac'
init_file=''

maxeval=400
output_base=900 #if higher than maxeval, only saves opt found design
name='results/cloak_sweepchifIm_gpr25_chif4_chidRe5+1e-2j_Rinner0d25_Router0d75'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -chifRe $chifRe -chidRe $chidRe -chidIm $chidIm -des_region $des_region -design_x $design_x -design_y $design_y -des_param1 $des_param1 -des_param2 $des_param2 -gpr $gpr -pml_sep $pml_sep -pml_think $pml_thick -init_type $init_type -maxeval $maxeval -output_base $output_base -name $name -chifIm_start $chifIm_start -chifIm_end $chifIm_end -chifIm_num $chifIm_num

