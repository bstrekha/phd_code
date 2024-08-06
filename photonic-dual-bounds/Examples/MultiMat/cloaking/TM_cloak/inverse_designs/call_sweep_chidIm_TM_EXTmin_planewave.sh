#!/bin/bash
#SBATCH --job-name=cloak_sweepchidIm_rand_gpr50_chif-10+1j_chidRe12_Rinner0d25_Router0d75
#SBATCH --output=OUTPUT/cloak_sweepchidIm_rand_gpr50_chif-10+1j_chidRe12_Rinner0d25_Router0d75.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --error=ERRORS/cloak_sweepchidIm_rand_gpr50_chif-10+1j_chidRe12_Rinner0d25_Router0d75.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=sweep_chidIm_TM_EXTmin_planewave.py

wavelength=1.0

chifRe=-10.0
chifIm=1.0
chidRe=12.0

chidIm_start=1.0
chidIm_end=0.000001
chidIm_num=13

des_region='circle'
design_x=1.5
design_y=1.5
des_param1=0.25
des_param2=0.75
gpr=50
pml_sep=0.50
pml_thick=0.50

init_type='rand'
init_file=''

maxeval=800
output_base=900 #if higher than maxeval, only saves opt found design
name='results/cloak_sweepchidIm_rand_gpr50_chif-10+1j_chidRe12_Rinner0d25_Router0d75'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -chifRe $chifRe -chifIm $chifIm -chidRe $chidRe -des_region $des_region -design_x $design_x -design_y $design_y -des_param1 $des_param1 -des_param2 $des_param2 -gpr $gpr -pml_sep $pml_sep -pml_think $pml_thick -init_type $init_type -maxeval $maxeval -output_base $output_base -name $name -chidIm_start $chidIm_start -chidIm_end $chidIm_end -chidIm_num $chidIm_num
