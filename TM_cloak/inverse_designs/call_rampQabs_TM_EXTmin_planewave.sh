#!/bin/bash
#SBATCH --job-name=gpr50_Rinner0d25_Router0d75_chif4+1e-3j_chid5+1e-3j
#SBATCH --output=OUTPUT/gpr50_Rinner0d25_Router0d75_chif4+1e-3j_chid5+1e-3j.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=10000
#SBATCH --error=ERRORS/gpr50_Rinner0d25_Router0d75_chif4+1e-3j_chid5+1e-3j.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=rampQabs_TM_EXTmin_planewave.py

wavelength=1.0
pow10Qabs_start=2
pow10Qabs_end=7
pow10Qabs_num=11
dwfactor2=1 #1=true

chifRe=4.0
chifIm=0.001
chidRe=5.0
chidIm=0.001

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

maxeval=600
output_base=900 #if higher than maxeval, only saves opt found design
name='results/gpr50_Rinner0d25_Router0d75_chif4+1e-3j_chid5+1e-3j'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -pow10Qabs_start $pow10Qabs_start -pow10Qabs_end $pow10Qabs_end -pow10Qabs_num $pow10Qabs_num -dwfactor2 $dwfactor2 -chifRe $chifRe -chifIm $chifIm -chidRe $chidRe -chidIm $chidIm -des_region $des_region -design_x $design_x -design_y $design_y -des_param1 $des_param1 -des_param2 $des_param2 -gpr $gpr -pml_sep $pml_sep -pml_think $pml_thick -init_type $init_type -maxeval $maxeval -output_base $output_base -name $name

