#!/bin/bash
#SBATCH --job-name=gcd_cloak_Pnum25_Qabs_sweep_Rinner0d25_Router0d75_gpr25_chif4+1e-3j_chid5+1e-3j
#SBATCH --output=OUTPUT/gcd_cloak_Pnum25_Qabs_sweep_Rinner0d25_Router0d75_gpr25_chif4+1e-3j_chid5+1e-3j.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=14000
#SBATCH --error=ERRORS/gcd_cloak_Pnum25_Qabs_sweep_Rinner0d25_Router0d75_gpr25_chif4+1e-3j_chid5+1e-3j.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=run_sweep_cloak_bound_gcd.py

wavelength=1.0
pow10Qabs_start=6
pow10Qabs_end=7
pow10Qabs_num=4
dwfactor2=1 #1=true

nprojx=5
nprojy=5
#Pnum = nprojx*projy in the code
all_local_constraints=0 #1 means yes
maxiter=120
saveint=1

chifRe=4.0
chifIm=0.001
chidRe=5.0
chidIm=0.001

des_region='circle'
design_x=1.5
design_y=1.5
des_param1=0.25
des_param2=0.75
gpr=25
pml_sep=0.50
pml_thick=0.50

sweep_type='Q'

name='results/gcd_cloak_Pnum25_Qabs_sweep_Rinner0d25_Router0d75_gpr25_chif4+1e-3j_chid5+1e-3j'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -pow10Qabs_start $pow10Qabs_start -pow10Qabs_end $pow10Qabs_end -pow10Qabs_num $pow10Qabs_num -dwfactor2 $dwfactor2 -nprojx $nprojx -nprojy $nprojy -all_local_constraints $all_local_constraints -maxiter $maxiter -chifRe $chifRe -chifIm $chifIm -chidRe $chidRe -chidIm $chidIm -des_region $des_region -design_x $design_x -design_y $design_y -des_param1 $des_param1 -des_param2 $des_param2 -gpr $gpr -pml_sep $pml_sep -pml_think $pml_thick -name $name -sweep_type $sweep_type -saveint $saveint

