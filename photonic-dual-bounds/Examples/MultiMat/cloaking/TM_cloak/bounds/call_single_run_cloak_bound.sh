#!/bin/bash
#SBATCH --job-name=cloak_localall_Qabsinf_Rinner0d25_Router0d35_gpr40_chif-10+1e-1j_chid12+1e-2j
#SBATCH --output=OUTPUT/cloak_localall_Qabsinf_Rinner0d25_Router0d35_gpr40_chif-10+1e-1j_chid12+1e-2j.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=10000
#SBATCH --error=ERRORS/cloak_localall_Qabsinf_Rinner0d25_Router0d35_gpr40_chif-10+1e-1j_chid12+1e-2j.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=run_cloak_bound.py

wavelength=1.0
#Qabs=1000

chifRe=-10.0
chifIm=0.1
chidRe=12.0
chidIm=0.01

nprojx=300
nprojy=300

des_region='circle'
design_x=0.7
design_y=0.7
des_param0=0.25
des_param1=0.35
gpr=30
pml_sep=0.50
pml_thick=0.50

name='results/cloak_localall_Qabsinf_Rinner0d25_Router0d35_gpr40_chif-10+1e-1j_chid12+1e-2j'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -chifRe $chifRe -chifIm $chifIm -chidRe $chidRe -chidIm $chidIm -nprojx $nprojx -nprojy $nprojy -des_region $des_region -design_x $design_x -design_y $design_y -des_param0 $des_param0 -des_param1 $des_param1 -gpr $gpr -pml_sep $pml_sep -pml_think $pml_thick -name $name

