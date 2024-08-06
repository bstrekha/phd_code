#!/bin/bash
#SBATCH --job-name=sweepchidIm_local6-6_gpr25_chif-10+1j_chidRe12_Rinner0d25_Router0d75
#SBATCH --output=OUTPUT/sweepchidIm_local6-6_gpr25_chif-10+1j_chidRe12_Rinner0d25_Router0d75.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=10000
#SBATCH --error=ERRORS/sweepchidIm_local6-6_gpr25_chif-10+1j_chidRe12_Rinner0d25_Router0d75.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=run_sweep_cloak_bound.py

wavelength=1.0

chifRe=-10.0
chifIm=1.0
chidRe=12.0

nprojx=6
nprojy=6
all_local_constraints=0 #1 means yes

des_region='circle'
design_x=1.5
design_y=1.5
des_param1=0.25
des_param2=0.75
gpr=25
pml_sep=0.50
pml_thick=0.50

name='results/sweepchidIm_local6-6_gpr25_chif-10+1j_chidRe12_Rinner0d25_Router0d75'

sweep_type='chidIm'
chidIm_start=1.0
chidIm_end=0.000001
chidIm_num=37
opttol=0.0000001
fakeSratio=0.005
iter_period=110

normalize_result=1 #0 = False, 1 = True

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -chifRe $chifRe -chifIm $chifIm -chidRe $chidRe -nprojx $nprojx -nprojy $nprojy -all_local_constraints $all_local_constraints -des_region $des_region -design_x $design_x -design_y $design_y -des_param1 $des_param1 -des_param2 $des_param2 -gpr $gpr -pml_sep $pml_sep -pml_think $pml_thick -name $name -sweep_type $sweep_type -chidIm_start $chidIm_start -chidIm_end $chidIm_end -chidIm_num $chidIm_num -opttol $opttol -normalize_result $normalize_result -fakeSratio $fakeSratio -iter_period $iter_period

