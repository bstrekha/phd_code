#!/bin/bash
#SBATCH --job-name=sweep_local15-15_Router_gpr40_Rinner0d3_chif10+1e-1j_chid12+1e-2j_Qabsinf
#SBATCH --output=OUTPUT/sweep_local15-15_Router_gpr40_Rinner0d3_chif10+1e-1j_chid12+1e-2j_Qabsinf.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=70000
#SBATCH --error=ERRORS/sweep_local15-15_Router_gpr40_Rinner0d3_chif10+1e-1j_chid12+1e-2j_Qabsinf.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=run_sweep_cloak_bound.py

wavelength=1.0

chifRe=10.0
chifIm=0.1
chidRe=12.0
chidIm=0.01

nprojx=15
nprojy=15
all_local_constraints=0 #1 means yes

des_region='circle'
des_param1=0.3
maxratioRouterRinner=7.0
numRsweep=20
gpr=40
pml_sep=0.50
pml_thick=0.50

opttol=0.000001
fakeSratio=0.001
iter_period=110

sweep_type='Router'

Qabs=10000000000000000000.0 #for Qabs=np.inf in code
#Qabs=500.0

name='results/sweep_local15-15_Router_gpr40_Rinner0d3_chif10+1e-1j_chid12+1e-2j_Qabsinf'

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -chifRe $chifRe -chifIm $chifIm -chidRe $chidRe -chidIm $chidIm -nprojx $nprojx -nprojy $nprojy -all_local_constraints $all_local_constraints -des_region $des_region -des_param1 $des_param1 -maxratioRouterRinner $maxratioRouterRinner -numRsweep $numRsweep -gpr $gpr -pml_sep $pml_sep -pml_think $pml_thick -name $name -opttol $opttol -fakeSratio $fakeSratio -sweep_type $sweep_type -iter_period $iter_period -Qabs $Qabs

