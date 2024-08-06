#!/bin/bash
#SBATCH --job-name=cloak_Router_sweep_vac_gpr50_circle_Rinner0d5_chif10+1e-1j_chid12+1e-2j_Qabsinf_maxeval300
#SBATCH --output=OUTPUT/cloak_Router_sweep_vac_gpr50_circle_Rinner0d5_chif10+1e-1j_chid12+1e-2j_Qabsinf_maxeval300.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=850
#SBATCH --error=ERRORS/cloak_Router_sweep_vac_gpr50_circle_Rinner0d5_chif10+1e-1j_chid12+1e-2j_Qabsinf_maxeval300.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=sweep_Router_TM_EXTmin_planewave.py

wavelength=1.0

Qabs=100000000000000000000.0 #for Qabs=np.inf in code
#Qabs=500.0

chifRe=10.0
chifIm=0.1
chidRe=12.0
chidIm=0.01

des_region='circle'
des_param1=0.5
maxratioRouterRinner=5.0
numRsweep=11
gpr=50
pml_sep=0.50
pml_thick=0.50

init_type='vac'
init_file=''

maxeval=300
output_base=900 #if higher than maxeval, only saves opt found design
name='results/cloak_Router_sweep_vac_gpr50_circle_Rinner0d5_chif10+1e-1j_chid12+1e-2j_Qabsinf_maxeval300'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -Qabs $Qabs -chifRe $chifRe -chifIm $chifIm -chidRe $chidRe -chidIm $chidIm -des_region $des_region -des_param1 $des_param1 -maxratioRouterRinner $maxratioRouterRinner -numRsweep $numRsweep -gpr $gpr -pml_sep $pml_sep -pml_think $pml_thick -init_type $init_type -maxeval $maxeval -output_base $output_base -name $name

