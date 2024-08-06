#!/bin/bash
#SBATCH --job-name=spectrum_cloak_Qabs_sweep_gpr50_Rinner0d25_Router0d75_chif4+1e-3j_chid5+1e-3j
#SBATCH --output=OUTPUT/spectrum_cloak_Qabs_sweep_gpr50_Rinner0d25_Router0d75_chif4+1e-3j_chid5+1e-3j.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=7000
#SBATCH --error=ERRORS/spectrum_cloak_Qabs_sweep_gpr50_Rinner0d25_Router0d75_chif4+1e-3j_chid5+1e-3j.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=ramp_spectrum_cloak_performance_TM_EXTmin_planewave.py

chifRe=4.0
chifIm=0.001
chidRe=5.0
chidIm=0.001

omega_pts=501

design_x=1.5
design_y=1.5
gpr=50
pml_sep=0.5

design_file='results/cloak_Qabs_sweep_vac_gpr50_Rinner0d25_Router0d75_chif4+1e-3j_chid5+1e-3j_Qabs5.0e+06_L1.5_optdof.npy'
background_file='results/cloak_Qabs_sweep_gpr50_Rinner0d25_Router0d75_chifRe4+1e-3j_chidRe5+1e-3j_L1.5_backgrounddof.npy'
save_name='results/cloak_Qabs_sweep_vac_gpr50_Rinner0d25_Router0d75_chif4+1e-3j_chid5+1e-3j_Qabs5.0e+06_L1.5_optdof'

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -chifRe $chifRe -chifIm $chifIm -chidRe $chidRe -chidIm $chidIm -omega_pts $omega_pts -design_x $design_x -design_y $design_y -gpr $gpr -pml_sep $pml_sep -design_file $design_file -background_file $background_file -save_name $save_name
