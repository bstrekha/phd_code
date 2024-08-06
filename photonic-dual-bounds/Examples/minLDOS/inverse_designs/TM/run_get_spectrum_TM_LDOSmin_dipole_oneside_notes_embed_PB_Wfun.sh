#!/bin/bash
#SBATCH --job-name=TMLDOSmin4+0j_dwQabs1e-1to1e7_periodicfullspace10by10_period1_maxeval500_Wfunc_normalized_vacstart_Qabs5.0e+06_L10.0_optdof_spectrum
#SBATCH --output=TMLDOSmin4+0j_dwQabs1e-1to1e7_periodicfullspace10by10_period1_maxeval500_Wfunc_normalized_vacstart_Qabs5.0e+06_L10.0_optdof_spectrum.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=14000
#SBATCH --error=TMLDOSmin4+0j_dwQabs1e-1to1e7_periodicfullspace10by10_period1_maxeval500_Wfunc_normalized_vacstart_Qabs5.0e+06_L10.0_optdof_spectrum.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=get_spectrum_TM_LDOSmin_dipole_oneside_notes_embed_PB_Wfun.py

wavelength=1.0
num_omega=1001
geometry='Cavity'
# geometry='halfspace'

ReChi=4.0
ImChi=0.0

gpr=100

design_x=10.00
design_y=10.00

vacuum_x=0.0
vacuum_y=0.0
dist_x=0.0

emitter_x=0.025
emitter_y=0.025

pml_thick=0.5
pml_sep=0.5

# cp checkPC_gapdes10by10.txt checkPC_gapdes10by10_test.txt 
design_file='TMLDOSmin4+0j_dwQabs1e-1to1e7_periodicfullspace10by10_period1_maxeval500_Wfunc_normalized_vacstart_Qabs5.0e+06_L10.0_optdof.txt'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -pow10Qabs_start $pow10Qabs_start -pow10Qabs_end $pow10Qabs_end -pow10Qabs_num $pow10Qabs_num -ReChi $ReChi -ImChi $ImChi -gpr $gpr -design_x $design_x -design_y $design_y -vacuum_x $vacuum_x -vacuum_y $vacuum_y -emitter_x $emitter_x -emitter_y $emitter_y -pml_thick $pml_thick -pml_sep $pml_sep -init_type $init_type -design_file $design_file -dist_x $dist_x -geometry $geometry -num_omega $num_omega

