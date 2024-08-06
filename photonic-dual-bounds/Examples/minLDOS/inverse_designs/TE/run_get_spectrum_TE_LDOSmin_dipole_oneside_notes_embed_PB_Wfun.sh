#!/bin/bash
#SBATCH --job-name=TELDOSmin4+1e-1j_Qabs1e4to1e6_des5by5_halfspace0d1_maxeval300_Wfunc_normalized_vacstart_Qabs1.0e+04_dof100_spectrum
#SBATCH --output=TELDOSmin4+1e-1j_Qabs1e4to1e6_des5by5_halfspace0d1_maxeval300_Wfunc_normalized_vacstart_Qabs1.0e+04_dof100_spectrum.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=4900
#SBATCH --error=TELDOSmin4+1e-1j_Qabs1e4to1e6_des5by5_halfspace0d1_maxeval300_Wfunc_normalized_vacstart_Qabs1.0e+04_dof100_spectrum.err

module purge
module load anaconda3/2020.11

prog=get_spectrum_TE_LDOSmin_dipole_oneside_notes_embed_PB_Wfun.py

wavelength=1.0
num_omega=2001
#geometry='Cavity'
geometry='halfspace'

ReChi=4.0
ImChi=0.1

gpr=100

design_x=5.00
design_y=5.00

vacuum_x=0.0
vacuum_y=0.0
dist_x=0.1

emitter_x=0.025
emitter_y=0.025
pol=2

pml_thick=0.5
pml_sep=0.5

# cp checkPC_gapdes10by10.txt checkPC_gapdes10by10_test.txt 
design_file='TELDOSmin4+1e-1j_Qabs1e4to1e6_des5by5_halfspace0d1_maxeval300_Wfunc_normalized_vacstart_Qabs1.0e+04_dof100.txt'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -pow10Qabs_start $pow10Qabs_start -pow10Qabs_end $pow10Qabs_end -pow10Qabs_num $pow10Qabs_num -ReChi $ReChi -ImChi $ImChi -gpr $gpr -design_x $design_x -design_y $design_y -vacuum_x $vacuum_x -vacuum_y $vacuum_y -emitter_x $emitter_x -emitter_y $emitter_y -pml_thick $pml_thick -pml_sep $pml_sep -init_type $init_type -design_file $design_file -dist_x $dist_x -geometry $geometry -num_omega $num_omega -pol $pol

