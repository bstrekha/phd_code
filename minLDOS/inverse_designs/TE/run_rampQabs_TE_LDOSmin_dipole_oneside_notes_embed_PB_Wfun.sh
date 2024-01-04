#!/bin/bash
#SBATCH --job-name=TESTINGES
#SBATCH --output=TESTINGES.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=50:00:00
#SBATCH --mem-per-cpu=6000
#SBATCH --error=TESTINGES.err

module load anaconda3/2020.11

prog=rampQabs_TE_LDOSmin_dipole_oneside_notes_embed_PB_Wfun.py

wavelength=1.0
pow10Qabs_start=1
pow10Qabs_end=6
pow10Qabs_num=5
Qinfonly=0 #1=true, 0=false

ReChi=4.0
ImChi=0.1

gpr=100 #pixels per wavelength

design_x=1.0 #design domain width
design_y=1.0 #design domain height

dist_x=0 #0 means 'full-space' case (dipole at center). non-zero value is interpreted as 'half-space' case (dipole on side)
emitter_x=0.031 #width of dipole source (this is 3 pixels width, so source is even around center)
emitter_y=0.031 #height of dipole source
pol=2 #1 is horizontal polarization, 2 is vertical polarization
vacuum_x=0.062 #if dist_x=0, how much of a vac cavity to generate around dipole
vacuum_y=0.062 #if dist_x=0, how much of a vac cavity to generate around dipole

pml_thick=0.5
pml_sep=1.0

init_type='vac'
init_file='TESTINGES_des5by5.txt'

output_base=200
maxeval=800
#name='TELDOSmin4+1e-2j_pol2_Qabs1e4to1e6_des5by5_halfspace_maxeval400_Wfunc_normalized_vacstart'
name='TESTINGES'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -pow10Qabs_start $pow10Qabs_start -pow10Qabs_end $pow10Qabs_end -pow10Qabs_num $pow10Qabs_num -Qinfonly $Qinfonly -ReChi $ReChi -ImChi $ImChi -gpr $gpr -design_x $design_x -design_y $design_y -vacuum_x $vacuum_x -vacuum_y $vacuum_y -emitter_x $emitter_x -emitter_y $emitter_y -pol $pol -pml_thick $pml_thick -pml_sep $pml_sep -init_type $init_type -init_file $init_file -output_base $output_base -name $name -maxeval $maxeval -dist_x $dist_x

