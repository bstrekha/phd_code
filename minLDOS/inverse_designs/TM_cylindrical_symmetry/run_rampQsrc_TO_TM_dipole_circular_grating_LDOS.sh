#!/bin/bash
#SBATCH --job-name=TMcyl_chi4+1e-1j_R50_gpr1200
#SBATCH --output=TMcyl_chi4+1e-1j_R50_gpr1200.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=50:00:00
#SBATCH --mem-per-cpu=10000
#SBATCH --error=TMcyl_chi4+1e-1j_R50_gpr1200.err

module load anaconda3/2020.11

prog=rampQsrc_TO_TM_dipole_circular_grating_LDOS.py

wavelength=1.0
pow10Qabs_start=5
pow10Qabs_end=6
pow10Qabs_num=5
includeQinf=-1 #1=true, -1=false

ReChi=4.0
ImChi=0.1

gpr=1200 #pixels per wavelength

r_inner=1.0 #design domain
r_outer=2.0 
pml_sep=1.0
pml_thick=0.5

init_type='vac'
#init_file='TESTINGES_des5by5.txt'

output_base=12000 #make larger than maxeval to only save opt_dof
maxeval=10
name='DATA/TESTTMcyl_chi4+1e-1j_R1_gpr1200'

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -pow10Qabs_start $pow10Qabs_start -pow10Qabs_end $pow10Qabs_end -pow10Qabs_num $pow10Qabs_num -includeQinf $includeQinf -ReChi $ReChi -ImChi $ImChi -gpr $gpr -r_inner $r_inner -r_outer $r_outer -pml_sep $pml_sep -pml_thick $pml_thick -init_type $init_type -output_base $output_base -maxeval $maxeval -name $name

