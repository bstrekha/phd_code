#!/bin/bash
#SBATCH --job-name=TM1d_chi4+0j_gpr1200_vac_Qsrc5.0e+06_Lsweep1to200_logspaced_80pts_gpr1200
#SBATCH --output=OUTPUTTXT/TM1d_chi4+0j_gpr1200_vac_Qsrc5.0e+06_Lsweep1to200_logspaced_80pts_gpr1200.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=28000
#SBATCH --error=ERRORS/TM1d_chi4+0j_gpr1200_vac_Qsrc5.0e+06_Lsweep1to200_logspaced_80pts_gpr1200.err

module load anaconda3/2020.11

prog=rampQsrc_TO_TM_dipole_1d_grating_LDOS_Lsweep.py

wavelength=1.0
Qsrc=5e6

ReChi=4.0
ImChi=0.0

gpr=1200 #pixels per wavelength

L_start=1.0 #design domain
L_end=200
L_pts=80
pml_sep=1.0
pml_thick=3.0

init_type='vac'
#init_file='TESTINGES_des5by5.txt'

output_base=12000 #make larger than maxeval to only save opt_dof
maxeval=10000
name='DATA/TM1d_chi4+0j_gpr1200_vac_Qsrc5.0e+06_Lsweep1to200_logspaced_80pts_gpr1200'
xtol_abs=0.0

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -Qsrc $Qsrc -ReChi $ReChi -ImChi $ImChi -gpr $gpr -L_start $L_start -L_end $L_end -L_pts $L_pts -pml_sep $pml_sep -pml_thick $pml_thick -init_type $init_type -output_base $output_base -maxeval $maxeval -name $name -xtol_abs $xtol_abs

