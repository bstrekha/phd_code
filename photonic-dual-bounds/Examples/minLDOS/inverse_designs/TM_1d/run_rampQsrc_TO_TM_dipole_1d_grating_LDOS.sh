#!/bin/bash
#SBATCH --job-name=TM1d_chi4+0j_L40_gpr1200_vac
#SBATCH --output=OUTPUTTXT/TM1d_chi4+0j_L40_gpr1200_vac.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=28000
#SBATCH --error=ERRORS/TM1d_chi4+0j_L40_gpr1200_vac.err

module load anaconda3/2020.11

prog=rampQsrc_TO_TM_dipole_1d_grating_LDOS.py

wavelength=1.0
pow10Qsrc_start=-1
pow10Qsrc_end=7
pow10Qsrc_num=17
dwfactor2=1 #1=true, -1=false
includeQinf=-1 #1=true, -1=false

ReChi=4.0
ImChi=0.0

gpr=1200 #pixels per wavelength

zMin=0.0 #design domain
zMax=40.0
pml_sep=1.0
pml_thick=3.0

init_type='vac'
#init_file='TESTINGES_des5by5.txt'

output_base=13000 #make larger than maxeval to only save opt_dof
maxeval=10000
name='DATA/TM1d_chi4+0j_L40_gpr1200_vac'
xtol_abs=0.0

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -pow10Qsrc_start $pow10Qsrc_start -pow10Qsrc_end $pow10Qsrc_end -pow10Qsrc_num $pow10Qsrc_num -includeQinf $includeQinf -dwfactor2 $dwfactor2 -ReChi $ReChi -ImChi $ImChi -gpr $gpr -zMin $zMin -zMax $zMax -pml_sep $pml_sep -pml_thick $pml_thick -init_type $init_type -output_base $output_base -maxeval $maxeval -name $name -xtol_abs $xtol_abs