#!/bin/bash
#SBATCH --job-name=TM1d_chi4+1e-1j_L10_gpr1200_vac_Qsrc5.0e+06_optdof_spectrum
#SBATCH --output=OUTPUTTXT/TM1d_chi4+1e-1j_L10_gpr1200_vac_Qsrc5.0e+06_optdof_spectrum.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --error=ERRORS/TM1d_chi4+1e-1j_L10_gpr1200_vac_Qsrc5.0e+06_optdof_spectrum.err

module load anaconda3/2020.11

prog=rampQsrc_TM_dipole_1d_grating_LDOS_spectrum.py

wavelength=1.0
ReChi=4.0
ImChi=0.0

gpr=1200 #pixels per wavelength
omega_pts=2501

zMin=0.0 #design domain
zMax=10.0
pml_sep=1.0
pml_thick=3.0

init_type='file'
init_file='DATA/TM1d_chi4+1e-1j_L10_gpr1200_vac_Qsrc5.0e+06_optdof.txt'
name='DATA/TM1d_chi4+1e-1j_L10_gpr1200_vac_Qsrc5.0e+06_optdof_lossless_spectrum'

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -ReChi $ReChi -ImChi $ImChi -gpr $gpr -omega_pts $omega_pts -zMin $zMin -zMax $zMax -pml_sep $pml_sep -pml_thick $pml_thick -init_type $init_type -init_file $init_file -name $name

