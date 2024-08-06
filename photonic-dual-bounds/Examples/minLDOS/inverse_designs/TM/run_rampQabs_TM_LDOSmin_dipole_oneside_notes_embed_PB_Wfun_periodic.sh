#!/bin/bash
#SBATCH --job-name=TMLDOSmin4+1e-1j_dwQabs1e-1to1e6_periodicfullspace_N8_period0p6_maxeval500_Wfunc_normalized_vacstart
#SBATCH --output=TMLDOSmin4+1e-1j_dwQabs1e-1to1e6_periodicfullspace_N8_period0p6_maxeval500_Wfunc_normalized_vacstart.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=12000
#SBATCH --error=TMLDOSmin4+1e-1j_dwQabs1e-1to1e6_periodicfullspace_N8_period0p6_maxeval500_Wfunc_normalized_vacstart.err

module purge
module load anaconda3/2023.3
conda activate phd

prog=rampQabs_TM_LDOSmin_dipole_oneside_notes_embed_PB_Wfun_periodic.py

wavelength=1.0
pow10Qabs_start=-1
pow10Qabs_end=7
pow10Qabs_num=17
dwfactor2=1
geometry='Cavity'
#geometry='halfspace'
periodic=1 #whether to make structure periodic 0: false, 1: true
numCellsX=8 #if periodic, how many cells into supercell
numCellsY=8 #if periodic, how many cells into supercell
wavelengthPerCellX=0.6
wavelengthPerCellY=0.6

ReChi=4.0
ImChi=0.1

gpr=100

design_x=4.8
design_y=4.8

vacuum_x=0.0
vacuum_y=0.0
dist_x=0.0

emitter_x=0.025
emitter_y=0.025

pml_thick=0.50
pml_sep=0.50

# cp checkPC_gapdes10by10.txt checkPC_gapdes10by10_test.txt #checkquartcrystalQabs1d0des10by10_orig.txt checkquartcrystalQabs1d0des10by10.txt
#TMLDOSmin4+1j_Qabs0e1to1e6_des10by10_periodicfullspace_maxeval1000_Wfunc_normalized_vacstartDESIGN
init_type='vac'
init_file='TMLDOSmin4+1e-1j_dwQabs1e-1to1e6_periodicfullspace_N8_period0p6_maxeval500_Wfunc_normalized_vacstartDESIGN.txt'

output_base=2000
maxeval=500
name='TMLDOSmin4+1e-1j_dwQabs1e-1to1e6_periodicfullspace_N8_period0p6_maxeval500_Wfunc_normalized_vacstart'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $prog -wavelength $wavelength -pow10Qabs_start $pow10Qabs_start -pow10Qabs_end $pow10Qabs_end -pow10Qabs_num $pow10Qabs_num -dwfactor2 $dwfactor2 -ReChi $ReChi -ImChi $ImChi -gpr $gpr -design_x $design_x -design_y $design_y -vacuum_x $vacuum_x -vacuum_y $vacuum_y -emitter_x $emitter_x -emitter_y $emitter_y -pml_thick $pml_thick -pml_sep $pml_sep -init_type $init_type -init_file $init_file -output_base $output_base -name $name -maxeval $maxeval -dist_x $dist_x -geometry $geometry -periodic $periodic -numCellsX $numCellsX -numCellsY $numCellsY -wavelengthPerCellX $wavelengthPerCellX -wavelengthPerCellY $wavelengthPerCellY

