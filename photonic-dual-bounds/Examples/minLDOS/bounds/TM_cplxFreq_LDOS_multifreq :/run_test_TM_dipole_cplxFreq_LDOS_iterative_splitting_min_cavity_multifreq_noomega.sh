#!/bin/bash
#SBATCH --job-name=TMLDOSmin1d0by1d0cav_gpr40_noomega_1e6_vacPrad
#SBATCH --output=LDOSmin_Msparse_Qabs1e6_chi4+0j_gpr40_Des1d0by1d0_Cavity0d1by0d1_noomega_vacPrad.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=1800
#SBATCH --error=error1d0by1d0mincav_1e6_gpr40_noomega_vacPrad.err

#module purge
#module load anaconda3/2020.11

prog=test_TM_dipole_cplxFreq_LDOS_iterative_splitting_min_cavity_multifreq_noomega.py

Qabs=1
Qabstol=100
ReChi=4
ImChi=0.1

wavelength=1
gpr=40
design_x=1.0
design_y=1.0
vacuum_x=0.0
vacuum_y=0.0
emitter_x=0.025
emitter_y=0.025
box_L=1.0 #assume equal side lengths
design_x=$box_L
design_y=$box_L #leave option to change size later
dist=0.0

pml_sep=0.5
pml_thick=0.5

Num_Poles=1
geometry='Cavity'
alg='Newton'
lag_input='Lags_1d0by1d0mincav_1e6_gpr40_noomega_vacPrad.txt'
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#mpiexec -n $SLURM_NTASKS_PER_NODE python test_pixel_basis_idmap_bounds.py

python $prog -wavelength $wavelength -Qabs $Qabs -ReChi $ReChi -ImChi $ImChi -gpr $gpr -design_x $design_x -design_y $design_y -dist $dist -pml_sep $pml_sep -pml_thick $pml_thick -alg $alg -lag_input $lag_input -vacuum_x $vacuum_x -vacuum_y $vacuum_y -emitter_x $emitter_x -emitter_y $emitter_y -Num_Poles $Num_Poles -geometry $geometry -Qabstol $Qabstol
