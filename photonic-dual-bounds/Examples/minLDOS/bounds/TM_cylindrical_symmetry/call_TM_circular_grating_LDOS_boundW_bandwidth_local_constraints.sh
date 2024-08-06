#!/bin/bash
#SBATCH --job-name=TM_local_sub0p1lambda_cyl_chi4+1e-1j_R8_dw_sweep
#SBATCH --output=OUTPUTTXT/TM_local_sub0p1lambda_cyl_chi4+1e-1j_R8_dw_sweep.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=14000
#SBATCH --error=ERRORS/TM_local_sub0p1lambda_cyl_chi4+1e-1j_R8_dw_sweep_errors.err

module purge
module load anaconda3/2020.11

prog=sweep_TM_circular_grating_bandwidth_local_constraints.py
filename='DATA/TM_local_sub0p1lambda_circular_grating_dw_run_R8_chi4+1e-1i'
re_chi=4.0
im_chi=0.1
outer_R=8
subreg_per_lambda=1.0 #number of subregions to add as local projection constraints per wavelength

python3 $prog -filename $filename -re_chi $re_chi -im_chi $im_chi -outer_R $outer_R -subreg_per_lambda $subreg_per_lambda