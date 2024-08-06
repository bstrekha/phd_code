#!/bin/bash
#SBATCH --job-name=TM_cyl_chi4+1e-1j_R32_dw_sweep
#SBATCH --output=OUTPUTTXT/TM_cyl_chi4+1e-1j_R32_dw_sweep.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=25000
#SBATCH --error=ERRORS/TM_cyl_chi4+1e-1j_R32_dw_sweep_errors.err

module purge
module load anaconda3/2020.11

prog=sweep_TM_circular_grating_bandwidth.py
filename='DATA/TM_circular_grating_dw_run_R32_chi4+1e-1i'
re_chi=4.0
im_chi=0.1
outer_R=32


python3 $prog -filename $filename -re_chi $re_chi -im_chi $im_chi -outer_R $outer_R