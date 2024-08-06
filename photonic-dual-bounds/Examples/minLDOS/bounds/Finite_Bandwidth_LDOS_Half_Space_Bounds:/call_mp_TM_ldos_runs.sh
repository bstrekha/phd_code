#!/bin/bash
#SBATCH --job-name=chi4+0i
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --error=LDOS_maxx_errors.err
#SBATCH --partition=photon-planck
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=bstrekha@princeton.edu

prog=get_mp_TM_data_bounds.py
filename='mp_TM_Qsrc_run_chi4+1e-1i'
re_chi=4.0
im_chi=0.1


python3 $prog -filename $filename -re_chi $re_chi -im_chi $im_chi

