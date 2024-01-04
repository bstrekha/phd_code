#!/bin/bash
#SBATCH --job-name=TEST_chi4+1e-3i
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=110:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --error=LDOS_min_errors.err

module load anaconda3/2022.5

prog=get_mp_TEy_data_bounds.py
filename='TEST_mp_TEy_Qsrc_run_d1e-1_chi4+1e-3i'
re_chi=4
im_chi=0.001
d=0.1

python3 $prog -filename $filename -re_chi $re_chi -im_chi $im_chi -d $d
