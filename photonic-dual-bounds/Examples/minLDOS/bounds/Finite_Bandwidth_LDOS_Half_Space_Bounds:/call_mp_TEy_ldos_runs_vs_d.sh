#!/bin/bash
#SBATCH --job-name=chi4+0i
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=110:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --error=LDOS_min_errors.err

module load anaconda3/2022.5

prog=get_mp_TEy_data_bounds_vs_d.py
filename='mp_TEy_drun_Qsrc10000_chi4+0i'
re_chi=4
im_chi=0
Qsrc=10000

python3 $prog -filename $filename -re_chi $re_chi -im_chi $im_chi -Qsrc $Qsrc

