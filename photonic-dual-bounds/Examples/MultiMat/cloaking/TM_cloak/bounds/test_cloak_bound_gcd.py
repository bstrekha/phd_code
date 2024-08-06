import numpy as np 
import subprocess
import os
from shlex import split
current_path = os.path.dirname(os.path.realpath(__file__))

def sendtoHPC(ram, time, chifRe, chifIm, chidRe, chidIm, des, wavelength, des_region, design_x, design_y, 
              des_params, gpr, obj, save, opttol, Qabs, fakeSratio, iter_period,
              Pnum, gcd_maxiter, pml_sep, pml_thick, do_checks):
    command = f'bash sp_all_cloak_bound_gcd.sh -p {ram} -t {time} -r {chifRe} -i {chifIm} -R {chidRe} -I {chidIm} -d {des} -w {wavelength} -e {des_region} -x {design_x} -y {design_y} -D {des_params[0]} {des_params[1]} -g {gpr} -o {obj} -s {save} -O {opttol} -Q {Qabs} -F {fakeSratio} -T {iter_period} -P {Pnum} -G {gcd_maxiter} -m {pml_sep} -M {pml_thick} -c {do_checks}'
    command = split(command)
    subprocess.run(command, cwd=current_path)
    return 0

# Example submit 
# This is effectively a python interface for submitting jobs on the cluster. Here is how it works:
# 1. In this script, you can call sendtoHPC for a given set of parameters. For those set, you should choose an appropriate ram (in mb) and time (in hours)
# 2. The sendtoHPC function will then call the bash script with the appropriate parameters and submit the job 
# 3. Every single call to sendtoHPC will be a separate job on the cluster. This means you can run many jobs in parallel!
for chifRe in [2.0, 2.5, 3.0]:
    sendtoHPC(1, 1, chifRe, 0.01, 4.0, 0.0001, 1.0, 1.0, 'circle', 1.0, 1.0, [0.25, 0.5], 20, 'EXT', 1, 1e-4, np.inf, 1e-3, 80, 10, 20, 0.5, 0.5, False)