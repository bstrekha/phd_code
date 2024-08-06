import numpy as np 
import subprocess
import os
from shlex import split
current_path = os.path.dirname(os.path.realpath(__file__))

def sendtoHPC(ram, time, chifRe, chifIm, chidRe, chidIm, wavelength, des_region, design_x, design_y, 
              nprojx, nprojy, des_params, gpr, obj, save, opttol, Qabs, fakeSratio, iter_period,
              pml_sep, pml_thick, do_checks):
    command = f'bash sp_all_parallel_cloak_bound_2sides.sh -p {ram} -t {time} -r {chifRe} -i {chifIm} -R {chidRe} -I {chidIm} -w {wavelength} -e {des_region} -x {design_x} -y {design_y} -D {des_params[0]} -E {des_params[1]} -g {gpr} -o {obj} -s {save} -O {opttol} -Q {Qabs} -F {fakeSratio} -T {iter_period} -m {pml_sep} -M {pml_thick} -c {do_checks} -a {nprojx} -b {nprojy}'
    command = split(command)
    subprocess.run(command, cwd=current_path)
    return 0

# Example submit 
# This is effectively a python interface for submitting jobs on the cluster. Here is how it works:
# 1. In this script, you can call sendtoHPC for a given set of parameters. For those set, you should choose an appropriate ram (in mb) and time (in hours)
# 2. The sendtoHPC function will then call the bash script with the appropriate parameters and submit the job 
# 3. Every single call to sendtoHPC will be a separate job on the cluster. This means you can run many jobs in parallel!

#Qabs runs
Rinner=0.25
Router=0.75
wavelength=1.0
design_x=1.5
design_y=1.5
nprojx=12
nprojy=12
gpr=25
pml_sep=0.5
pml_thick=0.5
run_time_hrs=72
mem_mbs=400
opttol=1e-5
save=1 #0 means no. 1 means yes
fakeSratio=0.025
iter_period=120
do_checks=False
des_params=[Rinner, Router]
# for chif in [-4.0 + 1j]:
for chif in [4.0 + 1e-3j]:
    for chid in [5.0 + 1e-3j]:
        for Qabs in np.flip(np.logspace(2, 7, 31))/2.0:
# for chif in [4.0 + 1e-3j, -4 + 1j]:
#     for chid in [3.0 + 1e-3j, 5.0 + 1e-3j]:
#         for Qabs in np.flip(np.logspace(2, 7, 31))/2.0:
            sendtoHPC(mem_mbs, run_time_hrs, np.real(chif), np.imag(chif), np.real(chid), np.imag(chid), wavelength, 'circle', design_x, design_y, nprojx, nprojy, des_params, gpr, 'EXT', save, opttol, Qabs, fakeSratio, iter_period, pml_sep, pml_thick, do_checks)