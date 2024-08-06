#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:12:47 2022

@author: venya
"""

import numpy as np
from get_TM_dipole_circular_grating_LDOS_boundW import get_TM_dipole_circular_grating_LDOS_global_bound
import os.path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-filename',action='store',type=str,default='TMsomename')
parser.add_argument('-re_chi',action='store',type=float,default=5.0)
parser.add_argument('-im_chi',action='store',type=float,default=1.0)
parser.add_argument('-outer_R',action='store',type=float,default=10.0)
args, unknown = parser.parse_known_args()
filename = args.filename

cur_chi = args.re_chi + args.im_chi*1j
outer_R = args.outer_R

print('outerR:', outer_R)

dw_list = np.logspace(1, -7, 320) #in units of w0 

for k in range(len(dw_list)):
    Qsrc = 1.0/2/dw_list[k]
    #args of function are
    #(chi, wvlgth, r_inner, r_outer, pml_sep, pml_thick, gpr, Qabs=np.inf, justAsym=False, opttol=1e-2, fakeSratio=1e-2, iter_period=20)
    ans = list(map(float,get_TM_dipole_circular_grating_LDOS_global_bound(cur_chi, 1.0, 0.0, outer_R, 0.5, 1.5, 400, Qabs=Qsrc)))
    if os.path.exists(filename+'_dw_bounds.npy'):
        more_bounds = np.load(filename+'_dw_bounds.npy')
        more_bounds = np.append(more_bounds, ans[1])
        np.save(filename+'_dw_bounds.npy', more_bounds)
    else:
        more_bounds = np.array([ans[1]])
        np.save(filename+'_dw_bounds.npy', more_bounds)
        
    if os.path.exists(filename+'_dw_list.npy'):
        more_list = np.load(filename+'_dw_list.npy')
        more_list = np.append(more_list, dw_list[k])
        np.save(filename+'_dw_list.npy', more_list)
    else:
        more_list = np.array([dw_list[k]])
        np.save(filename+'_dw_list.npy', more_list)
    
