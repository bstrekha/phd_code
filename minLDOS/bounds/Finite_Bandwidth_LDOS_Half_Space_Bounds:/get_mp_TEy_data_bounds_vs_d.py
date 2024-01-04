#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:12:47 2022

@author: venya
"""

import numpy as np
from TE_AsymUP_halfspace_mpmath import np_TEy_halfspace_bound
import os.path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-filename',action='store',type=str,default='TMsomename')
parser.add_argument('-re_chi',action='store',type=float,default=4.0)
parser.add_argument('-im_chi',action='store',type=float,default=0.1)
parser.add_argument('-Qsrc',action='store',type=float,default=10000.0)
args, unknown = parser.parse_known_args()
filename = args.filename

cur_chi = args.re_chi + args.im_chi*1j
Qsrc = args.Qsrc
cur_k0 = 2*np.pi*(1.0 + 1j/(2.0*Qsrc))

d_list = np.logspace(3, -1, 120)

for k in range(len(d_list)):
    cur_d = d_list[k]
    ans = list(map(float,np_TEy_halfspace_bound(cur_d, cur_chi, cur_k0)))
    if os.path.exists(filename+'_d_bounds.npy'):
        more_bounds = np.load(filename+'_d_bounds.npy')
        more_bounds = np.append(more_bounds, ans[0])
        np.save(filename+'_d_bounds.npy', more_bounds)
    else:
        more_bounds = np.array([ans[0]])
        np.save(filename+'_d_bounds.npy', more_bounds)
        
    if os.path.exists(filename+'_d_list.npy'):
        more_list = np.load(filename+'_d_list.npy')
        more_list = np.append(more_list, d_list[k])
        np.save(filename+'_d_list.npy', more_list)
    else:
        more_list = np.array([d_list[k]])
        np.save(filename+'_d_list.npy', more_list)
