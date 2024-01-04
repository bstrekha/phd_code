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
parser.add_argument('-im_chi',action='store',type=float,default=0.0)
parser.add_argument('-d',action='store',type=float,default=0.1)
args, unknown = parser.parse_known_args()
filename = args.filename

d = args.d
chi = args.re_chi + args.im_chi*1j

Qsrc_list = np.logspace(0, 6, 180)

for k in range(len(Qsrc_list)):
    ktilde = 2*np.pi*(1.0 + 1j/(2.0*Qsrc_list[k]))
    ans = list(map(float,np_TEy_halfspace_bound(d, chi, ktilde)))
    #ans[0] is dual bound
    #ans[1] is optimal theta
    if os.path.exists(filename+'_Qsrc_bounds.npy'):
        more_bounds = np.load(filename+'_Qsrc_bounds.npy')
        more_bounds = np.append(more_bounds, ans[0])
        np.save(filename+'_Qsrc_bounds.npy', more_bounds)
    else:
        more_bounds = np.array([ans[0]])
        np.save(filename+'_Qsrc_bounds.npy', more_bounds)
        
    if os.path.exists(filename+'_Qsrc_list.npy'):
        more_list = np.load(filename+'_Qsrc_list.npy')
        more_list = np.append(more_list, Qsrc_list[k])
        np.save(filename+'_Qsrc_list.npy', more_list)
    else:
        more_list = np.array([Qsrc_list[k]])
        np.save(filename+'_Qsrc_list.npy', more_list)
        
    if os.path.exists(filename+'_Qsrc_thetas.npy'):
        more_thetas = np.load(filename+'_Qsrc_thetas.npy')
        more_thetas = np.append(more_thetas, ans[1])
        np.save(filename+'_Qsrc_thetas.npy', more_thetas)
    else:
        more_thetas = np.array(ans[1])
        np.save(filename+'_Qsrc_thetas.npy', more_thetas)
