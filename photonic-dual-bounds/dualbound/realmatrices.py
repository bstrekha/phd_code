#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:35:24 2022

@author: jewelmohajan
"""

import numpy as np
import scipy.sparse as sp

def cmplxtoreal_matrix(A):
    realA=np.real(A)
    imagA=np.imag(A)
    h1=sp.hstack((realA,-imagA))#A1=np.append(realA,-imagA,axis=1)
    h2=sp.hstack((imagA,realA))#A2=np.append(imagA,realA,axis=1)
    reA=sp.vstack((h1,h2))
    return reA

def cmplxtoreal_vector(v):
    realA=np.real(v)
    imagA=np.imag(v)
    rev=np.append(realA,imagA,axis=0)
    return rev


def conjugation_matrix(n): #dimension n
    A=np.append(np.append(np.eye(n),np.zeros((n,n)),axis=1),np.append(np.zeros((n,n)),-np.eye(n),axis=1),axis=0)
    return A

def cmplx_multiplier_matrix(n,a,b): # real part a, imag part b; z=a+ib; dimension n
    A=sp.vstack((sp.hstack((a*sp.eye(n),-b*sp.eye(n))),sp.hstack((b*sp.eye(n),a*sp.eye(n)))))
    return A

def imag_innerproduct_matrix(n):
    A=np.append(np.append(np.zeros((n,n)),np.eye(n),axis=1),np.append(np.eye(n),np.zeros((n,n)),axis=1),axis=0)
    return A

def real_innerproduct_matrix(n):
    A=np.append(np.append(np.eye(n),np.zeros((n,n)),axis=1),np.append(np.zeros((n,n)),-np.eye(n),axis=1),axis=0)
    return A

#conjugation_matrix=np.append(np.append(np.eye(n),np.zeros((n,n)),axis=1),np.append(np.zeros((n,n)),-np.eye(n),axis=1),axis=0)
#cmplx_multiplier_matrix=np.append(np.append(a*np.eye(n),-b*np.eye(n),axis=1),np.append(b*np.eye(n),a*np.eye(n),axis=1),axis=0)