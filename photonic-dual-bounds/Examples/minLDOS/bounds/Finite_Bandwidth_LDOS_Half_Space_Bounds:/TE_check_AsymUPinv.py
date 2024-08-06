#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 09:23:14 2022

@author: pengning
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy import integrate

import sympy as sym
from sympy.parsing.sympy_parser import parse_expr
import mpmath
from mpmath import mp

from TE_AsymUP_halfspace import get_TEy_AsymUPinv_expiky_y

kx, kyr, kyi = sym.symbols('kx kyr kyi', real=True)
ky = kyr + sym.I*kyi
kyc = kyr - sym.I*kyi

Pr, Pi = sym.symbols('Pr Pi', real=True)
P = Pr + sym.I*Pi
Pc = Pr - sym.I*Pi

mf = sym.symbols('mf', real=True) #material factor, Im(P/chi^*)

Rx, Ry, r = sym.symbols('Rx Ry r')

symbol_dict = {'kx':kx, 'kyr':kyr, 'kyi':kyi, 'pr':Pr, 'pi':Pi, 'mf':mf,
               'Rx':Rx, 'Ry':Ry, 'r':r, 'I':sym.I}

math_module = ["numpy"]
#math_module = ["mpmath"]; mp.dps=30

#v = Rx*exp(r*y) \hat{x} + Ry*exp(r*y) \hat{y}
str_AsymUPv_xpol_ryCoeff = '(kyi^4*(mf + pi)*Rx - 2*kyi*kyr*pr*r*(r*Rx - I*kx*Ry) + (kyr^2 + r^2)*(kyr^2*(mf + pi)*Rx + r*(mf*r*Rx + I*kx*pi*Ry)) + kyi^2*(2*kyr^2*(mf + pi)*Rx - r*(2*mf*r*Rx + pi*r*Rx + I*kx*pi*Ry)))/(kyi^4 + 2*kyi^2*(kyr^2 - r^2) + (kyr^2 + r^2)^2)'
str_AsymUPv_xpol_ryCoeff = str_AsymUPv_xpol_ryCoeff.replace('^', '**')
AsymUPv_xpol_ryCoeff = parse_expr(str_AsymUPv_xpol_ryCoeff, local_dict=symbol_dict)
AsymUPv_xpol_ryCoeff_func = sym.lambdify([kx, kyr, kyi, Pr, Pi, mf, Rx, Ry, r], AsymUPv_xpol_ryCoeff, modules=math_module)


str_AsymUPv_xpol_ikyCoeff = '-1/4*((pi + I*pr)*(kyi*Rx + I*(-(kyr*Rx) + kx*Ry)))/(kyi - I*kyr + r)'
AsymUPv_xpol_ikyCoeff = parse_expr(str_AsymUPv_xpol_ikyCoeff, local_dict=symbol_dict)
AsymUPv_xpol_ikyCoeff_func = sym.lambdify([kx, kyr, kyi, Pr, Pi, mf, Rx, Ry, r], AsymUPv_xpol_ikyCoeff, modules=math_module)


str_AsymUPv_xpol_mikycCoeff = '-1/4*((pi - I*pr)*(kyi*Rx + I*(kyr*Rx + kx*Ry)))/(kyi + I*kyr + r)'
AsymUPv_xpol_mikycCoeff = parse_expr(str_AsymUPv_xpol_mikycCoeff, local_dict=symbol_dict)
AsymUPv_xpol_mikycCoeff_func = sym.lambdify([kx, kyr, kyi, Pr, Pi, mf, Rx, Ry, r], AsymUPv_xpol_mikycCoeff, modules=math_module)


str_AsymUPv_ypol_ryCoeff = '(I*kx*r*(-(kyi^2*pi) + 2*kyi*kyr*pr + pi*(kyr^2 + r^2))*Rx + kx^2*(-(kyi^2*pi) + 2*kyi*kyr*pr + pi*(kyr^2 + r^2))*Ry + (mf + pi)*(kyi^4 + 2*kyi^2*(kyr^2 - r^2) + (kyr^2 + r^2)^2)*Ry)/(kyi^4 + 2*kyi^2*(kyr^2 - r^2) + (kyr^2 + r^2)^2)'
str_AsymUPv_ypol_ryCoeff = str_AsymUPv_ypol_ryCoeff.replace('^', '**')
AsymUPv_ypol_ryCoeff = parse_expr(str_AsymUPv_ypol_ryCoeff, local_dict=symbol_dict)
AsymUPv_ypol_ryCoeff_func = sym.lambdify([kx, kyr, kyi, Pr, Pi, mf, Rx, Ry, r], AsymUPv_ypol_ryCoeff, modules=math_module)


str_AsymUPv_ypol_ikyCoeff = '(kx*(pi + I*pr)*((-I)*kyi*Rx - kyr*Rx + kx*Ry))/(4*(kyi - I*kyr)*(kyi - I*kyr + r))'
AsymUPv_ypol_ikyCoeff = parse_expr(str_AsymUPv_ypol_ikyCoeff, local_dict=symbol_dict)
AsymUPv_ypol_ikyCoeff_func = sym.lambdify([kx, kyr, kyi, Pr, Pi, mf, Rx, Ry, r], AsymUPv_ypol_ikyCoeff, modules=math_module)


str_AsymUPv_ypol_mikycCoeff = '(kx*(pi - I*pr)*((-I)*kyi*Rx + kyr*Rx + kx*Ry))/(4*(kyi + I*kyr)*(kyi + I*kyr + r))'
AsymUPv_ypol_mikycCoeff = parse_expr(str_AsymUPv_ypol_mikycCoeff, local_dict=symbol_dict)
AsymUPv_ypol_mikycCoeff_func = sym.lambdify([kx, kyr, kyi, Pr, Pi, mf, Rx, Ry, r], AsymUPv_ypol_mikycCoeff, modules=math_module)



def get_AsymUP_Rx_plus_Ry_exp_ry(ygrid, kx, kyr, kyi, Pr, Pi, mf, Rx, Ry, r):
    
    ky = kyr + 1j*kyi
    print('ky get_AsymUP', ky)
    exp_ry = np.exp(r*ygrid)
    exp_iky = np.exp(1j*ky*ygrid)
    exp_mikyc = np.exp(-1j*np.conj(ky)*ygrid)
    
    ywave = np.zeros((2,len(ygrid)), dtype=np.complex)
    ywave[0,:] = (AsymUPv_xpol_ryCoeff_func(kx, kyr, kyi, Pr, Pi, mf, Rx, Ry, r) * exp_ry
                  + AsymUPv_xpol_ikyCoeff_func(kx, kyr, kyi, Pr, Pi, mf, Rx, Ry, r) * exp_iky
                  + AsymUPv_xpol_mikycCoeff_func(kx, kyr, kyi, Pr, Pi, mf, Rx, Ry, r) * exp_mikyc
                  )
    
    ywave[1,:] = (AsymUPv_ypol_ryCoeff_func(kx, kyr, kyi, Pr, Pi, mf, Rx, Ry, r) * exp_ry
                  + AsymUPv_ypol_ikyCoeff_func(kx, kyr, kyi, Pr, Pi, mf, Rx, Ry, r) * exp_iky
                  + AsymUPv_ypol_mikycCoeff_func(kx, kyr, kyi, Pr, Pi, mf, Rx, Ry, r) * exp_mikyc
                  )
    
    return ywave


def plot_TE_cplx_ywave(TE_ywave, ygrid):
    
    TEx_ywave = TE_ywave[0,:]
    TEy_ywave = TE_ywave[1,:]
    
    plt.figure(figsize=(4,16))
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(ncols=4)
    ax1.plot(ygrid, np.real(TEx_ywave))
    ax1.set_title('real x pol')
    ax2.plot(ygrid, np.imag(TEx_ywave))
    ax2.set_title('imag x pol')
    ax3.plot(ygrid, np.real(TEy_ywave))
    ax3.set_title('real y pol')
    ax4.plot(ygrid, np.imag(TEy_ywave))
    ax4.set_title('imag y pol')
    plt.show()

def plot_TE_cplx_ywave_comparison(TE_ywave1, TE_ywave2, ygrid):
    
    TEx_ywave1 = TE_ywave1[0,:]
    TEy_ywave1 = TE_ywave1[1,:]
    
    TEx_ywave2 = TE_ywave2[0,:]
    TEy_ywave2 = TE_ywave2[1,:]
    
    plt.figure(figsize=(4,16))
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(ncols=4)
    ax1.plot(ygrid, np.real(TEx_ywave1))
    ax1.plot(ygrid, np.real(TEx_ywave2), '--')
    ax1.set_title('real x pol')
    ax2.plot(ygrid, np.imag(TEx_ywave1))
    ax2.plot(ygrid, np.imag(TEx_ywave2), '--')
    ax2.set_title('imag x pol')
    ax3.plot(ygrid, np.real(TEy_ywave1))
    ax3.plot(ygrid, np.real(TEy_ywave2), '--')
    ax3.set_title('real y pol')
    ax4.plot(ygrid, np.imag(TEy_ywave1))
    ax4.plot(ygrid, np.imag(TEy_ywave2), '--')
    ax4.set_title('imag y pol')
    plt.show()
    

def check_AsymUPinv(chi, phase, k0, kx, ymax=10):
    """
    check the accuracy of the inverse image of AsymUPinv for TE by comparing
    AsymUP acting on (AsymUPinv S1) and S1
    with S1 here being e^{ik_y y}(-kx \hat{x} + kx**2/ky \hat{y})
    """
    
    ky = np.sqrt(k0**2 - kx**2)
    kyr = np.real(ky); kyi = np.imag(ky)
    Pr = np.real(phase); Pi = np.imag(phase)
    mf = np.imag(phase/np.conj(chi))
    
    print('ky check_AsymUPinv', ky)
    r_list, Rx_list, Ry_list = get_TEy_AsymUPinv_expiky_y(chi, phase, k0, kx)
    
    ygrid = np.linspace(0, ymax, 1000)
    expiky = np.exp(1j*ky*ygrid)
    S1_ywave = np.zeros((2,len(ygrid)), dtype=np.complex)
    S1_ywave[0,:] = -kx * expiky
    S1_ywave[1,:] = (kx**2/ky) * expiky
    
    test_ywave = np.zeros_like(S1_ywave, dtype=np.complex)
    for i in range(len(r_list)):
        r = r_list[i]
        Rx = Rx_list[i]
        Ry = Ry_list[i]
        
        test_ywave += get_AsymUP_Rx_plus_Ry_exp_ry(ygrid, kx, kyr, kyi, Pr, Pi, mf, Rx, Ry, r)
        
    plot_TE_cplx_ywave_comparison(S1_ywave, test_ywave, ygrid)
    
