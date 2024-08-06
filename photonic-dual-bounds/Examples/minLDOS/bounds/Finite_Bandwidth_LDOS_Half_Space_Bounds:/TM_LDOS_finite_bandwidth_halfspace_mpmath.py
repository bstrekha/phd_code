#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:58:35 2022

@author: pengning
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import quad

import mpmath
from mpmath import mp


def positiveRe_Laplace_poles(chi, phase, kyr, kyi, Br, Bi):
    mat_fac = mp.im(phase/mp.conj(chi))
    BK = (Br*kyi + Bi*kyr)
    Delta = BK**2 + (8*mat_fac) * (Bi*kyr*kyi**2 - Br*kyr**2*kyi - mat_fac*kyr**2*kyi**2)
    #print('BK', BK, 'Delta', Delta)
    if Delta<0:
        sqrt_Delta = 1j*mp.sqrt(-Delta)
    else:
        sqrt_Delta = mp.sqrt(Delta)
    sqr_plus = kyi**2 - kyr**2 + (0.5/mat_fac) * (BK + sqrt_Delta)
    sqr_minus =kyi**2 - kyr**2 + (0.5/mat_fac) * (BK - sqrt_Delta)
    
    if (mp.re(sqr_plus)<0 and mp.fabs(mp.im(sqr_plus))<1000*mp.eps) or (mp.re(sqr_minus)<0 and mp.fabs(mp.im(sqr_minus))<1000*mp.eps):
        raise ValueError('L2 invertibility lost')
        
    #print('sqr_plus', sqr_plus, 'sqr_minus', sqr_minus)
    pole_plus = mp.sqrt(sqr_plus)
    pole_minus = mp.sqrt(sqr_minus)
    
    
    pole_minus *= mp.sign(mp.re(pole_minus))
    pole_plus *= mp.sign(mp.re(pole_plus)) #take the Re>0 root
    
    return pole_plus, pole_minus


def Laplace_free_params(kyr, kyi, Br, Bi, pole_plus, pole_minus):
    Amat = mp.matrix(2)
    Amat[0,0] = (Br+1j*Bi)*0.25 * (pole_plus + (kyi-1j*kyr)) * (pole_plus**2 - (kyi+1j*kyr)**2)
    Amat[0,1] = (Br-1j*Bi)*0.25 * (pole_plus + (kyi+1j*kyr)) * (pole_plus**2 - (kyi-1j*kyr)**2)
    
    Amat[1,0] = (Br+1j*Bi)*0.25 * (pole_minus + (kyi-1j*kyr)) * (pole_minus**2 - (kyi+1j*kyr)**2)
    Amat[1,1] = (Br-1j*Bi)*0.25 * (pole_minus + (kyi+1j*kyr)) * (pole_minus**2 - (kyi-1j*kyr)**2)

    bvec = mp.matrix(2,1)
    bvec[0] = (pole_plus - (kyi-1j*kyr)) * (pole_plus**2 - (kyi+1j*kyr)**2)
    bvec[1] = (pole_minus - (kyi-1j*kyr)) * (pole_minus**2 - (kyi+1j*kyr)**2)

    gamma_list = mp.lu_solve(Amat, bvec)
    gamma_plus = gamma_list[1]
    gamma_minus = gamma_list[0]
    
    return gamma_plus, gamma_minus


def Laplace_residue_coeffs(chi, phase, kyr, kyi, Br, Bi, pole_plus, pole_minus, gamma_plus, gamma_minus):
    mat_fac = mp.im(phase / np.conj(chi))
    denom_plus = - mat_fac * 2 * pole_plus * (pole_plus**2 - pole_minus**2)
    num_plus = ( (-pole_plus-(kyi-1j*kyr)) * (pole_plus**2 - (kyi+1j*kyr)**2)
                - 0.25*(Br+1j*Bi) * (-pole_plus+(kyi-1j*kyr))*(pole_plus**2-(kyi+1j*kyr)**2)*gamma_minus
                - 0.25*(Br-1j*Bi) * (-pole_plus+(kyi+1j*kyr))*(pole_plus**2-(kyi-1j*kyr)**2)*gamma_plus
                )
    
    R_plus = num_plus / denom_plus
    
    denom_minus = - mat_fac * 2 * pole_minus * (pole_minus**2 - pole_plus**2)
    num_minus = ( (-pole_minus-(kyi-1j*kyr))*(pole_minus**2-(kyi+1j*kyr)**2) 
                 - 0.25*(Br+1j*Bi) * (-pole_minus+(kyi-1j*kyr))*(pole_minus**2-(kyi+1j*kyr)**2)*gamma_minus
                 - 0.25*(Br-1j*Bi) * (-pole_minus+(kyi+1j*kyr))*(pole_minus**2-(kyi-1j*kyr)**2)*gamma_plus
                )
    
    R_minus = num_minus / denom_minus
    
    return R_plus, R_minus


def bound_integrand_Cauchy_relaxed(d, chi, phase, k0, kx):
    """
    calculates the kx integrand 
    for the finite bandwidth TM LDOS bound near a half space with the
    Cauchy relaxation |<S2| AsymUP^-1 |S1>| <= |<S1| AsymUP^-1 |S1>| applied
    """
    k0r = mp.re(k0); k0i = mp.im(k0)
    PA = mp.sqrt(mp.re(phase*mp.conj(phase)))
    
    ky = mp.sqrt(k0**2 - kx**2)
    kyr = mp.re(ky); kyi = mp.im(ky)

    B = mp.conj(phase)*k0**2 / ky
    Br = mp.re(B); Bi = mp.im(B)
    
    pole_plus, pole_minus = positiveRe_Laplace_poles(chi, phase, kyr, kyi, Br, Bi)
    
    gamma_plus, gamma_minus = Laplace_free_params(kyr, kyi, Br, Bi, pole_plus, pole_minus)
        
    R_plus, R_minus = Laplace_residue_coeffs(chi, phase, kyr, kyi, Br, Bi, pole_plus, pole_minus, gamma_plus, gamma_minus)
    
    return PA * (k0r**2+k0i**2)**1.5 * mp.exp(-2*kyi*d)/(kyr**2+kyi**2) * mp.re(R_plus/(pole_plus+(kyi+1j*kyr)) + R_minus/(pole_minus+(kyi+1j*kyr)))


def bound_integrand(d, chi, phase, k0, kx, toDouble=False):
    """
    calculates the kx integrand 
    for the finite bandwidth TM LDOS bound near a half space
    """
    
    k0r = mp.re(k0); k0i = mp.im(k0)
    PA = mp.sqrt(mp.re(phase*mp.conj(phase)))
    
    ky = mp.sqrt(k0**2 - kx**2)
    kyr = mp.re(ky); kyi = mp.im(ky)

    B = mp.conj(phase)*k0**2 / ky
    Br = mp.re(B); Bi = mp.im(B)
    
    pole_plus, pole_minus = positiveRe_Laplace_poles(chi, phase, kyr, kyi, Br, Bi)
    
    #if np.isnan(pole_plus) or np.isnan(pole_minus):
    #    print('d', d, 'chi', chi, 'cratio', cratio, 'k0', k0, 'kx', kx, flush=True)
    
    gamma_plus, gamma_minus = Laplace_free_params(kyr, kyi, Br, Bi, pole_plus, pole_minus)
        
    R_plus, R_minus = Laplace_residue_coeffs(chi, phase, kyr, kyi, Br, Bi, pole_plus, pole_minus, gamma_plus, gamma_minus)
   
    wtil = k0r + 1j*k0i
    Ntil = k0r/(k0r**2 + k0i**2)
    S2AinvS1_integrand = 0.5*mp.re(mp.conj(phase) * (k0**3/ky**2) * mp.exp(2j*ky*d) * (R_plus/(pole_plus+kyi-1j*kyr) + R_minus/(pole_minus+kyi-1j*kyr)) / (wtil*Ntil) )
    S1AinvS1_integrand = 0.5*PA * (k0r**2+k0i**2)**1.5/(kyr**2+kyi**2) * mp.exp(-2*kyi*d) * mp.re( (R_plus/(pole_plus+(kyi+1j*kyr)) + R_minus/(pole_minus+(kyi+1j*kyr))) / (mp.sqrt(k0r**2+k0i**2)*Ntil) )
    
    if toDouble:
        return np.double(S2AinvS1_integrand + S1AinvS1_integrand)
    else:
        return S2AinvS1_integrand + S1AinvS1_integrand



def TM_halfspace_fixed_phase_bound(d, chi, phase, k0):
    """
    evaluate the integral in 3 parts: the traveling regime, the lightline regime, the evanescent regime
    if necessary in future adjust so large bandwidths don't raise error
    """
    k0r = mp.re(k0); k0i = mp.im(k0)
    integrand = lambda kx: bound_integrand(d, chi, phase, k0, kx)
    
    end_kx = 3*k0r+50*k0i
    integral = mp.quad(integrand, [0, k0r, k0r+50*k0i, end_kx])

    delta_kx = 10*k0r
    while True: #keep on integrating the deep evanescent tail until desired accuracy is reached
        delta_integral = mp.quad(integrand, [end_kx, end_kx+delta_kx])
        integral += delta_integral
        if mp.fabs(delta_integral / integral) < 1e-5:
            break
        end_kx += delta_kx
        
    #extra factor of 2 for the symmetric integral
    integral *= 2
    
    #prefactor
    wvlgth0 = 2*mp.pi / k0r
    rhovac0 = 2*mp.pi/8
    rho0 = (k0r**2 + k0i**2)/k0r/(4*mp.pi)*mp.atan2(k0r, k0i)
    #return 1 + (wvlgth0/4/mp.pi**2) * rhovac0/rho0 * integral #the 1 is the constant vacuum contribution
    return (wvlgth0/4/mp.pi**2) * rhovac0/rho0 * integral #the 1 is the constant vacuum contribution

def mp_ternary_search_min(func, a, b, abstol):
    # use ternary search to find minimum of unary func
    
    l = a
    r = b
    fl = func(a)
    fr = func(b)
    while True:
        m1 = l + (r-l)/3; m2 = m1 + (r-l)/3
        fm1 = func(m1); fm2 = func(m2)
        if fm1<fm2:
            r = m2
        else:
            l = m1
        if (r-l)<abstol:
            break
        print('l', l, 'r', r)
    return (l+r)/2, func((l+r)/2)


def TM_halfspace_bound(d, chi, k0):
    """
    calculate the tightest possible dual bound for a TM dipole near a half-space design domain
    given the complex global energy conservation constraints
    for dielectrics chi should have a little loss to avoid division by 0 in the numerics
    use lossless dielectrics code for real positive chi
    """
    theta_boundfunc = lambda angle: TM_halfspace_fixed_phase_bound(d, chi, mp.exp(1j*angle), k0)
    
    #phase angle -pi < theta < pi; find upper and lower limits on phase angle theta
    delta_theta = mp.im(k0) / mp.re(k0) / 2
    theta_r = 1.3*delta_theta
    probe_bound = theta_boundfunc(theta_r)
    
    while True: #find upper bound on optimal phase angle theta
        reduced_stepsize = False
        while True:
            if theta_r+delta_theta > np.pi:
                delta_theta = 2*(mp.pi-theta_r)/3.0
                reduced_stepsize = True
                continue
            try:
                t = theta_boundfunc(theta_r+delta_theta)
                break
            except ValueError: #inverse of AsymUP acting on S1 is not in L2
                delta_theta /= 2
                reduced_stepsize = True
            
        theta_r += delta_theta
        #print('theta_r', theta_r)
        if not reduced_stepsize:
            delta_theta *= 2
        if t>probe_bound:
            break
        probe_bound = t
        
    
    theta_l = 2*theta_r/3
    probe_bound = theta_boundfunc(theta_l)
    delta_theta = theta_r / 2.0
    onBoundary = False
    while True: #find lower bound on optimal phase angle theta
        reduced_stepsize = False
        while True:
            if theta_l - delta_theta < -np.pi:
                delta_theta = 2*(theta_l+np.pi)/3.0
                reduced_stepsize = True
                continue
            try:
                t = theta_boundfunc(theta_l-delta_theta)
                break
            except ValueError: #inverse of AsymUP acting on S1 is not in L2
                delta_theta /= 2
                reduced_stepsize = True
        
        theta_l -= delta_theta
        #print('theta_l', theta_l, 'probe_bound', probe_bound, 't', t)
        if not reduced_stepsize:
            delta_theta *= 2
        if t>probe_bound:
            break
        if reduced_stepsize and delta_theta < max(mp.im(k0)/mp.re(k0), mp.fabs(theta_l))*1e-3:
            onBoundary = True
            break
        probe_bound = t
        
    if onBoundary:
        print('on L2 invertibility boundary', flush=True)
        theta_opt = theta_l
        bound = theta_boundfunc(theta_opt)
    else:
        print('do ternary search to find theta_opt', flush=True)
        theta_opt, bound = mp_ternary_search_min(theta_boundfunc, theta_l, theta_r, (theta_r-theta_l)*1e-3)
        
    print('theta_l', theta_l, 'theta_opt', theta_opt, 'theta_r', theta_r)
  
    return bound, theta_opt

"""
below are functions for plotting various quantities for comparison / study / debugging
"""

def plot_bound_depending_on_theta(d, chi, k0, theta_min, theta_max):

    theta_list = np.linspace(theta_min, theta_max, 100)
    bound_list = np.zeros_like(theta_list)
    for i, theta in enumerate(theta_list):
        try:
            bound_list[i] = TM_halfspace_fixed_phase_bound(d, chi, np.exp(1j*theta), k0)
        except ValueError:
            break
    
    plt.figure()
    plt.plot(theta_list, bound_list)
    plt.xlabel('theta')
    plt.ylabel('Bound on LDOS enh')
    plt.title('chi'+str(chi)+'d'+str(d))
    plt.show()
    
    min_ind = np.argmin(bound_list)
    print('minimum bound found', bound_list[min_ind], 'with theta', theta_list[min_ind])
    

