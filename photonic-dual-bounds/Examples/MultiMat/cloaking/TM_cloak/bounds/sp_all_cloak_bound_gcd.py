#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('../../../../../')
np.set_printoptions(threshold=sys.maxsize)
import time, sys, argparse
from dualbound.Maxwell import TM_FDFD as TM
from dualbound.Lagrangian.zops_utils import zops_sparse_all as zsp
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dualbound.Optimization.fakeSource_with_restart_singlematrix import fakeS_with_restart_singlematrix as full_Newton
from dualbound.Optimization.fakeSource_with_restart_singlematrix import fakeS_single_restart_Newton as Newton
from dualbound.Lagrangian import dualgrad_sparse_all as dualgrad
from dualbound.Constraints.gcd_sparse import gcd_sparse_all

def sp_get_bandwidth_cloak_bound(chif, chid, wvlgth, des_region, des_params, obj, design_x, design_y, gpr, 
                                pml_sep, pml_thick, opt_params, Qabs=np.inf, TESTS={'just_mask': False}, do_checks=False, normalize_result=True):
    opttol, fakeSratio, iter_period, gcd_maxiter, Pnum = opt_params['opttol'], opt_params['fakeSratio'], opt_params['iter_period'], opt_params['gcd_maxiter'], opt_params['Pnum']
    save_func = opt_params['save_func']
    assert(des_region in ['rect', 'circle'])
    assert(obj in ['EXT'])
    
    dl = 1.0/gpr
    Mx = int(np.round(design_x/dl))
    My = int(np.round(design_y/dl))
    Npml = int(np.round(pml_thick/dl))
    Npmlsep = int(np.round(pml_sep/dl))
    nonpmlNx = Mx + 2*Npmlsep
    nonpmlNy = My + 2*Npmlsep
    Nx = nonpmlNx + 2*Npml
    Ny = nonpmlNy + 2*Npml
    wv = wvlgth
    omega = 2*np.pi/wv*(1 + 1j/2/Qabs)
    
    background_mask = np.zeros((nonpmlNx,nonpmlNy), dtype=bool) #the fixed object, which is considered to be part of background
    if des_region == 'rect':
        LxN = int(des_params[0]/dl)
        LyN = int(des_params[1]/dl)
        background_mask[Npmlsep + Mx//2 - LxN//2:Npmlsep + Mx//2 + LxN//4, Npmlsep + My//2 - LyN//2:Npmlsep + My//2 + LyN//2] = True
    elif des_region == 'circle':
        originx = Npmlsep + Mx//2
        originy = Npmlsep + My//2
        radius = des_params[0]
        assert(radius >= 0)
        assert(radius <= min(design_x/2, design_y/2))
        radiusN = int(radius/dl)
        for i in range(radiusN):
            for j in range(radiusN):
                if np.sqrt(i**2 + j**2) <= radiusN:
                    background_mask[originx+i, originy+j] = True
                    background_mask[originx+i, originy-j] = True
                    background_mask[originx-i, originy+j] = True
                    background_mask[originx-i, originy-j] = True
    
    #background = np.zeros((nonpmlNx,nonpmlNy), dtype=bool) #can check bounds with only design region.
    if do_checks:
        config2 = np.zeros((nonpmlNx,nonpmlNy))
        config2[background_mask] = 2.0
        plt.imshow(config2)
        plt.savefig('check_background.png')
        
    design_mask = np.zeros((nonpmlNx,nonpmlNy), dtype=bool)

    if des_region == 'rect':
        design_mask[Npmlsep:Npmlsep+Mx, Npmlsep:Npmlsep+My] = True
        design_mask[background_mask] = False
    elif des_region == 'circle':
        originx = Npmlsep+Mx//2
        originy = Npmlsep+My//2
        radius = des_params[1]
        assert(radius >= 0)
        assert(radius <= min(design_x/2, design_y/2))
        radiusN = int(radius/dl)
        for i in range(radiusN):
            for j in range(radiusN):
                if np.sqrt(i**2 + j**2) <= radiusN:
                    design_mask[originx+i, originy+j] = True
                    design_mask[originx+i, originy-j] = True
                    design_mask[originx-i, originy+j] = True
                    design_mask[originx-i, originy-j] = True
        design_mask[background_mask] = False

    if do_checks:
        config = np.zeros((nonpmlNx,nonpmlNy))
        config[design_mask] = 1.0
        plt.imshow(config)
        plt.savefig('check_design.png')
    
    if do_checks:
        config3 = np.zeros((nonpmlNx,nonpmlNy))
        config3[design_mask] = 1.0
        config3[background_mask] += 2.0
        plt.imshow(config3)
        plt.savefig('check_design+background.png')

    if TESTS['just_mask']:
        return design_mask, background_mask, nonpmlNx, nonpmlNy, Npml, Npmlsep, dl
        
    background_chi = background_mask * chif #change from boolean mask of fixed object to complex numeric array with chi value

    print(f"Making Gdd for lambda={wv}", flush=True)
    Gfddinv, _ = TM.get_Gddinv(wv, dl, nonpmlNx, nonpmlNy, (Npml, Npml), design_mask, Qabs=Qabs, chigrid=background_chi)
    N = Gfddinv.shape[0] #number of grid points in design region
    print('N: ', N, flush=True)
    
    ################# calculate the fixed vectors #################
    Z = 1.0 # dimensionless units
    C_0 = 1.0
    
    big_background_chi = np.zeros((Nx, Ny), dtype=complex) #background chi over grid including pml region
    big_background_chi[Npml:-Npml, Npml:-Npml] = background_chi[:,:]
        
    big_design_mask = np.zeros((Nx, Ny), dtype=bool)
    big_design_mask[Npml:-Npml, Npml:-Npml] = design_mask[:,:] #design mask over grid including pml region

    #generate an initial plane wave.
    cx = Npml + Npmlsep//2 #position of current sheet
    Ei = TM.get_TM_linesource_field(wv, dl, Nx, Ny, cx, Npml, bloch_x=0.0, bloch_y=0.0, amp=1.0, Qabs=Qabs, chigrid=None) #plane wave in vacuum

    if do_checks:
        #should look like planewave
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        ax1.imshow(np.real(Ei), cmap='RdBu')
        ax2.imshow(np.imag(Ei), cmap='RdBu')
        plt.savefig('test_TM_planewave_Ei.png')
    
    G0inv = TM.get_TM_MaxwellOp(wv, dl, Nx, Ny, Npml, bloch_x=0, bloch_y=0, Qabs=Qabs) / omega**2
    Gfinv = G0inv + (TM.get_diagM_from_chigrid(omega, big_background_chi) / omega**2)
    
    #we need |S1> = G_{0}^{-1} G_{f} V_{f} |E^i>
    S1 = big_background_chi*Ei
    S1 = S1.flatten()
    S1 = spla.spsolve(Gfinv, S1)
    S1 = G0inv @ S1
    EiS1 = np.vdot(Ei, S1) * dl**2
    background_term = np.imag(EiS1*omega/2/Z)
    if not save_func is None:
        saveFunc = lambda iternum, optdual, dualconst, optLags, optgrad, Plist : save_func(iternum, optdual, dualconst, optLags, optgrad, Plist, background_term)
    else: 
        saveFunc = save_func
    
    #we need |S3> = G_{f,dd}^{-1} G_{f} G_{0}^{-1} |E^{i*}>
    S3 = G0inv @ np.conj(Ei.flatten())
    S3 = spla.spsolve(Gfinv, S3)
    #Gddinv, _ = TM.get_Gddinv(wv, dl, Nx, Ny, Npml, design_mask, bloch_x=0, bloch_y=0, Qabs=Qabs)
    S3 = np.reshape(S3, (Nx, Ny))
    S3 = S3[big_design_mask]
    S3 = Gfddinv @ S3.flatten()
    
    #we need |S2> = |S3^*>
    S2 = np.conj(S3)
    
    #we need |E_bowtie> = G_f G_{0}^{-1}|E^i>
    #E_bowtie = G0inv @ (Ei.flatten())
    #E_bowtie = spla.spsolve(Gfinv, E_bowtie)
    test_Ebowtie = True
    if test_Ebowtie:
        Ebowtie = G0inv @ (Ei.flatten())
        Ebowtie = spla.spsolve(Gfinv, Ebowtie) #this gives the same as E_bowtie
    E_bowtie = TM.get_TM_linesource_field(wv, dl, Nx, Ny, cx, Npml, bloch_x=0.0, bloch_y=0.0, amp=1.0, Qabs=Qabs, chigrid=big_background_chi)
    E_bowtie = E_bowtie.flatten()
    #compare the two methods. If they agree, suggests that Gfinv is correct.
    if do_checks:
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        ax1.imshow(np.real(np.reshape(E_bowtie, (Nx,Ny))), cmap='RdBu')
        ax2.imshow(np.imag(np.reshape(E_bowtie, (Nx,Ny))), cmap='RdBu')
        plt.savefig('test_E_bowtie.png')
        
    if do_checks and test_Ebowtie:
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        ax1.imshow(np.real(np.reshape(Ebowtie, (Nx,Ny))), cmap='RdBu')
        ax2.imshow(np.imag(np.reshape(Ebowtie, (Nx,Ny))), cmap='RdBu')
        plt.savefig('test_Ebowtie.png')
        
    if do_checks and test_Ebowtie:
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        ax1.imshow(np.real(np.reshape(Ebowtie - E_bowtie, (Nx,Ny))), cmap='RdBu')
        ax2.imshow(np.imag(np.reshape(Ebowtie - E_bowtie, (Nx,Ny))), cmap='RdBu')
        plt.savefig('test_E_bowtieminusEbowtie.png')
        print('done vector calculations')
    
    if do_checks and test_Ebowtie:
        print('|E_bowtie - Ebowtie|/|E_bowtie| = ', np.linalg.norm(E_bowtie - Ebowtie)/np.linalg.norm(E_bowtie))
        
    E_bowtie = E_bowtie[big_design_mask.flatten()] #only need I_{d}|E_bowtie> (that is, E_bowtie over the design region)
    
    #######################set up optimization##################
    
    # region ############### get Lagrangian gradient functions #####################
    
    # notes on constraints:
    # we only need the Sym part because gcd will set some P ~ iI and we will extract the Asym
    # gradZTT and gradZTS_S are the same sign because ZTT contains the negative part of the constraint 
    def get_gradZTT(Plist): 
        return [zsp.Sym(Gfddinv.T.conj() @ ((1/np.conj(chid))*P) @ Gfddinv - P @ Gfddinv) for P in Plist]
    
    def get_gradZTS_S(Plist):
        return [Gfddinv.T.conj() @ P.T.conj() @ E_bowtie / 2 for P in Plist]
        
    def get_new_vects_heuristic(GT, fSlist, ZTT):
        T = Gfddinv @ GT
        UdagT = T/chid - GT # It is GT, work it out
        violation = -UdagT.conj() * T + E_bowtie.conj() * T

        for i in range(len(fSlist)): #contribution to violation from fake Source terms
            ZTTinvfS = spla.spsolve(ZTT, fSlist[i])
            GinvZTTinvfS = Gfddinv @ ZTTinvfS
            violation -= (1.0/np.conj(chid)) * (np.conj(GinvZTTinvfS) * GinvZTTinvfS) - np.conj(ZTTinvfS) * GinvZTTinvfS

        Laggradfac_phase = -violation / np.abs(violation)

        eigw, eigv = spla.eigs(ZTT, k=1)
        eigw = eigw[0]
        eigv = eigv[:,0]
        Ginveigv = Gfddinv @ eigv
        minfac = np.conj(Ginveigv) * Ginveigv / np.conj(chid) - np.conj(eigv) * Ginveigv
        mineig_phase = minfac / np.abs(minfac)

        new_vec = np.conj(Laggradfac_phase + mineig_phase)
        new_vec *= np.abs(np.real(-new_vec * violation))

        return new_vec
    
    # if do_checks:
    #     test_gradZTT = get_gradZTT([sp.eye(N, dtype=complex)])
    #     test_gradZTS_S = get_gradZTS_S([sp.eye(N, dtype=complex)])
    #     densities = []
    #     for i in range(len(test_gradZTT)):
    #         densities.append(zsp.density(test_gradZTT[i]))
    #     print(np.round(np.array(densities), 4)) #should be small so that sparse formulation makes sense

    # endregion

    O_quad = sp.csc_array(np.zeros((N, N), dtype=complex)) #extinction, no quadratic term
    O_lin_S = (np.conj(omega)/2/Z) * S2 * (-1/2j) * dl**2 * (-1) #-1 because optimization does maximization and we want min extinction
    

	############### Define optimization functions #####################
    print("Running optimization:")
#     opt_func = lambda Lags_include, dgfunc, validityfunc, mineigfunc, fSlist : full_Newton(Lags_include, dgfunc, validityfunc, mineigfunc, opttol,
#                                                                                             False, fakeSratio, iter_period=iter_period, verbose=True)
    
    opt_func = lambda Lags_include, dgfunc, validityfunc, mineigfunc, fSlist : Newton(Lags_include, dgfunc, validityfunc, mineigfunc, fSlist, opttol,
                                                                                            False, fakeSratio=fakeSratio, iter_period=iter_period)
    dualconst = 0 
    starting = [False, np.array([1, 0], dtype=float)]
    saveint = 1

    Plist, optLags, include, optgrad, optdual, gradZTT, gradZTS_S, ZTTchofac = gcd_sparse_all(N, get_gradZTT, get_gradZTS_S, dualconst, opt_func, starting, O_quad, O_lin_S, 
                   saveFunc, Pnum, gcd_maxiter, saveint, get_new_vects_heuristic)
    
    return optdual, background_term
