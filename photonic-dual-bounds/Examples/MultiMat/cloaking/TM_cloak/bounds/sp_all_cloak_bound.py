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
from dualbound.Optimization.fakeSource_with_restart_singlematrix import fakeS_with_restart_singlematrix as Newton
from dualbound.Optimization.BFGS_fakeSource_with_restart import BFGS_fakeS_with_restart as BFGS
from dualbound.Lagrangian import dualgrad_sparse_all as dualgrad
import dualbound.Lagrangian.multimat.zops_multimat_sp as zmultimat

def sp_get_bandwidth_cloak_bound(chif, chid, wvlgthList, nmat, cons, des_region, des_params, obj, design_x, design_y, gpr, 
                                       NProjx, NProjy, pml_sep, pml_thick, Qabs=np.inf, opttol=1e-4, fakeSratio=1e-3, 
                                       iter_period=20, opttype='bfgs', init_lags=None, TESTS={'just_mask': False}, do_checks=False, init_global_lags=None, normalize_result=True):
    assert(des_region in ['rect', 'circle'])
    assert(obj in ['EXT'])
    assert(opttype in ['bfgs', 'newton'])
    assert(len(wvlgthList) == 1)
    
    dl = 1.0/gpr
    Mx = int(np.round(design_x/dl))
    My = int(np.round(design_y/dl))
    Npml = int(np.round(pml_thick/dl))
    Npmlsep = int(np.round(pml_sep/dl))
    nonpmlNx = Mx + 2*Npmlsep
    nonpmlNy = My + 2*Npmlsep
    Nx = nonpmlNx + 2*Npml
    Ny = nonpmlNy + 2*Npml
    nsource = len(wvlgthList)
    wv = wvlgthList[0]
    omega = 2*np.pi/wv*(1 + 1j/2/Qabs)
    omega0 = 2*np.pi/wv
    
    background_mask = np.zeros((nonpmlNx,nonpmlNy), dtype=bool) #the fixed object, which is considered to be part of background
    if des_region == 'rect':
        #written this way for symmetry
        xwidth = int(des_params[0]/dl)
        ywidth = int(des_params[1]/dl)
        assert(xwidth <= Mx)
        assert(ywidth <= My)
        background_mask[Npmlsep + Mx//2 - xwidth//2:Npmlsep + Mx//2 + xwidth//2, Npmlsep + My//2 - ywidth//2:Npmlsep + My//2 + ywidth//2] = True
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
        xwidth = int(des_params[0]/dl)
        ywidth = int(des_params[1]/dl)
        assert(xwidth <= Mx)
        assert(ywidth <= My)
        background_mask[Npmlsep + Mx//2 - xwidth//2:Npmlsep + Mx//2 + xwidth//2, Npmlsep + My//2 - ywidth//2:Npmlsep + My//2 + ywidth//2] = True
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
    if do_checks:
        print('N: ', N)
        print('sum(design_mask): ', np.sum(design_mask), flush=True)
    Gfddinv = sp.csr_array(Gfddinv)
    Ginvlist = [Gfddinv]

    if NProjx > Mx:
        NProjx = Mx
    if NProjy > My:
        NProjy = My
        
    ################# calculate the fixed vectors #################
    Z = 1.0 # dimensionless units
    C_0 = 1.0
    eps0 = 1/Z/C_0
    
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
    vacPrad = 1.0
    
    #######################set up optimization##################
    
    #Plist = [np.eye(N, dtype=complex)] #used when only had global constraints
    
    Plist = generateP(NProjx*NProjy, N)

    npixels = len(Plist)
    nm = nmat
    total_numlag = int(npixels*(2*(nsource)**2 + 2*(nm*(nm-1)/2)*(nsource*(nsource+1)/2)))

    print(f'total numlag: {total_numlag}')
    # endregion

    # region ############### get Lagrangian gradients #####################
    print('nm: ', nm)
    print('nsource: ', nsource)
    gradZTT = zmultimat.get_gradZTT(Plist, nmat, nsource, np.array([[chid]]), Ginvlist)
    for i in range(len(gradZTT)):
        gradZTT[i] = sp.csc_array(gradZTT[i]) # We will no longer be using the coordinates, csc will be more efficient for cholesky
        
    gradZTS_S = zmultimat.get_gradZTS_S(E_bowtie, Plist, Ginvlist, nm, nsource)
    del Plist #only need Plist to get gradZTT and gradZTS_S to do the optimization.
    S_gradZSS_S = np.zeros(len(gradZTT))
    
    include = np.ones(len(gradZTT), dtype=bool)

    densities = []
    for i in range(len(gradZTT)):
        densities.append(zsp.density(gradZTT[i]))
    print(np.round(np.array(densities), 4)) #should be small so that sparse formulation makes sense
    # endregion

    # region ############### get objective operators #####################
    O_quad = sp.csc_array(np.zeros((N, N), dtype=complex)) #extinction, no quadratic term
    #O_lin_S = (k/2/Z) * S2 * dl**2 
    O_lin_S = (np.conj(omega)/2/Z) * S2 * (-1/2j) * dl**2 * (-1) #-1 because optimization does maximization and we want min extinction
    ZTTchofac = zsp.Cholesky_analyze_ZTT(O_quad, gradZTT) # This is the sparse chofac of ZTT "symbolically", it becomes easier to then extract the actual factorization
    # endregion
    
    # region #### initialize lags ####
    if init_lags is None:
        Lags = np.random.rand(len(include))*0
        iter, start, numrestart, limit = 0, 1, 1, 1e3
        tPowerConst, interval = nsource*nsource*2, 2*nsource+2
        Lags[1:tPowerConst:interval] = start # Counts all the lags that have asym(G)
        Lags_include = Lags[include]
    else:
        Lags = np.zeros(len(include))
        Lags[0:nsource*nsource*2] = init_lags[0:nsource*nsource*2]
        Lags[npixels*nsource*nsource*2: npixels*nsource*nsource*2 + len(init_lags[nsource*nsource*2:])] = init_lags[nsource*nsource*2:]
        Lags_include = Lags[include]
        
    if init_global_lags is not None:
        Lags[0:len(init_global_lags)] = init_global_lags

    print(f"Initial Lags: {Lags_include}")

    validityfunc = lambda dof: zsp.check_spatialProj_incLags_validity(dof, include, O_quad, gradZTT, ZTTchofac) # it will use the previously calculated chofac to get the decomposition really quickly (and check if ZTT > 0)
    assert(validityfunc(Lags_include) >= 0)
    print(f"Found positive definite point: {Lags_include}")
    # endregion
    
	############### Define optimization functions #####################
    print("Running optimization:")
    mineigfunc = lambda dof, v0=None: zsp.get_inc_ZTT_mineig(dof, include, O_quad, gradZTT, eigvals_only=False, v0=v0) # v0 is an initial guess at the lowest eigenvector. By default, it is random but if you have some information on it you could pass it along. 
    
    # First do some Newton steps. If you're doing global optimization, low opttol. Otherwise just skip this part 
    if (NProjx == 1 and NProjy == 1) and (opttype == 'newton'):
        opttolnewton = 1e-3
        dgfunc = lambda dof, grad, dofHess, fSl, get_grad=True, get_Hess=True: (
            dualgrad.get_dual_and_derivatives(dof, grad, dofHess, include, O_lin_S, O_quad, gradZTS_S, gradZTT, 
                                S_gradZSS_S, fSl, ZTTchofac, dualconst=0, get_grad=get_grad, get_Hess=get_Hess)) 
        optincLags, optincgrad, dualval, objval, conv = Newton(Lags_include, dgfunc, validityfunc, mineigfunc, opttol=opttolnewton, 
                                                                fakeSratio=fakeSratio, iter_period=iter_period, gradConverge=False)

        Lags_include = optincLags #[include]

    # Then optimize with BFGS:
    print("Entering BFGS Optimization.")
    dgfunc = lambda dof, grad, fSl, get_grad=True: (
            dualgrad.get_dual_and_derivatives(dof, grad, np.array([]), include, O_lin_S, O_quad, gradZTS_S, gradZTT, 
                                              S_gradZSS_S, fSl, ZTTchofac, dualconst=0, get_grad=get_grad, get_Hess=False))
        
    optincLags, optincgrad, dualval, objval, conv = BFGS(Lags_include, dgfunc, validityfunc, mineigfunc, opttol=opttol, 
                                                         fakeSratio=fakeSratio, iter_period=iter_period, gradConverge=False)

    optgrad = np.zeros(len(include), dtype=np.double)
    optgrad[include] = optincgrad[:]
    print('the remaining constraint violations')
    print(optgrad)
    print('final lags')
    print(optincLags)

    result = (background_term - dualval)
    if normalize_result:
        result = result/background_term
    print('background_term:', background_term)
    print('bound Tbowtie part:', -dualval)
    print('min_enh bound:', result)
    return np.real(result), optincLags, optincgrad, conv, NProjx, NProjy

def generateP(nproj, size):
    #creates projection matrices in sparse form.
    #converted to dense form one at a in get_gradZTT and get_gradZTS_S to save memory
    projlist = []
    if nproj >= size:
        #projlist.append(np.eye(size))
        projlist.append(sp.csr_array(np.eye(size)))
        for i in range(1, size):
            A = np.zeros((size, size))
            A[i, i] = 1
            projlist.append(sp.csr_array(A))
            #projlist.append(A)
    else:
        interval = size // nproj 
        #projlist.append(np.eye(size))
        projlist.append(sp.csr_array(np.eye(size)))
        for i in range(1, nproj):
            A = np.zeros((size, size))
            A[i*interval:(i+1)*interval, i*interval:(i+1)*interval] = np.eye(interval)
            projlist.append(sp.csr_array(A))
            #projlist.append(A)
    return projlist