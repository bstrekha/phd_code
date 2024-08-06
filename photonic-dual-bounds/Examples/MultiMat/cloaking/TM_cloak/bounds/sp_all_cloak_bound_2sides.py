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

def sp_get_bandwidth_cloak_bound_2sides(chif, chid, wvlgthList, nmat, cons, des_region, des_params, obj, design_x, design_y, gpr, 
                                       NProjx, NProjy, pml_sep, pml_thick, Qabs=np.inf, opttol=1e-4, fakeSratio=1e-3, 
                                       iter_period=20, opttype='bfgs', init_lags=None, TESTS={'just_mask': False}, do_checks=False, init_global_lags=None, normalize_result=True, angle_y=0.0):
    assert(des_region in ['rect', 'circle'])
    assert(obj in ['EXT'])
    assert(opttype in ['bfgs', 'newton'])
    assert(len(wvlgthList) == 2)
    
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
    Ginvlist = []
    for wvlgth in wvlgthList:
        Ginvlist.append(Gfddinv)

    if NProjx > Mx:
        NProjx = Mx
    if NProjy > My:
        NProjy = My
        
    ################# calculate the fixed vectors (the 'source vectors' for optimization) #################
    Z = 1.0 # dimensionless units
    C_0 = 1.0
    eps0 = 1/Z/C_0
    
    big_background_chi = np.zeros((Nx, Ny), dtype=complex) #background chi over grid including pml region
    big_background_chi[Npml:-Npml, Npml:-Npml] = background_chi[:,:]
        
    big_design_mask = np.zeros((Nx, Ny), dtype=bool)
    big_design_mask[Npml:-Npml, Npml:-Npml] = design_mask[:,:] #design mask over grid including pml region

    
    G0inv = TM.get_TM_MaxwellOp(wv, dl, Nx, Ny, Npml, bloch_x=0, bloch_y=0, Qabs=Qabs) / omega**2
    Gfinv = G0inv + (TM.get_diagM_from_chigrid(omega, big_background_chi) / omega**2)
    
    #calculate x direction plane wave
    #generate an initial plane wave.
    cx = Npml + Npmlsep//2 #position of current sheet
    Ei_cx = TM.get_TM_linesource_field(wv, dl, Nx, Ny, cx, Npml, bloch_x=0.0, bloch_y=0.0, amp=1.0, Qabs=Qabs, chigrid=None) #plane wave in vacuum
    if do_checks:
        #should look like planewave
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        ax1.imshow(np.real(Ei_cx), cmap='RdBu')
        ax2.imshow(np.imag(Ei_cx), cmap='RdBu')
        plt.savefig('test_TM_planewave_Ei_cx.png')
        
    #we need |S1> = G_{0}^{-1} G_{f} V_{f} |E^i>
    S1_cx = big_background_chi*Ei_cx
    S1_cx = S1_cx.flatten()
    S1_cx = spla.spsolve(Gfinv, S1_cx)
    S1_cx = G0inv @ S1_cx
    EiS1_cx = np.vdot(Ei_cx, S1_cx) * dl**2
    background_term_cx = np.imag(EiS1_cx*omega/2/Z)
    print('background_term_cx = ', background_term_cx)
    
    #we need |S3> = G_{f,dd}^{-1} G_{f} G_{0}^{-1} |E^{i*}>
    S3_cx = G0inv @ np.conj(Ei_cx.flatten())
    S3_cx = spla.spsolve(Gfinv, S3_cx)
    #Gddinv, _ = TM.get_Gddinv(wv, dl, Nx, Ny, Npml, design_mask, bloch_x=0, bloch_y=0, Qabs=Qabs)
    S3_cx = np.reshape(S3_cx, (Nx, Ny))
    S3_cx = S3_cx[big_design_mask]
    S3_cx = Gfddinv @ S3_cx.flatten()
    
    #we need |S2> = |S3^*>
    S2_cx = np.conj(S3_cx)
    
    #we need |E_bowtie> = G_f G_{0}^{-1}|E^i>
    #E_bowtie = G0inv @ (Ei.flatten())
    #E_bowtie = spla.spsolve(Gfinv, E_bowtie)
    test_Ebowtie = False
    if test_Ebowtie:
        Ebowtie_cx = G0inv @ (Ei_cx.flatten())
        Ebowtie_cx = spla.spsolve(Gfinv, Ebowtie_cx) #this gives the same as E_bowtie
    E_bowtie_cx = TM.get_TM_linesource_field(wv, dl, Nx, Ny, cx, Npml, bloch_x=0.0, bloch_y=0.0, amp=1.0, Qabs=Qabs, chigrid=big_background_chi)
    E_bowtie_cx = E_bowtie_cx.flatten()
    #compare the two methods. If they agree, suggests that Gfinv is correct.
    if do_checks:
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        ax1.imshow(np.real(np.reshape(E_bowtie_cx, (Nx,Ny))), cmap='RdBu')
        ax2.imshow(np.imag(np.reshape(E_bowtie_cx, (Nx,Ny))), cmap='RdBu')
        plt.savefig('test_E_bowtie_cx.png')
        
    if do_checks and test_Ebowtie:
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        ax1.imshow(np.real(np.reshape(Ebowtie_cx, (Nx,Ny))), cmap='RdBu')
        ax2.imshow(np.imag(np.reshape(Ebowtie_cx, (Nx,Ny))), cmap='RdBu')
        plt.savefig('test_Ebowtie_cx.png')
        
    if do_checks and test_Ebowtie:
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        ax1.imshow(np.real(np.reshape(Ebowtie_cx - E_bowtie_cx, (Nx,Ny))), cmap='RdBu')
        ax2.imshow(np.imag(np.reshape(Ebowtie_cx - E_bowtie_cx, (Nx,Ny))), cmap='RdBu')
        plt.savefig('test_E_bowtieminusEbowtie_cx.png')
        print('done vector calculations')
    
    if do_checks and test_Ebowtie:
        print('|E_bowtie_cx - Ebowtie_cx|/|E_bowtie_cx| = ', np.linalg.norm(E_bowtie_cx - Ebowtie_cx)/np.linalg.norm(E_bowtie_cx))
        
    E_bowtie_cx = E_bowtie_cx[big_design_mask.flatten()] #only need I_{d}|E_bowtie> (that is, E_bowtie over the design region)
    
    #calculate y direction plane wave
    #generate an initial plane wave.
    cy = Npml + Npmlsep//2 #position of current sheet
    Ei_cy = TM.get_TM_linesource_field_cy(wv, dl, Nx, Ny, cy, Npml, bloch_x=0.0, bloch_y=0.0, amp=1.0, Qabs=Qabs, chigrid=None, angle=angle_y) #plane wave in vacuum

    if do_checks:
        #should look like planewave
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        ax1.imshow(np.real(Ei_cy), cmap='RdBu')
        ax2.imshow(np.imag(Ei_cy), cmap='RdBu')
        plt.savefig('test_TM_planewave_Ei_cy.png')
    #we need |S1> = G_{0}^{-1} G_{f} V_{f} |E^i>
    S1_cy = big_background_chi*Ei_cy
    S1_cy = S1_cy.flatten()
    S1_cy = spla.spsolve(Gfinv, S1_cy)
    S1_cy = G0inv @ S1_cy
    EiS1_cy = np.vdot(Ei_cy, S1_cy) * dl**2
    background_term_cy = np.imag(EiS1_cy*omega/2/Z)
    print('background_term_cy = ', background_term_cy)
    
    background_term = background_term_cx + background_term_cy
    print('background_term = ', background_term)
    
    #we need |S3> = G_{f,dd}^{-1} G_{f} G_{0}^{-1} |E^{i*}>
    S3_cy = G0inv @ np.conj(Ei_cy.flatten())
    S3_cy = spla.spsolve(Gfinv, S3_cy)
    #Gddinv, _ = TM.get_Gddinv(wv, dl, Nx, Ny, Npml, design_mask, bloch_x=0, bloch_y=0, Qabs=Qabs)
    S3_cy = np.reshape(S3_cy, (Nx, Ny))
    S3_cy = S3_cy[big_design_mask]
    S3_cy = Gfddinv @ S3_cy.flatten()
    
    #we need |S2> = |S3^*>
    S2_cy = np.conj(S3_cy)
    
    #we need |E_bowtie> = G_f G_{0}^{-1}|E^i>
    #E_bowtie = G0inv @ (Ei.flatten())
    #E_bowtie = spla.spsolve(Gfinv, E_bowtie)
    #test_Ebowtie = True
    if test_Ebowtie:
        Ebowtie_cy = G0inv @ (Ei_cy.flatten())
        Ebowtie_cy = spla.spsolve(Gfinv, Ebowtie_cy) #this gives the same as E_bowtie
    E_bowtie_cy = TM.get_TM_linesource_field_cy(wv, dl, Nx, Ny, cy, Npml, bloch_x=0.0, bloch_y=0.0, amp=1.0, Qabs=Qabs, chigrid=big_background_chi, angle=angle_y)
    E_bowtie_cy = E_bowtie_cy.flatten()
    #compare the two methods. If they agree, suggests that Gfinv is correct.
    if do_checks:
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        ax1.imshow(np.real(np.reshape(E_bowtie_cy, (Nx,Ny))), cmap='RdBu')
        ax2.imshow(np.imag(np.reshape(E_bowtie_cy, (Nx,Ny))), cmap='RdBu')
        plt.savefig('test_E_bowtie_cy.png')
        
    if do_checks and test_Ebowtie:
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        ax1.imshow(np.real(np.reshape(Ebowtie_cy, (Nx,Ny))), cmap='RdBu')
        ax2.imshow(np.imag(np.reshape(Ebowtie_cy, (Nx,Ny))), cmap='RdBu')
        plt.savefig('test_Ebowtie_cy.png')
        
    if do_checks and test_Ebowtie:
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        ax1.imshow(np.real(np.reshape(Ebowtie_cy - E_bowtie_cy, (Nx,Ny))), cmap='RdBu')
        ax2.imshow(np.imag(np.reshape(Ebowtie_cy - E_bowtie_cy, (Nx,Ny))), cmap='RdBu')
        plt.savefig('test_E_bowtieminusEbowtie_cy.png')
        print('done vector calculations')
    
    if do_checks and test_Ebowtie:
        print('|E_bowtie_cy - Ebowtie_cy|/|E_bowtie_cy| = ', np.linalg.norm(E_bowtie_cy - Ebowtie_cy)/np.linalg.norm(E_bowtie_cy))
        
    E_bowtie_cy = E_bowtie_cy[big_design_mask.flatten()] #only need I_{d}|E_bowtie> (that is, E_bowtie over the design region)
        
    source_list = []
    source_list.append(E_bowtie_cx)
    source_list.append(E_bowtie_cy)
    
    N = len(E_bowtie_cx)
    
    S2_list = []
    S2_list.append(S2_cx)
    S2_list.append(S2_cy)
                    
    expandedS2_forOlin = np.zeros((nsource, nmat, N), dtype=complex) #for objective
    expanded_E_bowtie = np.zeros((nsource, N), dtype=complex) #for constraints
    for s in range(nsource):
        for m in range(nmat):
            expandedS2_forOlin[s, m, :] = S2_list[s]
        expanded_E_bowtie[s, :] = source_list[s]
                                 
    expandedS2_forOlin = np.reshape(expandedS2_forOlin, nsource*nmat*N, order='C') #for objective
    expanded_E_bowtie = np.reshape(expanded_E_bowtie, nsource*N, order='C') #for constraints
    
    #######################set up optimization##################
    
    #Plist = [np.eye(N, dtype=complex)] #used when only had global constraints
    
    Plist = generateP(NProjx*NProjy, N)

#     cons = [1, 1]
#     npixels = len(Plist)
#     nm = nmat #1 for us
#     total_numlag = int(npixels*(2*(nsource)**2 + 2*(nm*(nm-1)/2)*(nsource*(nsource+1)/2)))
#     include = zmultimat.include_helper(npixels, nsource, nm, powerCons=cons[0], ortho=cons[1])
#     print(f'total numlag: {total_numlag}')
    
    # endregion

    # region ############### get Lagrangian gradients #####################
    print('nmat: ', nmat)
    print('nsource: ', nsource)
    chidlist = np.reshape([chid, chid], (nsource, nmat)) #(2,1) shape
    gradZTT = zmultimat.get_gradZTT(Plist, nmat, nsource, chidlist, Ginvlist)
    include = np.ones(len(gradZTT), dtype=bool)
    print('total numlag: ', len(gradZTT))
    for i in range(len(gradZTT)):
        gradZTT[i] = sp.csc_array(gradZTT[i]) # We will no longer be using the coordinates, csc will be more efficient for cholesky
        
#     gradZTS_S = zmultimat.get_gradZTS_S(E_bowtie, Plist, Ginvlist, nm, nsource)
#     del Plist #only need Plist to get gradZTT and gradZTS_S to do the optimization.
#     S_gradZSS_S = np.zeros(len(gradZTT))
    
    gradZTS_S = zmultimat.get_gradZTS_S(expanded_E_bowtie, Plist, Ginvlist, nmat, nsource) 
    S_gradZSS_S = np.zeros(len(include))
    
    del Plist #only need Plist to get gradZTT and gradZTS_S to do the optimization.

    densities = []
    for i in range(len(gradZTT)):
        densities.append(zsp.density(gradZTT[i]))
    print(np.round(np.array(densities), 4)) #should be small so that sparse formulation makes sense
    # endregion

    # region ############### get objective operators #####################
    O_quad = sp.csc_array(np.zeros((nsource*N, nsource*N), dtype=complex)) #extinction, no quadratic term    
    O_lin_S = (np.conj(omega)/2/Z) * expandedS2_forOlin * (-1/2j) * dl**2 * (-1) #-1 because optimization does maximization and we want min extinction
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