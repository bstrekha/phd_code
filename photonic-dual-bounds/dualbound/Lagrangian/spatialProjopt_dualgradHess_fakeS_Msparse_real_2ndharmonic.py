#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 12:40:30 2022

@author: jewelmohajan
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sksparse.cholmod as chol
from .spatialProjopt_vecs_Msparse_real import get_Tvec, get_ZTTcho_Tvec, get_ZTTcho_Tvec_gradTvec_real, get_ZTTcho_Tvec_gradTvec


###METHOD USING SPARSE CHOLESKY DECOMPOSITION FOR LINEAR SOLVE###
def get_2ndharmonic_spatialProj_dualgrad_fakeS_Msparse(C_T2, C_S2, Lags, grad, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=0.0, include=None, chofac=None, mineigtol=None):
    if include is None:
        include = [True]*len(Lags)
        
    ZTS_S = O_lin.copy()
    ZTT = O_quad.copy()
    for i in range(len(Lags)):
        ZTS_S += Lags[i] * gradZTS_S[i]
        ZTT += Lags[i] * gradZTT[i]
    ZTT=ZTT.tocsc()
    SymZTT = (ZTT+ZTT.T)/2

    ZTTcho, T = get_ZTTcho_Tvec(2*SymZTT, ZTS_S, chofac=chofac)
    if mineigtol is None:
        ZTTfScho = ZTTcho
    else:
        ZTTfScho = chol.cholesky(2*SymZTT - mineigtol*sp.eye(ZTT.shape[0], format='csc'))
        """
        if chofac is None:
            print('chofac is None')
            ZTTfScho = chol.cholesky(ZTT.T - mineigtol*sp.eye(ZTT.shape[0], format='csc'))
        else:
            ZTTfScho = chofac.cholesky(2*SymZTT - mineigtol*sp.eye(ZTT.shape[0], format='csc'))
        """
        
    dualval = dualconst
    dualval += np.real(np.vdot(T, ZTT @ T)) + Lags[-2]*C_T2 + Lags[-1]*C_S2 #+ Lags[-2]*np.real(np.vdot(T[np.int(len(O_lin)/3):np.int(2*len(O_lin)/3)], Lags[-1]*T[np.int(len(O_lin)/3):np.int(2*len(O_lin)/3)])) + np.real(np.vdot(T[np.int(2*len(O_lin)/3):np.int(len(O_lin))], T[np.int(2*len(O_lin)/3):np.int(len(O_lin))]))
    
    if len(grad)>0:
        grad[:] = 0
        for i in range(len(Lags)):
            if include[i]:
                #print("T.shape",T.shape)
                grad[i] += -np.real(np.vdot(T, gradZTT[i].dot(T))) + np.real(np.vdot(T, gradZTS_S[i]))
                
                if i == (len(Lags)-2):
                    grad[i] += C_T2
                if i == (len(Lags)-1):
                    grad[i] += C_S2
                
        for _, fS in enumerate(fSlist):
            ZTTfSinv_fS = ZTTfScho.solve_A(fS)
            dualval += np.real(np.vdot(fS, ZTTfSinv_fS))
            
            for i in range(len(Lags)):
                if include[i]:
                    grad[i] += -np.real(np.vdot(ZTTfSinv_fS, (gradZTT[i]+gradZTT[i].T).dot(ZTTfSinv_fS)))
                    #grad[i] += -np.real(np.vdot(fS, gradZTT[i].T.dot(fS)))

    else:
        for _, fS in enumerate(fSlist):
            ZTTfSinv_fS = ZTTfScho.solve_A(fS)
            dualval += np.real(np.vdot(fS, ZTTfSinv_fS))

    return dualval

def get_2ndharmonic_spatialProj_dualgradHess_fakeS_Msparse(C_T2, C_S2, Lags, grad, Hess, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=0.0, include=None, chofac=None, mineigtol=None):
    if include is None:
        include = [True]*len(Lags)
        
    ZTS_S = O_lin.copy()
    ZTT = O_quad.copy()
    for i in range(len(Lags)):
        ZTS_S += Lags[i] * gradZTS_S[i]
        ZTT += Lags[i] * gradZTT[i]
    ZTT=ZTT.tocsc()
    SymZTT = (ZTT+ZTT.T)/2

    ZTTcho, T, gradT = get_ZTTcho_Tvec_gradTvec_real(2*SymZTT, gradZTT, ZTS_S, gradZTS_S, chofac=chofac)
    if mineigtol is None:
        ZTTfScho = ZTTcho
    else:
        if chofac is None:
            print('chofac is None')
            ZTTfScho = chol.cholesky(ZTT - mineigtol*sp.eye(ZTT.shape[0], format='csc'))
        else:
            ZTTfScho = chofac.cholesky(ZTT - mineigtol*sp.eye(ZTT.shape[0], format='csc'))

    dualval = dualconst + np.real(np.vdot(T, ZTT @ T)) + Lags[-2]*C_T2 + Lags[-1]*C_S2

    grad[:] = 0
    Hess[:,:] = 0
    
    for i in range(len(Lags)):
        if include[i]:
            grad[i] += -np.real(np.vdot(T, gradZTT[i] @ T)) + np.real(np.vdot(T, gradZTS_S[i]))
            
            if i == (len(Lags)-2):
                grad[i] += C_T2
            if i == (len(Lags)-1):
                grad[i] += C_S2
            
    for i in range(len(Lags)):
        if not include[i]:
            continue
        for j in range(i,len(Lags)):
            if not include[j]:
                continue
            Hess[i,j] += np.real(np.vdot(gradT[i],-(gradZTT[j] +gradZTT[j].T) @ T + gradZTS_S[j]))
            if i!=j:
                Hess[j,i] = Hess[i,j]

    for _, fS in enumerate(fSlist):
        ZTTfSinv_fS = ZTTfScho.solve_A(fS)
        dualval += np.real(np.vdot(fS, ZTTfSinv_fS))
        ZTTfSinv_gradZTT_ZTTfSinv_fS = []
        for i in range(len(Lags)):
            if include[i]:
                gradZTT_ZTTfSinv_fS = (gradZTT[i] + gradZTT[i].T)  @ ZTTfSinv_fS
                grad[i] += -np.real(np.vdot(ZTTfSinv_fS, gradZTT_ZTTfSinv_fS))
                ZTTfSinv_gradZTT_ZTTfSinv_fS.append(ZTTfScho.solve_A(gradZTT_ZTTfSinv_fS))
            else:
                ZTTfSinv_gradZTT_ZTTfSinv_fS.append(None)

        for i in range(len(Lags)):
            if not include[i]:
                continue
            for j in range(i,len(Lags)):
                if not include[j]:
                    continue
                Hess[i,j] += 2*np.real(np.vdot(ZTTfSinv_fS, (gradZTT[i] + gradZTT[i].T) @ ZTTfSinv_gradZTT_ZTTfSinv_fS[j]))
                if i!=j:
                    Hess[j,i] = Hess[i,j]
                
    return dualval

def get_spatialProj_dualgrad_fakeS_Msparse(Lags, grad, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=0.0, include=None, chofac=None, mineigtol=None):
    if include is None:
        include = [True]*len(Lags)
        
    ZTS_S = O_lin.copy()
    ZTT = O_quad.copy()
    for i in range(len(Lags)):
        ZTS_S += Lags[i] * gradZTS_S[i]
        ZTT += Lags[i] * gradZTT[i]
    ZTT=ZTT.tocsc()
    SymZTT = (ZTT+ZTT.T)/2

    ZTTcho, T = get_ZTTcho_Tvec(2*SymZTT, ZTS_S, chofac=chofac)
    if mineigtol is None:
        ZTTfScho = ZTTcho
    else:
        ZTTfScho = chol.cholesky(SymZTT - mineigtol*sp.eye(ZTT.shape[0], format='csc'))
        """
        if chofac is None:
            print('chofac is None')
            ZTTfScho = chol.cholesky(ZTT.T - mineigtol*sp.eye(ZTT.shape[0], format='csc'))
        else:
            ZTTfScho = chofac.cholesky(2*SymZTT - mineigtol*sp.eye(ZTT.shape[0], format='csc'))
        """
        
    dualval = dualconst
    dualval += np.real(np.vdot(T, ZTT @ T)) 
    
    if len(grad)>0:
        grad[:] = 0
        for i in range(len(Lags)):
            if include[i]:
                #print("T.shape",T.shape)
                grad[i] += -np.real(np.vdot(T, gradZTT[i].dot(T))) + np.real(np.vdot(T, gradZTS_S[i]))


        for _, fS in enumerate(fSlist):
            
            ZTTfSinv_fS = ZTTfScho.solve_A(fS)
            dualval += np.real(np.vdot(fS, ZTTfSinv_fS))

            for i in range(len(Lags)):
                if include[i]:
                    grad[i] += -np.real(np.vdot(ZTTfSinv_fS, (gradZTT[i] + gradZTT[i].T).dot(ZTTfSinv_fS)))
                    #grad[i] += -np.real(np.vdot(fS, gradZTT[i].T.dot(fS)))

    else:
        for _, fS in enumerate(fSlist):
            
            ZTTfSinv_fS = ZTTfScho.solve_A(fS)
            dualval += np.real(np.vdot(fS, ZTTfSinv_fS))

    return dualval


def get_spatialProj_dualgradHess_fakeS_Msparse(Lags, grad, Hess, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=0.0, include=None, chofac=None, mineigtol=None):
    if include is None:
        include = [True]*len(Lags)
        
    ZTS_S = O_lin.copy()
    ZTT = O_quad.copy()
    for i in range(len(Lags)):
        ZTS_S += Lags[i] * gradZTS_S[i]
        ZTT += Lags[i] * gradZTT[i]
    SymZTT = (ZTT+ZTT.T)/2

    ZTTcho, T, gradT = get_ZTTcho_Tvec_gradTvec_real(2*SymZTT, gradZTT, ZTS_S, gradZTS_S, chofac=chofac)
    if mineigtol is None:
        ZTTfScho = ZTTcho
    else:
        if chofac is None:
            print('chofac is None')
            ZTTfScho = chol.cholesky(ZTT - mineigtol*sp.eye(ZTT.shape[0], format='csc'))
        else:
            ZTTfScho = chofac.cholesky(ZTT - mineigtol*sp.eye(ZTT.shape[0], format='csc'))

    dualval = dualconst + np.real(np.vdot(T, ZTT @ T))

    grad[:] = 0
    Hess[:,:] = 0
    
    for i in range(len(Lags)):
        if include[i]:
            grad[i] += -np.real(np.vdot(T, gradZTT[i] @ T)) + np.real(np.vdot(T, gradZTS_S[i]))
            
    for i in range(len(Lags)):
        if not include[i]:
            continue
        for j in range(i,len(Lags)):
            if not include[j]:
                continue
            Hess[i,j] += np.real(np.vdot(gradT[i],-(gradZTT[j] +gradZTT[j].T) @ T + gradZTS_S[j]))
            if i!=j:
                Hess[j,i] = Hess[i,j]

    for _, fS in enumerate(fSlist):
        ZTTfSinv_fS = ZTTfScho.solve_A(fS)
        dualval += np.real(np.vdot(fS, ZTTfSinv_fS))
        ZTTfSinv_gradZTT_ZTTfSinv_fS = []
        for i in range(len(Lags)):
            if include[i]:
                gradZTT_ZTTfSinv_fS = (gradZTT[i] + gradZTT[i].T)  @ ZTTfSinv_fS
                grad[i] += -np.real(np.vdot(ZTTfSinv_fS, gradZTT_ZTTfSinv_fS))
                ZTTfSinv_gradZTT_ZTTfSinv_fS.append(ZTTfScho.solve_A(gradZTT_ZTTfSinv_fS))
            else:
                ZTTfSinv_gradZTT_ZTTfSinv_fS.append(None)

        for i in range(len(Lags)):
            if not include[i]:
                continue
            for j in range(i,len(Lags)):
                if not include[j]:
                    continue
                Hess[i,j] += 2*np.real(np.vdot(ZTTfSinv_fS, (gradZTT[i] + gradZTT[i].T) @ ZTTfSinv_gradZTT_ZTTfSinv_fS[j]))
                if i!=j:
                    Hess[j,i] = Hess[i,j]
                
    return dualval




def get_2ndharmonic_inc_spatialProj_dualgrad_fakeS_Msparse(C_T2, C_S2, incLags, incgrad, include, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=0.0, get_grad=True, chofac=None, mineigtol=None):

    Lagnum = len(include)
    Lags = np.zeros(Lagnum)
    Lags[include] = incLags[:]

    if get_grad:
        grad = np.zeros(Lagnum)
        dualval = get_2ndharmonic_spatialProj_dualgrad_fakeS_Msparse(C_T2, C_S2, Lags, grad, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include, chofac=chofac, mineigtol=mineigtol)
        incgrad[:] = grad[include]
    else:
        dualval = get_2ndharmonic_spatialProj_dualgrad_fakeS_Msparse(C_T2, C_S2, Lags, np.array([]), O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include, chofac=chofac, mineigtol=mineigtol)

    return dualval

def get_inc_spatialProj_dualgrad_fakeS_Msparse(incLags, incgrad, include, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=0.0, get_grad=True, chofac=None, mineigtol=None):

    Lagnum = len(include)
    Lags = np.zeros(Lagnum)
    Lags[include] = incLags[:]

    if get_grad:
        grad = np.zeros(Lagnum)
        dualval = get_spatialProj_dualgrad_fakeS_Msparse(Lags, grad, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include, chofac=chofac, mineigtol=mineigtol)
        incgrad[:] = grad[include]
    else:
        dualval = get_spatialProj_dualgrad_fakeS_Msparse(Lags, np.array([]), O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include, chofac=chofac, mineigtol=mineigtol)

    return dualval


def get_inc_spatialProj_dualgradHess_fakeS_Msparse(incLags, incgrad, incHess, include, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=0.0, get_grad=True, get_Hess=True, chofac=None, mineigtol=None):
    Lagnum = len(include)
    Lags = np.zeros(Lagnum)
    Lags[include] = incLags[:]
    
    if get_Hess:
        grad = np.zeros(Lagnum)
        Hess = np.zeros((Lagnum,Lagnum))
        dualval = get_spatialProj_dualgradHess_fakeS_Msparse(Lags,grad,Hess, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include, mineigtol=mineigtol)
        incgrad[:] = grad[include] #[:] since we are modifying in place
        incHess[:,:] = Hess[np.ix_(include,include)]
    elif get_grad:
        grad = np.zeros(Lagnum)
        dualval = get_spatialProj_dualgrad_fakeS_Msparse(Lags,grad, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include, mineigtol=mineigtol)
        incgrad[:] = grad[include]
    else:
        dualval = get_spatialProj_dualgrad_fakeS_Msparse(Lags,np.array([]), O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include, mineigtol=mineigtol)
        
    return dualval
def get_2ndharmonic_inc_spatialProj_dualgradHess_fakeS_Msparse(C_T2, C_S2, incLags, incgrad, incHess, include, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=0.0, get_grad=True, get_Hess=True, chofac=None, mineigtol=None):
    Lagnum = len(include)
    Lags = np.zeros(Lagnum)
    Lags[include] = incLags[:]
    
    if get_Hess:
        grad = np.zeros(Lagnum)
        Hess = np.zeros((Lagnum,Lagnum))
        dualval = get_2ndharmonic_spatialProj_dualgradHess_fakeS_Msparse(C_T2, C_S2, Lags,grad,Hess, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include, mineigtol=mineigtol)
        incgrad[:] = grad[include] #[:] since we are modifying in place
        
        incHess[:,:] = Hess[np.ix_(include,include)]
    elif get_grad:
        grad = np.zeros(Lagnum)
        dualval = get_2ndharmonic_spatialProj_dualgrad_fakeS_Msparse(C_T2, C_S2, Lags,grad, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include, mineigtol=mineigtol)
        incgrad[:] = grad[include]
    else:
        dualval = get_2ndharmonic_spatialProj_dualgrad_fakeS_Msparse(C_T2, C_S2, Lags,np.array([]), O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include, mineigtol=mineigtol)
        
    return dualval