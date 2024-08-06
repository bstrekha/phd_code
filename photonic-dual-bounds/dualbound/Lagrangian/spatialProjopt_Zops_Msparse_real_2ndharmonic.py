#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 12:42:36 2022

This is part of the grad/Hess engine for spatial projection versions of the 
original global constraint <S|T>-<T|U|T>, formulated with sparse matrices based on the 
Maxwell operator. 
@author: jewelmohajan
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sksparse.cholmod as chol
from realmatrices import cmplxtoreal_matrix, cmplxtoreal_vector, conjugation_matrix, cmplx_multiplier_matrix

def get_real_2ndharminic_Msparse_gradZTT(C_J2T1, G1ddinv, G1invdagPdaglist, G2invdagPdaglist, UP11list, UP22list, UP12list, UP21list, Proj_mat_S2_list):
    n_basis = UP11list[0].shape[0]
    n_basis_real =  2 * n_basis
    n_pdof = 3 * n_basis_real #number of primal dofs, corresponding to size of ZTT / gradZTT matrices
    n_proj_vec = len(Proj_mat_S2_list)
    n_proj_mat = len(UP11list)
    n_cplx_projLags = 4 #2**2
    Lagnum = n_proj_mat * 2 * n_cplx_projLags # 2 for real and imag
    Lagnum += 2 # this 2 from T2 and S2 inequality constraint to ensure PDness of the global constraint
    Lagnum += 2*n_proj_vec
    gradZ = [None] * Lagnum
    
    for i in range(n_proj_vec):
        ind_vec_re = i*2
        ind_vec_im = i*2 + 1
        Proj = Proj_mat_S2_list[i]
        mul_mat=sp.lil_matrix((n_basis_real,n_basis_real))
        vec_constraint_real = sp.lil_matrix((n_pdof,n_pdof))
        vec_constraint_imag = sp.lil_matrix((n_pdof,n_pdof))
        mul_mat[0*n_basis:(0+1)*n_basis,0*n_basis:(0+1)*n_basis] = np.real(C_J2T1)*Proj
        mul_mat[0*n_basis:(0+1)*n_basis,1*n_basis:(1+1)*n_basis] = -np.imag(C_J2T1)*Proj
        mul_mat[1*n_basis:(1+1)*n_basis,0*n_basis:(0+1)*n_basis] = -np.imag(C_J2T1)*Proj
        mul_mat[1*n_basis:(1+1)*n_basis,1*n_basis:(1+1)*n_basis] = -np.real(C_J2T1)*Proj
        
        vec_constraint_real[0*n_basis_real:(0+1)*n_basis_real , 0*n_basis_real:(0+1)*n_basis_real] = (cmplxtoreal_matrix(G1ddinv)).T @ (mul_mat @ cmplxtoreal_matrix(G1ddinv))
        vec_constraint_imag[0*n_basis_real:(0+1)*n_basis_real , 0*n_basis_real:(0+1)*n_basis_real] = (cmplxtoreal_matrix(G1ddinv)).T @ (mul_mat @ cmplxtoreal_matrix(G1ddinv))
        
        gradZ[ind_vec_re] = vec_constraint_real.tocsc()
        gradZ[ind_vec_im] = vec_constraint_imag.tocsc()
    
    ind_off_vec=2*n_proj_vec
    for i in range(n_proj_mat):
        for j in range(4):
            ind_re=ind_off_vec + i*8 + j
            ind_im=ind_off_vec + i*8 + j+4
            gradZ_re = sp.lil_matrix((n_pdof,n_pdof))
            gradZ_im = sp.lil_matrix((n_pdof,n_pdof))
            if j==0:
                SymUP11 = (UP11list[i] + UP11list[i].conj().T) / 2
                AsymUP11 = (UP11list[i] - UP11list[i].conj().T) / 2j
                gradZ_re[0*n_basis_real:(0+1)*n_basis_real , 0*n_basis_real:(0+1)*n_basis_real] = cmplxtoreal_matrix(SymUP11)   
                gradZ_im[0*n_basis_real:(0+1)*n_basis_real , 0*n_basis_real:(0+1)*n_basis_real] = cmplxtoreal_matrix(AsymUP11)
            elif j==1:
                UP12 = UP12list[i]
                G1invdagPdag = G1invdagPdaglist[i]
                gradZ_re[0*n_basis_real:(0+1)*n_basis_real , 1*n_basis_real:(1+1)*n_basis_real] = cmplxtoreal_matrix(UP12)   
                gradZ_im[0*n_basis_real:(0+1)*n_basis_real , 1*n_basis_real:(1+1)*n_basis_real] = cmplx_multiplier_matrix(n_basis,0,-1)*cmplxtoreal_matrix(UP12)
                gradZ_re[0*n_basis_real:(0+1)*n_basis_real , 2*n_basis_real:(2+1)*n_basis_real] = -cmplxtoreal_matrix(G1invdagPdag)   
                gradZ_im[0*n_basis_real:(0+1)*n_basis_real , 2*n_basis_real:(2+1)*n_basis_real] = -cmplx_multiplier_matrix(n_basis,0,-1)*cmplxtoreal_matrix(G1invdagPdag)
            elif j==2:
                UP21 = UP21list[i]
                gradZ_re[1*n_basis_real:(1+1)*n_basis_real , 0*n_basis_real:(0+1)*n_basis_real] = cmplxtoreal_matrix(UP21)   
                gradZ_im[1*n_basis_real:(1+1)*n_basis_real , 0*n_basis_real:(0+1)*n_basis_real] = cmplx_multiplier_matrix(n_basis,0,-1)*cmplxtoreal_matrix(UP21)
            elif j==3:
                G2invdagPdag = G2invdagPdaglist[i]
                SymUP22 = (UP22list[i] + UP22list[i].conj().T) / 2
                AsymUP22 = (UP22list[i] - UP22list[i].conj().T) / 2j
                gradZ_re[1*n_basis_real:(1+1)*n_basis_real , 1*n_basis_real:(1+1)*n_basis_real] = cmplxtoreal_matrix(SymUP22)   
                gradZ_im[1*n_basis_real:(1+1)*n_basis_real , 1*n_basis_real:(1+1)*n_basis_real] = cmplxtoreal_matrix(AsymUP22)
                gradZ_re[1*n_basis_real:(1+1)*n_basis_real , 2*n_basis_real:(2+1)*n_basis_real] = -cmplxtoreal_matrix(G2invdagPdag)   
                gradZ_im[1*n_basis_real:(1+1)*n_basis_real , 2*n_basis_real:(2+1)*n_basis_real] = -cmplx_multiplier_matrix(n_basis,0,1)*cmplxtoreal_matrix(G2invdagPdag)

            gradZ[ind_re] = gradZ_re.tocsc()
            gradZ[ind_im] = gradZ_im.tocsc()
    ind_off = 2*n_proj_vec + 8*n_proj_mat
    
    gradZ_T2T2 = sp.lil_matrix((n_pdof,n_pdof))
    gradZ_T2T2[1*n_basis_real:(1+1)*n_basis_real , 1*n_basis_real:(1+1)*n_basis_real] = sp.eye(n_basis_real)
    gradZ[ind_off] = gradZ_T2T2.tocsc()
    gradZ_S2S2 = sp.lil_matrix((n_pdof,n_pdof))
    gradZ_S2S2[2*n_basis_real:(2+1)*n_basis_real , 2*n_basis_real:(2+1)*n_basis_real] = sp.eye(n_basis_real)
    gradZ[ind_off+1] = gradZ_S2S2.tocsc()
    
    return gradZ

def get_real_2ndharminic_Msparse_gradZTS_S(C_T2, C_S2, S1, G2ddinv, G1invdagPdaglist, G2invdagPdaglist, Proj_vec_S2_list):
    n_basis = G1invdagPdaglist[0].shape[0]
    n_basis_real = 2 * n_basis
    n_proj_vec = len(Proj_vec_S2_list)
    n_proj_mat = len(G1invdagPdaglist)
    n_pdof = 3 * n_basis_real
    
    Lagnum = 2*n_proj_vec + 8*n_proj_mat  + 2
    gradZTS_S = [None] * Lagnum
    
    for i in range(n_proj_vec):
        ind_vec_re = i*2
        ind_vec_im = i*2 + 1
        Proj = Proj_vec_S2_list[i]
        gradZTS_S_vec_re = np.zeros(3*n_basis_real)
        gradZTS_S_vec_im = np.zeros(3*n_basis_real)
        gradZTS_S_vec_re[2*n_basis_real:(2+1)*n_basis_real] = cmplxtoreal_matrix(G2ddinv).T @ np.append(Proj,np.zeros_like(Proj))
        gradZTS_S_vec_im[2*n_basis_real:(2+1)*n_basis_real] = cmplxtoreal_matrix(G2ddinv).T @ np.append(np.zeros_like(Proj),Proj)
        
        gradZTS_S[ind_vec_re] = gradZTS_S_vec_re
        gradZTS_S[ind_vec_im] = gradZTS_S_vec_im
        
    ind_off_vec=2*n_proj_vec
    for i in range(n_proj_mat):
        for j in range(4):
            ind_re=ind_off_vec + i*8 + j
            ind_im=ind_off_vec + i*8 + j+4
            gradZTS_S_re = np.zeros(3*n_basis_real)
            gradZTS_S_im = np.zeros(3*n_basis_real)
            G1invdagPdag = G1invdagPdaglist[i]
            G2invdagPdag = G2invdagPdaglist[i]
            if j==0:
                gradZTS_S_re[0*n_basis_real:(0+1)*n_basis_real] = cmplxtoreal_vector(G1invdagPdag @ S1)
                gradZTS_S_im[0*n_basis_real:(0+1)*n_basis_real] = cmplx_multiplier_matrix(n_basis,0,1) @ cmplxtoreal_vector(G1invdagPdag @ S1)
            elif j==1:
                print('...')
            elif j==2:
                gradZTS_S_re[1*n_basis_real:(1+1)*n_basis_real] = cmplxtoreal_vector(G2invdagPdag @ S1)
                gradZTS_S_im[1*n_basis_real:(1+1)*n_basis_real] = cmplx_multiplier_matrix(n_basis,0,-1) @ cmplxtoreal_vector(G2invdagPdag @ S1)
            else: # j==3
                print('...')
            gradZTS_S[ind_re] = gradZTS_S_re
            gradZTS_S[ind_im] = gradZTS_S_im
    
    ind_off = 2*n_proj_vec + 8*n_proj_mat
    gradZTS_S[ind_off] = np.zeros(3*n_basis_real)
    gradZTS_S[ind_off+1] = np.zeros(3*n_basis_real)
    
    return gradZTS_S



def get_real_msmf_Msparse_gradZTT(UP11list,UP22list,UP12list,UP21list): # all complex matrix
    n_basis = UP11list[0].shape[0]
    n_basis_real =  2 * n_basis
    n_S=2
    n_pdof = n_S * n_basis_real #number of primal dofs, corresponding to size of ZTT / gradZTT matrices
    n_proj = len(UP11list) #number of projection operators
    n_cplx_projLags = n_S**2

    Lagnum = n_proj * 2 * n_cplx_projLags
    gradZ = [None] * Lagnum
    
    for l in range(n_proj):
        ind_offset_Lags = l * 2 * n_cplx_projLags

        for i in range(n_S):
            for j in range(n_S): #go through all the cross-source constraints
                ind_re = ind_offset_Lags + i*n_S + j
                ind_im = ind_offset_Lags + n_cplx_projLags + i*n_S + j
                #create the gradient matrices for these particular indices
                gradZ_re = sp.lil_matrix((n_pdof,n_pdof))
                gradZ_im = sp.lil_matrix((n_pdof,n_pdof))
                """
                if i==j:
                    SymUMP = (UMP + UMP.conj().T) / 2
                    AsymUMP = (UMP - UMP.conj().T) / 2j
                    gradZ_re[i*n_basis_real:(i+1)*n_basis_real , i*n_basis_real:(i+1)*n_basis_real] = cmplxtoreal_matrix(SymUMP)   
                    gradZ_im[i*n_basis_real:(i+1)*n_basis_real , i*n_basis_real:(i+1)*n_basis_real] = cmplxtoreal_matrix(AsymUMP)
                else:
                    UMPH = UMP.conj().T
                    gradZ_re[i*n_basis_real:(i+1)*n_basis_real , j*n_basis_real:(j+1)*n_basis_real] = UMP/2
                    gradZ_re[j*n_basis_real:(j+1)*n_basis_real , i*n_basis_real:(i+1)*n_basis_real] = UMPH/2
                    gradZ_im[i*n_basis_real:(i+1)*n_basis_real , j*n_basis_real:(j+1)*n_basis_real] = -1j*UMP/2
                    gradZ_im[j*n_basis_real:(j+1)*n_basis_real , i*n_basis_real:(i+1)*n_basis_real] = 1j*UMPH/2
                """
                if i==0 and j==0:
                    UMP=UP11list[l]
                    print('ind_im',ind_im)
                    SymUMP = (UMP + UMP.conj().T) / 2
                    AsymUMP = (UMP - UMP.conj().T) / 2j
                    gradZ_re[i*n_basis_real:(i+1)*n_basis_real , i*n_basis_real:(i+1)*n_basis_real] = cmplxtoreal_matrix(SymUMP)   
                    gradZ_im[i*n_basis_real:(i+1)*n_basis_real , i*n_basis_real:(i+1)*n_basis_real] = cmplxtoreal_matrix(AsymUMP)
                elif i==1 and j==1:
                    print('ind_im',ind_im)
                    UMP=UP22list[l]
                    SymUMP = (UMP + UMP.conj().T) / 2
                    AsymUMP = (UMP - UMP.conj().T) / 2j
                    gradZ_re[i*n_basis_real:(i+1)*n_basis_real , i*n_basis_real:(i+1)*n_basis_real] = cmplxtoreal_matrix(SymUMP)   
                    gradZ_im[i*n_basis_real:(i+1)*n_basis_real , i*n_basis_real:(i+1)*n_basis_real] = cmplxtoreal_matrix(AsymUMP)
                elif i==0 and j==1:
                    UMP=UP12list[l]
                    gradZ_re[i*n_basis_real:(i+1)*n_basis_real , i*n_basis_real:(i+1)*n_basis_real] = cmplxtoreal_matrix(UMP)   
                    gradZ_im[i*n_basis_real:(i+1)*n_basis_real , i*n_basis_real:(i+1)*n_basis_real] = cmplx_multiplier_matrix(n_basis,0,-1)*cmplxtoreal_matrix(UMP)
                else: #i==1 and j==0:
                    UMP=UP21list[l]
                    gradZ_re[i*n_basis_real:(i+1)*n_basis_real , i*n_basis_real:(i+1)*n_basis_real] = cmplxtoreal_matrix(UMP)   
                    gradZ_im[i*n_basis_real:(i+1)*n_basis_real , i*n_basis_real:(i+1)*n_basis_real] = cmplx_multiplier_matrix(n_basis,0,-1)*cmplxtoreal_matrix(UMP)
                    
                gradZ[ind_re] = gradZ_re.tocsc()
                gradZ[ind_im] = gradZ_im.tocsc()
    
    return gradZ

def get_real_msmf_Msparse_gradZTS_S(S1, S2, G1invdagPdaglist, G2invdagPdaglist): # S1 and S2 complex vctor but G is in complex still
    n_basis = G1invdagPdaglist[0].shape[0]
    n_basis_real = 2 * n_basis
    n_S=2
    n_proj = len(G1invdagPdaglist)
    n_cplx_projLags = n_S**2
    gradZTS_S = [None] * (2*n_cplx_projLags*n_proj)

    for l in range(n_proj):
        ind_offset_Lags = l * 2 * n_cplx_projLags
        for i in range(n_S):
            if i==0:
                GinvdagPdag = G1invdagPdaglist[l]
            else: #i==1
                GinvdagPdag = G2invdagPdaglist[l]
            for j in range(n_S):
                ind_re = ind_offset_Lags + i*n_S + j
                ind_im = ind_offset_Lags + n_cplx_projLags + i*n_S + j
                gradZTS_S_re = np.zeros(2*n_basis_real)
                gradZTS_S_im = np.zeros(2*n_basis_real)
                if i==0 and j==0:    
                    gradZTS_S_re[i*n_basis_real:(i+1)*n_basis_real] = cmplxtoreal_vector(GinvdagPdag @ S1)
                    gradZTS_S_im[i*n_basis_real:(i+1)*n_basis_real] = cmplx_multiplier_matrix(n_basis,0,1) @ cmplxtoreal_vector(GinvdagPdag @ S1)
                if i==0 and j==1: # need to be i*n_basis_real:(i+1)*n_basis_real 
                    gradZTS_S_re[i*n_basis_real:(i+1)*n_basis_real] = cmplxtoreal_vector(GinvdagPdag @ S2)
                    gradZTS_S_im[i*n_basis_real:(i+1)*n_basis_real] = cmplx_multiplier_matrix(n_basis,0,-1) @ cmplxtoreal_vector(GinvdagPdag @ S2)
                if i==1 and j==0:    
                    gradZTS_S_re[i*n_basis_real:(i+1)*n_basis_real] = cmplxtoreal_vector(GinvdagPdag @ S1)
                    gradZTS_S_im[i*n_basis_real:(i+1)*n_basis_real] = cmplx_multiplier_matrix(n_basis,0,-1) @ cmplxtoreal_vector(GinvdagPdag @ S1)
                if i==1 and j==1:    
                    gradZTS_S_re[i*n_basis_real:(i+1)*n_basis_real] = cmplxtoreal_vector(GinvdagPdag @ S2)
                    gradZTS_S_im[i*n_basis_real:(i+1)*n_basis_real] = cmplx_multiplier_matrix(n_basis,0,1) @ cmplxtoreal_vector(GinvdagPdag @ S2)
                    

                gradZTS_S[ind_re] = gradZTS_S_re
                gradZTS_S[ind_im] = gradZTS_S_im

    return gradZTS_S

def get_ZTT(Lags, O, gradZTT):
    ZTT = O.copy()
    #print("ZTT shape", ZTT.shape)
    for i in range(len(Lags)):
        #print("gradZTT[i] shape", gradZTT[i].shape)
        ZTT += Lags[i] * gradZTT[i]
    ZTT=ZTT.tocsc()
    return ZTT

def Cholesky_analyze_ZTT(O, gradZTT):
    Lags = np.random.rand(len(gradZTT))
    ZTT = get_ZTT(Lags, O, gradZTT)
    SymZTT=(ZTT+ZTT.T)/2
    print('analyzing ZTT of format and shape', ZTT.format, ZTT.shape, 'and # of nonzero elements', ZTT.count_nonzero())
    return chol.analyze(2*SymZTT)





def get_Msparse_gradZTT(UPlist):
    gradZ = []
    for i in range(len(UPlist)):
        SymUP = (UPlist[i]+UPlist[i].conj().T)/2
        AsymUP = (UPlist[i]-UPlist[i].conj().T)/(2j)
        gradZ.append(cmplxtoreal_matrix(SymUP))
        gradZ.append(cmplxtoreal_matrix(AsymUP))
    return gradZ


def get_multiSource_Msparse_gradZTT(n_S, UPlist):
    n_basis = UPlist[0].shape[0]
    n_pdof = n_S * n_basis #number of primal dofs, corresponding to size of ZTT / gradZTT matrices
    n_cplx_projLags = n_S**2

    Lagnum = len(UPlist) * 2 * n_cplx_projLags
    gradZ = [None] * Lagnum

    for l in range(len(UPlist)):
        UMP = UPlist[l]
        UMPH = UPlist[l].T.conj()
        
        SymUMP = (UMP+UMPH)/2
        AsymUMP = (UMP-UMPH)/2j
        ind_offset_Lags = l * 2 * n_cplx_projLags

        for i in range(n_S):
            for j in range(n_S): #go through all the cross-source constraints
                ind_re = ind_offset_Lags + i*n_S + j
                ind_im = ind_offset_Lags + n_cplx_projLags + i*n_S + j

                #create the gradient matrices for these particular indices
                gradZ_re = sp.lil_matrix((n_pdof,n_pdof), dtype=np.complex)
                gradZ_im = sp.lil_matrix((n_pdof,n_pdof), dtype=np.complex)

                if i==j:
                    gradZ_re[i*n_basis:(i+1)*n_basis , i*n_basis:(i+1)*n_basis] = SymUMP
                    gradZ_im[i*n_basis:(i+1)*n_basis , i*n_basis:(i+1)*n_basis] = AsymUMP
                else:
                    gradZ_re[i*n_basis:(i+1)*n_basis , j*n_basis:(j+1)*n_basis] = UMP/2
                    gradZ_re[j*n_basis:(j+1)*n_basis , i*n_basis:(i+1)*n_basis] = UMPH/2
                    gradZ_im[i*n_basis:(i+1)*n_basis , j*n_basis:(j+1)*n_basis] = -1j*UMP/2
                    gradZ_im[j*n_basis:(j+1)*n_basis , i*n_basis:(i+1)*n_basis] = 1j*UMPH/2

                gradZ[ind_re] = gradZ_re.tocsc()
                gradZ[ind_im] = gradZ_im.tocsc()

    return gradZ


def get_mSmF_Msparse_gradZTT(n_S, chilist, Ginvlist, Plist):
    """
    allows for different frequency cross source constraints
    chilist, Ginvlist are over the n_S different freqs
    Plist is over the projection constraints imposed
    """
    n_basis = Plist[0].shape[0]
    n_pdof = n_S * n_basis #number of primal dofs, corresponding to size of ZTT / gradZTT matrices
    n_proj = len(Plist) #number of projection operators
    n_cplx_projLags = n_S**2

    Lagnum = n_proj * 2 * n_cplx_projLags
    gradZ = [None] * Lagnum

    for l in range(n_proj):
        P_l = Plist[l]
        ind_offset_Lags = l * 2 * n_cplx_projLags

        for i in range(n_S):
            for j in range(n_S): #go through all the cross-source constraints
                ind_re = ind_offset_Lags + i*n_S + j
                ind_im = ind_offset_Lags + n_cplx_projLags + i*n_S + j

                #generate UMP
                Ginv_i = Ginvlist[i]; Ginv_j = Ginvlist[j]
                chi_i = chilist[i]
                UMP =  (Ginv_i.conj().T @ P_l @ Ginv_j)/np.conj(chi_i) - P_l @ Ginv_j
                #create the gradient matrices for these particular indices
                gradZ_re = sp.lil_matrix((n_pdof,n_pdof), dtype=np.complex)
                gradZ_im = sp.lil_matrix((n_pdof,n_pdof), dtype=np.complex)

                if i==j:
                    SymUMP = (UMP + UMP.conj().T) / 2
                    AsymUMP = (UMP - UMP.conj().T) / 2j
                    gradZ_re[i*n_basis:(i+1)*n_basis , i*n_basis:(i+1)*n_basis] = SymUMP
                    gradZ_im[i*n_basis:(i+1)*n_basis , i*n_basis:(i+1)*n_basis] = AsymUMP
                else:
                    UMPH = UMP.conj().T
                    gradZ_re[i*n_basis:(i+1)*n_basis , j*n_basis:(j+1)*n_basis] = UMP/2
                    gradZ_re[j*n_basis:(j+1)*n_basis , i*n_basis:(i+1)*n_basis] = UMPH/2
                    gradZ_im[i*n_basis:(i+1)*n_basis , j*n_basis:(j+1)*n_basis] = -1j*UMP/2
                    gradZ_im[j*n_basis:(j+1)*n_basis , i*n_basis:(i+1)*n_basis] = 1j*UMPH/2

                gradZ[ind_re] = gradZ_re.tocsc()
                gradZ[ind_im] = gradZ_im.tocsc()

    return gradZ







def get_Msparse_gradZTS_S(Si, GinvdagPdaglist):
    gradZTS_S = []

    for l in range(len(GinvdagPdaglist)):
        GinvdagPdag_S = cmplxtoreal_matrix(GinvdagPdaglist[l]) @ Si
        gradZTS_S.append(GinvdagPdag_S)#/2.0)
        gradZTS_S.append(cmplx_multiplier_matrix(np.int(len(Si)/2),0,1)@ GinvdagPdag_S)#*1j/2.0)
    return gradZTS_S


def get_multiSource_Msparse_gradZTS_S(n_S, Si_st, GinvdagPdaglist):
    n_basis = GinvdagPdaglist[0].shape[0]
    n_cplx_projLags = n_S**2
    gradZTS_S = [None] * (2*n_cplx_projLags*len(GinvdagPdaglist))

    for l in range(len(GinvdagPdaglist)):
        GinvdagPdag = GinvdagPdaglist[l]
        ind_offset_Lags = l * 2 * n_cplx_projLags
        for i in range(n_S):
            Si_i = Si_st[i*n_basis:(i+1)*n_basis]
            for j in range(n_S):
                ind_re = ind_offset_Lags + i*n_S + j
                ind_im = ind_offset_Lags + n_cplx_projLags + i*n_S + j
                gradZTS_S_re = np.zeros(len(Si_st), dtype=np.complex)
                gradZTS_S_re[j*n_basis:(j+1)*n_basis] = GinvdagPdag @ Si_i / 2
                gradZTS_S_im = 1j * gradZTS_S_re

                gradZTS_S[ind_re] = gradZTS_S_re
                gradZTS_S[ind_im] = gradZTS_S_im

    return gradZTS_S


def get_mSmF_Msparse_gradZTS_S(n_S, Si_st, Ginvlist, Plist):
    n_basis = Plist[0].shape[0]
    n_proj = len(Plist)
    n_cplx_projLags = n_S**2
    gradZTS_S = [None] * (2*n_cplx_projLags*n_proj)

    for l in range(n_proj):
        P_l = Plist[l]
        ind_offset_Lags = l * 2 * n_cplx_projLags
        for i in range(n_S):
            Si_i = Si_st[i*n_basis:(i+1)*n_basis]
            for j in range(n_S):
                ind_re = ind_offset_Lags + i*n_S + j
                ind_im = ind_offset_Lags + n_cplx_projLags + i*n_S + j
                gradZTS_S_re = np.zeros(len(Si_st), dtype=np.complex)
                gradZTS_S_re[j*n_basis:(j+1)*n_basis] = Ginvlist[j].conj().T @ (P_l @ Si_i) / 2
                gradZTS_S_im = 1j * gradZTS_S_re

                gradZTS_S[ind_re] = gradZTS_S_re
                gradZTS_S[ind_im] = gradZTS_S_im

    return gradZTS_S


def check_Msparse_spatialProj_Lags_validity(Lags, O, gradZTT, chofac=None, mineigtol=None):
    ZTT = get_ZTT(Lags, O, gradZTT)
    SymZTT = (ZTT + ZTT.T)/2
    #print('SymZTT',SymZTT.format)
    if not (mineigtol is None):
        ZTT -= mineigtol * sp.eye(ZTT.shape[0], format='csc')
    try:
        if chofac is None:
            ZTTcho = chol.cholesky(2*SymZTT)
            tmp = ZTTcho.L() # necessary to attempt to access raw factor for checking matrix definiteness
        else:
            ZTTcho = chofac.cholesky(2*SymZTT)
            tmp = ZTTcho.L() # see above
    except chol.CholmodNotPositiveDefiniteError:
        return False
    return True

def check_Msparse_spatialProj_incLags_validity(incLags, include, O, gradZTT, chofac=None, mineigtol=None):
    Lags = np.zeros(len(include))
    Lags[include] = incLags[:]
    return check_Msparse_spatialProj_Lags_validity(Lags, O, gradZTT, chofac=chofac, mineigtol=mineigtol)


def get_Msparse_PD_ZTT_mineig(Lags, O, gradZTT, eigvals_only=False):
    """
    assuming ZTT is PD, find its minimum eigenvalue/vector using shift invert mode of spla.eigsh
    """
    ZTT = get_ZTT(Lags, O, gradZTT)
    SymZTT = (ZTT + ZTT.T)/2
    if eigvals_only:
        try:
            eigw = spla.eigsh(2*SymZTT, k=1, sigma=0.0, which='LM', return_eigenvectors=False)
        except BaseException as err:
            print('encountered error in sparse eigenvalue evaluation', err)
            eigw = la.eigvalsh(2*SymZTT.todense())
        return eigw[0]
    else:
        try:
            eigw, eigv = spla.eigsh(2*SymZTT, k=1, sigma=0.0, which='LM', return_eigenvectors=True)
        except BaseException as err:
            print('encountered error in sparse eigenvalue evaluation', err)
            eigw, eigv = la.eigh(2*SymZTT.todense())
        return eigw[0], eigv[:,0]
    
    
def get_Msparse_ZTT_mineig(Lags, O, gradZTT, eigvals_only=False):
    ZTT = get_ZTT(Lags, O, gradZTT)
    SymZTT = (ZTT+ZTT.T)/2
    if eigvals_only:
        eigw = spla.eigsh(2*SymZTT, k=1, which='SA', return_eigenvectors=False)
        return eigw[0]
    else:
        eigw, eigv = spla.eigsh(2*SymZTT, k=1, which='SA', return_eigenvectors=True)
        return eigw[0], eigv[:,0]


def get_Msparse_inc_ZTT_mineig(incLags, include, O, gradZTT, eigvals_only=False):
    Lags = np.zeros(len(include))
    Lags[include] = incLags[:]
    return get_Msparse_ZTT_mineig(Lags, O, gradZTT, eigvals_only=eigvals_only)


def get_Msparse_inc_PD_ZTT_mineig(incLags, include, O, gradZTT, eigvals_only=False):
    Lags = np.zeros(len(include))
    Lags[include] = incLags[:]
    return get_Msparse_PD_ZTT_mineig(Lags, O, gradZTT, eigvals_only=eigvals_only)
    

###method for finding derivatives of mineig of ZTT, to use for phase I (entering domain of duality) of optimization

def get_Msparse_ZTT_mineig_grad(ZTT, gradZTT):
    SymZTT = (ZTT+ZTT.T)/2
    eigw, eigv = spla.eigsh(2*SymZTT, k=1, which='SA', return_eigenvectors=True)
    eiggrad = np.zeros(len(gradZTT))
    
    for i in range(len(eiggrad)):
        eiggrad[i] = np.real(np.vdot(eigv[:,0], (gradZTT[i]+gradZTT[i].T).dot(eigv[:,0])))
    return eigw[0], eiggrad