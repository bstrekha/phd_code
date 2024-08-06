#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 21:46:00 2022

@author: Alessio Amaolo
This file contains helper functions that compute Z operators and gradients for
calculations with 1 source and 2 materials. It only implements global constraints.
"""

import numpy as np
import scipy.linalg as la
from numba import njit

def get_gradZTT(chiinvPlist, PGlist, Plist, n, N):
    '''
    Evaluates dZTT/d\lambda.

    Parameters
    ----------
    n : integer
        number of materials
    N : integer
        length of optimization vectors
    '''
    numpixels = Plist.shape[0]
    # print(len(chiinvPlist))
    S, A, CC = range(3) # 3 is the number of Lagrange multipliers that aren't of \lambda_{ A/S Oab}
    # There are n C 2 pairs, and each has 2 Lagrange multipliers, so n!/((n-2)!*2) * 2 = n*(n-1) multiplier
    ortho = np.array(range(n*(n-1))) + 3
    numlag = len(ortho) + 3 # lagrange multipliers (per pixel)
    # print(R, I, CC)
    # print(ortho)
    gradZTT = np.zeros((numlag*numpixels, N*(n+1), N*(n+1)), dtype=complex)
    # jPointTracker = ''

    for i in range(numpixels): # since Plist will always contain the identity, this will always be at least 1
        # print(i)
        PG = PGlist[i]
        PGH = PGlist[i].T.conjugate()
        SymPG = (PG + PGH)/2
        AsymPG = (PG - PGH)/(2j)

        Pj = Plist[i]

        SymTerm =  np.zeros((N*(n+1), N*(n+1)), dtype=complex)
        AsymTerm = np.zeros((N*(n+1), N*(n+1)), dtype=complex)
        CCTerm = np.zeros((N*(n+1), N*(n+1)), dtype=complex)

        SymTerm[0:N, 0:N] = -SymPG
        AsymTerm[0:N, 0:N] = AsymPG #+ np.eye(N, dtype=complex)*1e-14
        CCTerm[0:N, 0:N] = Pj

        for m in range(1, n+1):
            idx = m*N
            nidx = (m+1)*N
            chi_invP_m  = chiinvPlist[m-1, i]
            chi_invP_mH = chiinvPlist[m-1, i].T.conjugate()
            Sym_chi_invP_m = (chi_invP_m + chi_invP_mH)/2
            Asym_chi_invP_m = (chi_invP_m - chi_invP_mH)/(2j)

            SymTerm[idx:nidx, idx:nidx] += Sym_chi_invP_m
            AsymTerm[idx:nidx, idx:nidx] += -Asym_chi_invP_m
            CCTerm[idx:nidx, idx:nidx] += -Pj

        jPoint = i*numlag
        gradZTT[jPoint+S, :, :] = SymTerm
        gradZTT[jPoint+A, :, :] = AsymTerm
        gradZTT[jPoint+CC,:, :] = CCTerm

        # jPointTracker += f"{jPoint+S} {jPoint+A} {jPoint+CC} "
        # Compute Z^tt
        lagtracker = 0
        for a in range(1, n+1):
            for b in range(a+1, n+1):
                # jPointTracker += f"{jPoint+ortho[lagtracker]} {jPoint+ortho[lagtracker+1]} "

                gradZTT[jPoint + ortho[lagtracker  ], a*N:(a+1)*N, b*N:(b+1)*N] += -Pj/2
                gradZTT[jPoint + ortho[lagtracker+1], a*N:(a+1)*N, b*N:(b+1)*N] += -Pj/(2j)

                gradZTT[jPoint + ortho[lagtracker  ], b*N:(b+1)*N, a*N:(a+1)*N] += -Pj/2
                gradZTT[jPoint + ortho[lagtracker+1], b*N:(b+1)*N, a*N:(a+1)*N] += Pj/(2j)
                lagtracker += 2

    # print(f"Order of jPoints: {np.fromstring(jPointTracker, sep=' ', dtype=int)}")
    return gradZTT


def get_gradZTS_S(S1, Plist, n, N):
    numpixels = len(Plist)
    S, A, CC = range(3)
    ortho = np.array(range(n*(n-1))) + 3
    numlag = len(ortho) + 3 # lagrange multipliers (per pixel
    gradZTS_S = np.zeros((numlag*numpixels, N*(n+1)), dtype=complex)
    for i in range(numpixels):
        Pj = Plist[i]
        SymTerm = np.zeros((N*(n+1), N), dtype=complex)
        AsymTerm = np.zeros((N*(n+1), N), dtype=complex)

        SymTerm[0:N, 0:N] = Pj/2
        AsymTerm[0:N, 0:N] = -Pj/(2j)

        jPoint = i*numlag
        gradZTS_S[jPoint + S, :] = SymTerm @ S1
        gradZTS_S[jPoint + A, :] = AsymTerm @ S1

    return gradZTS_S

# @njit
def get_gradZTT_small(chiinvPlist, PGlist, Plist, n, N, numpixels):
    '''
    Evaluates dZTT/d\lambda.

    Parameters
    ----------
    n : integer
        number of materials
    N : integer
        length of optimization vectors
    '''
    S, A, CC = range(3) # 3 is the number of Lagrange multipliers that aren't of \lambda_{ A/S Oab}
    # There are n C 2 pairs, and each has 2 Lagrange multipliers, so n!/((n-2)!*2) * 2 = n*(n-1) multiplier
    ortho = np.array(range(n*(n-1))) + 3
    numlag = len(ortho) + 3 # lagrange multipliers (per pixel)
    gradZTT = np.zeros((numlag*numpixels, N*n, N*n), dtype=complex)

    for i in range(numpixels): # since Plist will always contain the identity, this will always be at least 1
        PG = PGlist[i]
        PGH = PGlist[i].T.conjugate()
        SymPG = (PG + PGH)/2
        AsymPG = (PG - PGH)/(2j)

        Pj = Plist[i]

        SymTerm =  np.zeros((N*n, N*n), dtype=complex)
        AsymTerm = np.zeros((N*n, N*n), dtype=complex)
        CCTerm = np.zeros((N*n, N*n), dtype=complex)

        # SymTerm[0:N, 0:N] = -SymPG
        # AsymTerm[0:N, 0:N] = AsymPG
        # CCTerm[0:N, 0:N] = Pj

        for m in range(n):
            idx = m*N
            nidx = (m+1)*N
            chi_invP_m  = chiinvPlist[m, i]
            chi_invP_mH = chiinvPlist[m, i].T.conjugate()
            Sym_chi_invP_m = (chi_invP_m + chi_invP_mH)/2
            Asym_chi_invP_m = (chi_invP_m - chi_invP_mH)/(2j)

            SymTerm[idx:nidx, idx:nidx] += Sym_chi_invP_m
            AsymTerm[idx:nidx, idx:nidx] += -Asym_chi_invP_m
            CCTerm[idx:nidx, idx:nidx] += -Pj

            for k in range(n):
                kidx = k*N
                nkidx = (k+1)*N
                SymTerm[idx:nidx, kidx:nkidx] += -SymPG
                AsymTerm[idx:nidx, kidx:nkidx] += AsymPG
                CCTerm[idx:nidx, kidx:nkidx] += Pj

        jPoint = i*numlag
        gradZTT[jPoint+S, :, :] = SymTerm
        gradZTT[jPoint+A, :, :] = AsymTerm
        gradZTT[jPoint+CC,:, :] = CCTerm

        # Compute Z^tt
        lagtracker = 0
        for a in range(n):
            for b in range(a+1, n):
                # jPointTracker += f"{jPoint+ortho[lagtracker]} {jPoint+ortho[lagtracker+1]} "

                gradZTT[jPoint + ortho[lagtracker  ], a*N:(a+1)*N, b*N:(b+1)*N] += -Pj/2
                gradZTT[jPoint + ortho[lagtracker+1], a*N:(a+1)*N, b*N:(b+1)*N] += -Pj/(2j)

                gradZTT[jPoint + ortho[lagtracker  ], b*N:(b+1)*N, a*N:(a+1)*N] += -Pj/2
                gradZTT[jPoint + ortho[lagtracker+1], b*N:(b+1)*N, a*N:(a+1)*N] += Pj/(2j)
                lagtracker += 2

    # print(f"Order of jPoints: {np.fromstring(jPointTracker, sep=' ', dtype=int)}")
    return gradZTT

def get_gradZTS_S_small(S1, Plist, n, N):
    numpixels = len(Plist)
    S, A, CC = range(3)
    ortho = np.array(range(n*(n-1))) + 3
    numlag = len(ortho) + 3 # lagrange multipliers (per pixel
    gradZTS_S = np.zeros((numlag*numpixels, N*n), dtype=complex)
    for i in range(numpixels):
        Pj = Plist[i]
        SymTerm = np.zeros((N*n, N), dtype=complex)
        AsymTerm = np.zeros((N*n, N), dtype=complex)

        for j in range(n):
            SymTerm[j*N:(j+1)*N, :] = Pj/2
            AsymTerm[j*N:(j+1)*N, :] = -Pj/(2j)

        jPoint = i*numlag
        gradZTS_S[jPoint + S, :] = SymTerm @ S1
        gradZTS_S[jPoint + A, :] = AsymTerm @ S1

    return gradZTS_S

# def get_gradZSS(S, factor):
#     np2 = len(S)
#     gradZSS = np.array([np.zeros((np2,np2)),np.zeros((np2,np2)),np.zeros((np2,np2)),np.zeros((np2,np2))], dtype=object)
#     gradZSS[-1][0:np2-2, 0:np2-2] = -factor*np.eye(np2-2, np2-2)
#     return gradZSS
#
# def get_ZTT(n_S, Lags, O, gradZTT):
#     ZTT = O.copy()
#     for i in range(len(Lags)):
#         #print('ZTT shape', ZTT.shape)
#         #print('gradZTT[i].shape', gradZTT[i].shape, flush=True)
#         ZTT += ((gradZTT[i].astype(complex))*Lags[i])
#     return ZTT
#
# def check_spatialProj_Lags_validity(n_S, Lags, O, gradZTT):
#     ZTT = get_ZTT(n_S, Lags, O, gradZTT)
#     try:
#         _ = la.cholesky(ZTT)
#         return 1
#     except la.LinAlgError:
#         return -1
#
# def check_spatialProj_incLags_validity(n_S, incLags, include, O, gradZTT):
#     # This is N source since it doesn't make a difference
#     Lags = np.zeros(len(include), dtype=np.double)
#     Lags[include] = incLags[:]
#     return check_spatialProj_Lags_validity(n_S, Lags, O, gradZTT)
#
# def get_ZTT_mineig(n_S, Lags, O, gradZTT, eigvals_only=False):
#     ZTT = get_ZTT(n_S, Lags, O, gradZTT)
#     if eigvals_only:
#         eigw = la.eigvalsh(ZTT)
#         return eigw[0]
#     else:
#         eigw, eigv = la.eigh(ZTT)
#         return eigw[0], eigv[:,0]
#
# def get_inc_ZTT_mineig(n_S, incLags, include, O, gradZTT, eigvals_only=False):
#     Lags = np.zeros(len(include))
#     Lags[include] = incLags[:]
#     return get_ZTT_mineig(n_S, Lags, O, gradZTT, eigvals_only=eigvals_only)
#
# if __name__ == '__main__':
#     print("Testing sptialProjopt_Zops_1source_2materials_singlematrix_numpy...\n")
#     S = np.array([2,4,2+1j, 6j, -1])
#     n = len(S)
#     np.random.seed(12345)
#     U1P = np.random.rand(n,n)
#     U2P = np.random.rand(n,n)
#     gradZTT = get_gradZTT(S, U1P, U2P)
#     print('|S><S| = \n', np.outer(S,S))
#     print('|S>/2 = \n', S/2)
#     print('U1P = \n', U1P)
#     print('U2P = \n', U2P)
#     for el in gradZTT:
#         print(el)
#
#     print()
#
#     grad_S = get_gradZTS_S(S)
#     print(grad_S)
