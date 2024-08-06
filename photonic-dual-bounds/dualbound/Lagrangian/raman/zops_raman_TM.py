'''
File containing the derivatives and helper functions for the Z operators for the Raman bounds problem
'''

import numpy as np
import scipy.linalg as la 
import matplotlib.pyplot as plt 
import scipy.optimize as sopt
import sys
sys.path.append('../../../')
from dualbound.Lagrangian.zops_utils import zops

def get_gradZTT(Ac, Acc, Plist, U1, U2, G1rd, G2dr, alpha, chi2, convexc=True):
	D = Plist.shape[1]
	num_regions = Plist.shape[0]
	gradZTT1 = np.zeros((num_regions, 4, 2, 2*D, 2*D), dtype=complex) # 1A, 1S, 2A, 2S, 3A, 3S, 4A, 4S
	for j in range(num_regions):
		P = Plist[j]
		# T1T1
		gradZTT1[j, 0, 0, 0:D, 0:D] +=  zops.Asym(U1 @ P)
		gradZTT1[j, 0, 1, 0:D, 0:D] +=  zops.Sym(U1 @ P)
		gradZTT1[j, 3, 0, 0:D, 0:D] += - Ac * zops.Asym(G1rd.conjugate().T @ alpha.conjugate().T @ G2dr.conjugate().T @ P)
		gradZTT1[j, 3, 1, 0:D, 0:D] += - Ac *  zops.Sym(G1rd.conjugate().T @ alpha.conjugate().T @ G2dr.conjugate().T @ P)
		
		# T2T2
		gradZTT1[j, 1, 0, D:, D:] += zops.Asym(U2 @ P)
		gradZTT1[j, 1, 1, D:, D:] +=  zops.Sym(U2 @ P)
		
		# T1T2
		gradZTT1[j, 1, 0, 0:D, D:] += - Ac * G1rd.conjugate().T @ alpha.conjugate().T @ G2dr.conjugate().T @ P / 2j
		gradZTT1[j, 1, 0, D:, 0:D] += gradZTT1[j, 1, 0, 0:D, D:].T.conjugate() 
		gradZTT1[j, 1, 1, 0:D, D:] += - Ac * G1rd.conjugate().T @ alpha.conjugate().T @ G2dr.conjugate().T @ P / 2
		gradZTT1[j, 1, 1, D:, 0:D] += gradZTT1[j, 1, 1, 0:D, D:].T.conjugate()

		gradZTT1[j, 2, 0, 0:D, D:] += U1 @ P / 2j 
		gradZTT1[j, 2, 0, D:, 0:D] += gradZTT1[j, 2, 0, 0:D, D:].T.conjugate()
		gradZTT1[j, 2, 1, 0:D, D:] += U1 @ P / 2
		gradZTT1[j, 2, 1, D:, 0:D] += gradZTT1[j, 2, 1, 0:D, D:].T.conjugate()

		gradZTT1[j, 3, 0, D:, 0:D] +=  (1/2j) * (U2 @ P)
		gradZTT1[j, 3, 0, 0:D, D:] += gradZTT1[j, 3, 0, D:, 0:D].T.conjugate()
		gradZTT1[j, 3, 1, D:, 0:D] +=  ( 1/2 ) * (U2 @ P)
		gradZTT1[j, 3, 1, 0:D, D:] +=  gradZTT1[j, 3, 1, D:, 0:D].T.conjugate()

	gradZTT1 = np.reshape(gradZTT1, [num_regions*4*2, 2*D, 2*D])

	if convexc:
		gradZTT2 = np.zeros((1,    2*D, 2*D), dtype=complex) # CC
		gradZTT2[0, 0:D, 0:D]    += - Acc * (G1rd.conjugate().T @ alpha.conjugate().T @ G2dr.conjugate().T @ G2dr @ alpha @ G1rd)
		gradZTT2[0, D: , D: ]    += ((np.imag(chi2)/(np.abs(chi2)**2))**2)*np.eye(D)
		gradZTT = np.append(gradZTT1, gradZTT2, axis=0)
		return gradZTT

	return gradZTT1
	
def include_helper(num_regions, code, convexc=True):
	include1 = np.zeros((num_regions, 4, 2), dtype=bool)
	for j in range(num_regions):
		include1[j, 0, 0] = eval(str(code)[0])
		include1[j, 0, 1] = eval(str(code)[1])
		include1[j, 1, 0] = eval(str(code)[2])
		include1[j, 1, 1] = eval(str(code)[3])
		include1[j, 2, 0] = eval(str(code)[4])
		include1[j, 2, 1] = eval(str(code)[5])
		include1[j, 3, 0] = eval(str(code)[6])
		include1[j, 3, 1] = eval(str(code)[7])

	include1 = np.reshape(include1, [num_regions*4*2])
	if convexc: 
		include2 = np.zeros((1), dtype=bool)
		include2[0] = 1 
		include = np.append(include1, include2, axis=0)
		return include
	return include1

def get_gradZTS_S(Ac, Acc, Plist, p, delta, alpha, G2dr, G1rd, S1, convexc=True):
	D = Plist.shape[1]
	num_regions = Plist.shape[0]
	gradZTS_S1 = np.zeros((num_regions, 4, 2, 2*D), dtype=complex)

	for j in range(num_regions):
		P = Plist[j]
		gradZTS_S1[j, 0, 0, 0:D] += - P @ p @ S1 / 2j
		gradZTS_S1[j, 0, 1, 0:D] +=   P @ p @ S1 / 2
		
		gradZTS_S1[j, 1, 0, D: ] += - Ac * P @ G2dr @ alpha @ delta @ S1 / 2j
		gradZTS_S1[j, 1, 1, D: ] +=   Ac * P @ G2dr @ alpha @ delta @ S1 / 2

		gradZTS_S1[j, 2, 0, D: ] += - P @ p @ S1 / 2j 
		gradZTS_S1[j, 2, 1, D: ] +=   P @ p @ S1 / 2

		gradZTS_S1[j, 3, 0, 0:D] += - Ac * P @ G2dr @ alpha @ delta @ S1 / 2j
		gradZTS_S1[j, 3, 1, 0:D] +=   Ac * P @ G2dr @ alpha @ delta @ S1 / 2

	gradZTS_S1 = np.reshape(gradZTS_S1, [num_regions*4*2, 2*D])

	if convexc:
		gradZTS_S2 = np.zeros((1, 2*D), dtype=complex)
		gradZTS_S2[0, 0:D] += Acc * G1rd.T.conjugate() @ alpha.T.conjugate() @ G2dr.T.conjugate() @ G2dr @ alpha @ delta @ S1	
		gradZTS_S = np.append(gradZTS_S1, gradZTS_S2, axis=0)
		return gradZTS_S

	return gradZTS_S1

def get_S_gradZSS_S(Acc, numlag, dl, delta, alpha, G2dr, S):
	S_gradZSS_S = np.zeros(numlag, dtype=complex)
	S_gradZSS_S[-1] = Acc * np.conjugate(S) @ delta.T @ alpha.T.conjugate() @ G2dr.T.conjugate() @ G2dr @ alpha @ delta @ S * dl*dl
	return S_gradZSS_S

def get_gradZTT_real_mp1(Ac, U1, U2, G1rd, G2dr, alpha, Plist):
	D = U1.shape[1]
	gradZTT = np.zeros((4, 2*D, 2*D), dtype=np.complex128) 
	P0, P1, P2, P3 = Plist[0], Plist[1], Plist[2], Plist[3]

	# T1T1
	gradZTT[0, 0:D, 0:D] +=  zops.Sym(U1 @ P0)
	gradZTT[3, 0:D, 0:D] += - Ac *  zops.Sym(G1rd.conjugate().T @ alpha.conjugate().T @ G2dr.conjugate().T @ P3)
	
	# T2T2
	gradZTT[1, D:, D:] +=  zops.Sym(U2 @ P1)
	
	# T1T2
	gradZTT[1, 0:D, D:] += - Ac * G1rd.conjugate().T @ alpha.conjugate().T @ G2dr.conjugate().T @ P1 / 2
	gradZTT[1, D:, 0:D] += gradZTT[1, 0:D, D:].T.conjugate()

	gradZTT[2, 0:D, D:] += U1 @ P2 / 2
	gradZTT[2, D:, 0:D] += gradZTT[2, 0:D, D:].T.conjugate()

	gradZTT[3, D:, 0:D] += U2 @ P3 / 2
	gradZTT[3, 0:D, D:] += gradZTT[3, D:, 0:D].T.conjugate()

	return gradZTT

def get_gradZTS_S_real_mp1(Ac, p, delta, alpha, G2dr, S1, Plist):
	P0, P1, P2, P3 = Plist[0], Plist[1], Plist[2], Plist[3]
	D = Plist[0].shape[0]
	gradZTS_S = np.zeros((4, 2*D), dtype=complex)

	gradZTS_S[0, 0:D] +=   P0.T.conjugate() @ p @ S1 / 2
	gradZTS_S[1, D: ] +=   Ac * P1.T.conjugate() @ G2dr @ alpha @ delta @ S1 / 2
	gradZTS_S[2, D: ] +=   P2.T.conjugate() @ p @ S1 / 2
	gradZTS_S[3, 0:D] +=   Ac * P3.T.conjugate() @ G2dr @ alpha @ delta @ S1 / 2

	return gradZTS_S

def get_new_vecs_raman_heuristic(N, cclasses, get_gradZTT, get_gradZTS_S, T, ZTT, niters, U1, U2, S1, p):
	T1, T2 = T[0:N], T[N:]
	violation1 = -(U1.T.conj() @ T1).conj() * T1 + (p @ S1).conj() * T1
	violation2 = -(U2.T.conj() @ T2).conj() * T2 + (p @ S1).conj() * T2
	violation3 = -(U1.T.conj() @ T1).conj() * T2 + (p @ S1).conj() * T2
	violation4 = -(U2.T.conj() @ T2).conj() * T1 + (p @ S1).conj() * T1
	violations = [violation1, violation2, violation3, violation4]
	
	Laggradfac_phase = np.zeros((cclasses, N), dtype=complex)
	for i, v in enumerate(violations):
		Laggradfac_phase[i, :] = -v / np.abs(v)

	eigw, eigv = la.eigh(ZTT)
	eigw = eigw[0]
	eigv = eigv[:,0]
	x1, x2 = eigv[0:N], eigv[N:]

	mineig_phase = np.zeros((cclasses, N), dtype=complex)
	minfac1 = (U1.T.conj() @ x1).conj() * x1
	minfac2 = (U2.T.conj() @ x2).conj() * x2
	minfac3 = (U1.T.conj() @ x1).conj() * x2
	minfac4 = (U2.T.conj() @ x2).conj() * x1
	minfac = [minfac1, minfac2, minfac3, minfac4]

	for i, m in enumerate(minfac):
		mineig_phase[i, :] = m / np.abs(m)

	new_vecs = np.zeros((cclasses, N), dtype=complex)
	for i in range(cclasses):
		new_vecs[i, :] = np.conj(Laggradfac_phase[i, :] + mineig_phase[i, :])
		new_vecs[i, :] *= np.abs(np.real(-new_vecs[i, :] * violations[i])) 
		new_vecs[i, :][np.isnan(new_vecs[i, :])] = 0.0

	return new_vecs
