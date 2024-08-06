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

def get_gradZTT(Plist, G1m, G2, G2m, delta, chi1, chi2, A1, Ac, convexc=True):
	N = Plist.shape[1]
	num_regions = Plist.shape[0]
	gradZTT1 = np.zeros((num_regions, 4, 2, 2*N, 2*N), dtype=complex) # 1A, 1S, 2A, 2S, 3A, 3S, 4A, 4S

	U1 = (1/chi1).conjugate()*np.eye(N) - G1m.conjugate().T
	U2 = (1/chi2).conjugate()*np.eye(N) - G2m.conjugate().T 

	for j in range(num_regions):
		P = Plist[j]
		# T1T1
		gradZTT1[j, 0, 0, 0:N, 0:N] += zops.Asym(U1 @ P)
		gradZTT1[j, 0, 1, 0:N, 0:N] +=  zops.Sym(U1 @ P)
		gradZTT1[j, 2, 0, 0:N, 0:N] += - A1 * zops.Sym( G1m.conjugate().T @ delta @ G2.conjugate().T @ P)
		gradZTT1[j, 2, 1, 0:N, 0:N] +=   A1 * zops.Asym(G1m.conjugate().T @ delta @ G2.conjugate().T @ P)
		
		# T2T2
		gradZTT1[j, 1, 0, N:, N:] += zops.Asym(U2 @ P)
		gradZTT1[j, 1, 1, N:, N:] +=  zops.Sym(U2 @ P)
		
		# T1T2
		gradZTT1[j, 1, 0, 0:N, N:] += -(A1/2j) * G1m.conjugate().T @ delta @ G2.conjugate().T @ P 
		gradZTT1[j, 1, 1, 0:N, N:] += -(A1/2 ) * G1m.conjugate().T @ delta @ G2.conjugate().T @ P
		gradZTT1[j, 2, 0, 0:N, N:] +=  (A1/2j) * (- P @ U2.conjugate().T)
		gradZTT1[j, 2, 1, 0:N, N:] +=  (A1/2 ) * (  P @ U2.conjugate().T)
		gradZTT1[j, 3, 0, 0:N, N:] +=  ( 1/2j) * (U1 @ P)
		gradZTT1[j, 3, 1, 0:N, N:] +=  ( 1/2 ) * (U1 @ P)

	gradZTT1[:, :, :, N:, 0:N] = np.transpose(gradZTT1[:, :, :, 0:N, N:].conjugate(), (0, 1, 2, 4, 3))
	gradZTT1 = np.reshape(gradZTT1, [num_regions*4*2, 2*N, 2*N])

	# Now that P, delta, and G have changed the important values of ZTT, we can remove ZTT at the dipole 
	# This is specific to the delta function source at the center (where dipole is)
	# This will eventually be generalized
	gradZTT1 = np.delete(gradZTT1, (N//4, 3*N//4, N+N//4, N+3*N//4), axis=1)
	gradZTT1 = np.delete(gradZTT1, (N//4, 3*N//4, N+N//4, N+3*N//4), axis=2)

	if convexc: 
		gradZTT2 = np.zeros((1,    2*N, 2*N), dtype=complex) # CC
		gradZTT2[0, 0:N, 0:N]    += - Ac * (G1m.conjugate().T @ delta @ G2.conjugate().T @ G2 @ delta @ G1m)
		gradZTT2[0, N: , N: ]    += ((np.imag(chi2)/(np.abs(chi2)**2))**2)*np.eye(N)

		# Same story here 
		gradZTT2 = np.delete(gradZTT2, (N//4, 3*N//4, N+N//4, N+3*N//4), axis=1)
		gradZTT2 = np.delete(gradZTT2, (N//4, 3*N//4, N+N//4, N+3*N//4), axis=2)

		gradZTT = np.append(gradZTT1, gradZTT2, axis=0)
		return gradZTT
	return gradZTT1
	

def get_gradZTS_S(Plist, G1m, G2, delta, A1, Ac, S1, convexc=True):
	N = Plist.shape[1]
	num_regions = Plist.shape[0]
	gradZTS_S1 = np.zeros((num_regions, 4, 2, 2*N), dtype=complex)
	if convexc: gradZTS_S2 = np.zeros((num_regions, 1, 2*N), dtype=complex)
	S2 = S1.conjugate()

	for j in range(num_regions):
		P = Plist[j]
		gradZTS_S1[j, 0, 0, 0:N] += np.conjugate(S2 @ P/2j).T 
		gradZTS_S1[j, 0, 1, 0:N] += np.conjugate(S2 @ P/2 ).T
		gradZTS_S1[j, 2, 0, 0:N] += np.conjugate(S2 @ ((A1*1j/2j)*(delta @ G2.conjugate().T @ P))).T
		gradZTS_S1[j, 2, 1, 0:N] += np.conjugate(S2 @ ((A1*1j/2 )*(delta @ G2.conjugate().T @ P))).T
		if convexc: gradZTS_S2[j, 0,    0:N] += np.conjugate(S2 @ (Ac * delta @ G2.conjugate().T @ G2 @ delta @ G1m)).T

		gradZTS_S1[j, 3, 0, N: ] += np.conjugate(S2 @ P/2j).T
		gradZTS_S1[j, 3, 1, N: ] += np.conjugate(S2 @ P/2 ).T
		gradZTS_S1[j, 1, 0, N: ] += np.conjugate(S2 @ ((1j*A1/2j)*(delta @ G2.conjugate().T @ P))).T
		gradZTS_S1[j, 1, 1, N: ] += np.conjugate(S2 @ ((1j*A1/2 )*(delta @ G2.conjugate().T @ P))).T

	gradZTS_S1 = np.reshape(gradZTS_S1, [num_regions*4*2, 2*N])
	gradZTS_S1 = np.delete(gradZTS_S1, (N//4, 3*N//4, N+N//4, N+3*N//4), axis=1) # See comment in ZTT 

	if convexc:
		gradZTS_S2 = np.reshape(gradZTS_S2, [num_regions*1, 2*N])
		gradZTS_S2 = np.delete(gradZTS_S2, (N//4, 3*N//4, N+N//4, N+3*N//4), axis=1) # See comment in ZTT

		gradZTS_S = np.append(gradZTS_S1, gradZTS_S2, axis=0)
		return gradZTS_S
	return gradZTS_S1


def get_S_gradZSS_S(numlag, G2, delta, Ac, S, dl):
	S_gradZSS_S = np.zeros(numlag, dtype=complex)
	S_gradZSS_S[-1] = Ac*np.dot(np.conjugate(S), delta @ G2.conjugate().T @ G2 @ delta @ S)*dl*dl
	return S_gradZSS_S

# def check_spatialProj_Lags_validity(n_S, Lags, O, gradZTT):
# 	ZTT = get_ZTT(n_S, Lags, O, gradZTT)
# 	N = O.shape[0]//2
# 	try:
# 		_ = la.cholesky(ZTT)
# 		return 1
# 	except la.LinAlgError:
# 		return -1

# def check_spatialProj_incLags_validity(n_S, incLags, include, O_quad, gradZTT):
#     Lags = np.zeros(len(include), dtype=np.double)
#     Lags[include] = incLags[:]
#     return check_spatialProj_Lags_validity(n_S, Lags, O_quad, gradZTT)

# def Lags_normsqr(Lags):
# 	return np.sum(Lags*Lags), 2*Lags

# def Lags_normsqr_Hess_np(Lags):
# 	return 2*np.eye(len(Lags))

# def get_ZTT_mineig(n_S, Lags, O, gradZTT, eigvals_only=False):
#     ZTT = get_ZTT(n_S, Lags, O, gradZTT)
#     if eigvals_only:
#         eigw = la.eigvalsh(ZTT)
#         return eigw[0]
#     else:
#         eigw, eigv = la.eigh(ZTT)
#         return eigw[0], eigv[:,0]

# def get_ZTT_mineig_grad(ZTT, gradZTT):
#     eigw, eigv = la.eigh(ZTT)
#     eiggrad = np.zeros(len(gradZTT))
#     for i in range(len(eiggrad)):
#         eiggrad[i] = np.real(np.vdot(eigv[:,0], gradZTT[i] @ eigv[:,0]))
#     return eiggrad

# def get_inc_ZTT_mineig(n_S, incLags, include, O, gradZTT, eigvals_only=False):
#     Lags = np.zeros(len(include))
#     Lags[include] = incLags[:]
#     return get_ZTT_mineig(n_S, Lags, O, gradZTT, eigvals_only=eigvals_only)

# def get_ZTT_gradmineig(n_S, incLags, include, O, gradZTT): #Ulist, Plist):
#     Lags = np.zeros(len(include))
#     Lags[include] = incLags[:]
#     ZTT = get_ZTT(n_S, Lags, O, gradZTT)
#     mineigJac = np.zeros((1,len(incLags)))
#     mineigJac[0,:] = get_ZTT_mineig_grad(ZTT, gradZTT)[include]
#     return mineigJac

# def spatialProjopt_find_feasiblept(n_S, Lagnum, include, O, gradZTT, maxiter):
# 	incLagnum = np.sum(include)
# 	initincLags = np.random.rand(incLagnum)

# 	mineigincfunc = lambda incL: get_inc_ZTT_mineig(n_S, incL, include, O, gradZTT, eigvals_only=True)
# 	Jacmineigincfunc = lambda incL: get_ZTT_gradmineig(n_S, incL, include, O, gradZTT)

# 	tolcstrt = 1e-4
# 	cstrt = sopt.NonlinearConstraint(mineigincfunc, tolcstrt, np.inf, jac=Jacmineigincfunc, keep_feasible=False)

# 	# np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
# 	lb = -np.inf*np.ones(incLagnum)
# 	ub = np.inf*np.ones(incLagnum)
# 	bnds = sopt.Bounds(lb,ub)
# 	try:
# 		res = sopt.minimize(Lags_normsqr, initincLags, method='trust-constr', jac=True, hess=Lags_normsqr_Hess_np,
# 							bounds=bnds, constraints=cstrt, options={'verbose':2,'maxiter':maxiter})
# 	except ValueError:
# 		global feasiblept
# 		Lags = np.zeros(Lagnum)
# 		Lags[include] = feasiblept
# 		return Lags

# 	Lags = np.zeros(Lagnum)
# 	Lags[include] = res.x
# 	Lags[1] = np.abs(Lags[1]) + 0.01
# 	return Lags