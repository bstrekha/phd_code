import numpy as np 
import scipy.sparse as sp
import scipy.linalg as la
# from numba import njit 
from jax import jit 
import jax.numpy as jnp
import jax.scipy.linalg as jla

@jit
def get_ZTT(Lags, O, gradZTT):
    return O + np.sum(Lags[:, None, None] * gradZTT, axis=0)

@jit
def get_ZTS_S(Lags, O_lin_S, gradZTS_S):
    return O_lin_S + np.sum(Lags[:, None] * gradZTS_S, axis=0)

@jit
def get_S_ZSS_S(Lags, S_gradZSS_S):
    return np.sum(Lags * S_gradZSS_S)

@jit
def get_ZTTcho_Tvec(ZTT, ZTS_S):
    ZTT_chofac = jla.cho_factor(ZTT)
    Tvec = jla.cho_solve(ZTT_chofac, ZTS_S)
    return ZTT_chofac, Tvec

def get_ZTTcho_Tvec_gradTvec(ZTT, gradZTT, ZTS_S, gradZTS_S):
    ZTT_chofac = la.cho_factor(ZTT) #later on need many solves with ZTT as coeff matrix, so do decomposition beforehand
    Tvec = la.cho_solve(ZTT_chofac, ZTS_S)
    
    gradTvec = []
    for i in range(len(gradZTT)):
        gradTvec.append(la.cho_solve(ZTT_chofac, -gradZTT[i] @ Tvec + gradZTS_S[i]))
    
    return ZTT_chofac, Tvec, gradTvec

def get_ZTT_mineig(Lags, O, gradZTT, eigvals_only=False):
    ZTT = get_ZTT(Lags, O, gradZTT)
    if eigvals_only:
        eigw = la.eigvalsh(ZTT)
        return eigw[0]
    else:
        eigw, eigv = la.eigh(ZTT)
        return eigw[0], eigv[:,0]

def get_inc_ZTT_mineig(incLags, include, O, gradZTT, eigvals_only=False):
    Lags = np.zeros(len(include))
    Lags[include] = incLags[:]
    return get_ZTT_mineig(Lags, O, gradZTT, eigvals_only=eigvals_only)

def get_ZTT_mineig_grad(ZTT, gradZTT):
    eigw, eigv = la.eigh(ZTT)
    eiggrad = np.zeros(len(gradZTT))
    for i in range(len(eiggrad)):
        eiggrad[i] = np.real(np.vdot(eigv[:,0], gradZTT[i] @ eigv[:,0]))
    return eiggrad

def get_ZTT_gradmineig(incLags, include, O, gradZTT): #Ulist, Plist):
    Lags = np.zeros(len(include))
    Lags[include] = incLags[:]
    ZTT = get_ZTT(Lags, O, gradZTT)
    mineigJac = np.zeros((1,len(incLags)))
    mineigJac[0,:] = get_ZTT_mineig_grad(ZTT, gradZTT)[include]
    return mineigJac

def check_spatialProj_Lags_validity(Lags, O, gradZTT):
    ZTT = get_ZTT(Lags, O, gradZTT)
    try:
        _ = la.cholesky(ZTT)
        return 1
    except la.LinAlgError:
        return -1

def check_spatialProj_incLags_validity(incLags, include, O, gradZTT):
    Lags = np.zeros(len(include), dtype=np.double)
    Lags[include] = incLags[:]
    return check_spatialProj_Lags_validity(Lags, O, gradZTT)

def Lags_normsqr(Lags):
    return np.sum(Lags*Lags), 2*Lags

def Lags_normsqr_Hess_np(Lags):
    return 2*np.eye(len(Lags))

def Sym(A):
    return np.real((A + A.T.conjugate())/2)

def Asym(A):
    return np.real((A - A.T.conjugate())/2j)