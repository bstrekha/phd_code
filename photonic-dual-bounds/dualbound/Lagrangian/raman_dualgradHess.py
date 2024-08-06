import numpy as np 
import scipy.linalg as la
from .spatialProjopt_vecs_numpy import get_ZTTcho_Tvec #, get_Tvec, get_ZTTcho_Tvec_gradTvec

def get_raman_dualgrad(Lags, grad, O_lin, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, fSlist, dualconst=0.0, include=None):
    if include is None:
        include = [True]*len(Lags)

    ZTT = O_quad.copy()
    ZTS_S = O_lin.copy()
    S_ZSS_S = 0 
    for i in range(len(Lags)):
        ZTT += Lags[i] * gradZTT[i]
        ZTS_S += Lags[i] * gradZTS_S[i]
        S_ZSS_S += Lags[i] * S_gradZSS_S[i]

    ZTTcho, T = get_ZTTcho_Tvec(ZTT, ZTS_S)
    dualval = dualconst + np.real(np.vdot(T, ZTT @ T)) + np.real(S_ZSS_S)
    if len(grad)>0:
        grad[:] = 0
    
        for i in range(len(Lags)):
            if include[i]:
                grad[i] += -np.real(np.vdot(T, gradZTT[i] @ T)) + 2*np.real(np.vdot(T, gradZTS_S[i])) + np.real(S_gradZSS_S[i])

        for _, fS in enumerate(fSlist):
            ZTTinv_fS = la.cho_solve(ZTTcho, fS)
            dualval += np.real(np.vdot(fS, ZTTinv_fS))
            for i in range(len(Lags)):
                if include[i]:
                    grad[i] += -np.real(np.vdot(ZTTinv_fS, gradZTT[i] @ ZTTinv_fS))

    else:
        for _, fS in enumerate(fSlist):
            ZTTinv_fS = la.cho_solve(ZTTcho, fS)
            dualval += np.real(np.vdot(fS, ZTTinv_fS))

    return dualval

def get_incgrad_from_grad(grad, include):
    return grad[include]

def get_incHess_from_Hess(Hess, include):
    return Hess[np.ix_(include,include)]

def get_inc_raman_dualgradHess(n_S, incLags, incgrad, incHess, include, O_lin, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, fSlist, dualconst=0.0, get_grad=True, get_Hess=True):
    Lagnum = len(include)
    Lags = np.zeros(Lagnum)
    Lags[include] = incLags[:]
    if get_Hess:
        print("don't do this!")
        exit()
        grad = np.zeros(Lagnum)
        Hess = np.zeros((Lagnum,Lagnum))
        dualval = get_raman_dualgradHess(n_S, Lags, grad,Hess, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include)
        incgrad[:] = get_incgrad_from_grad(grad, include) #[:] since we are modifying in place
        incHess[:,:] = get_incHess_from_Hess(Hess, include)
    elif get_grad:
        grad = np.zeros(Lagnum)
        dualval = get_raman_dualgrad(Lags, grad,         O_lin, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, fSlist, dualconst=dualconst, include=include)
        incgrad[:] = get_incgrad_from_grad(grad, include)
    else:
        dualval = get_raman_dualgrad(Lags, np.array([]), O_lin, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, fSlist, dualconst=dualconst, include=include)
    return dualval
