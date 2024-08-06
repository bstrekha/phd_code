import numpy as np 
import scipy.linalg as la
import scipy.sparse as sp
from .zops_utils import zops_sparse_all as zsp
from .zops_utils import zops
import sksparse.cholmod as chol

def dual_debugtool(Lags, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, dualconst, chofac, extradebug=None):
    from contextlib import suppress
    np.set_printoptions(linewidth=200)

    ZTT = zsp.get_ZTT(Lags, O_quad, gradZTT)
    ZTT_noquad = zsp.get_ZTT(Lags, sp.csc_array(np.zeros(O_quad.shape, dtype=complex)), gradZTT)
    ZTS_S = zsp.get_ZTS_S(Lags, O_lin_S, gradZTS_S)
    S_ZSS_S = zsp.get_S_ZSS_S(Lags, S_gradZSS_S)

    ZTTcho, T = zsp.get_ZTTcho_Tvec(ZTT, ZTS_S, chofac)
    quadratic_part = np.real(np.vdot(T, ZTT @ T))
    O_quad_part = np.real(np.vdot(T, O_quad @ T))
    Lag_quadratic_part = quadratic_part - O_quad_part
    assert np.isclose(Lag_quadratic_part, np.real(np.vdot(T, ZTT_noquad @ T)))

    constant_part = np.real(S_ZSS_S)
    dualval = dualconst + quadratic_part + constant_part

    partial_quadratic_lags = np.zeros(len(Lags))
    partial_constant = np.zeros(len(Lags))
    for idx, lag in enumerate(Lags):
        partial_lags = np.zeros(len(Lags))
        partial_lags[idx] = lag
        partial_ZTT = zsp.get_ZTT(partial_lags, sp.csc_array(np.zeros(O_quad.shape, dtype=complex)), gradZTT)
        partial_S_ZSS_S = zsp.get_S_ZSS_S(partial_lags, S_gradZSS_S)

        partial_quadratic_lags[idx] = np.real(np.vdot(T, partial_ZTT @ T))
        partial_constant[idx] = np.real(partial_S_ZSS_S)

    assert np.isclose(np.sum(partial_quadratic_lags), Lag_quadratic_part)

    print(f"Lags: {Lags}")
    print("dualval:", dualval)
    print("dualconst:", dualconst)
    with suppress(ZeroDivisionError): print("dualval/dualconst:", dualval/dualconst)
    print("constant_part:", constant_part)
    print("Lag_quadratic_part:", Lag_quadratic_part)
    print("O_quad_part:", O_quad_part)
    print("partial_quadratic_lags:", partial_quadratic_lags)
    if not extradebug is None:
        Ginv = extradebug[0]
        print("T:", Ginv @ T)
    return dualval, dualconst, constant_part, Lag_quadratic_part, O_quad_part, partial_quadratic_lags, partial_constant, T

def get_dualgrad(Lags, grad, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, fSlist, chofac=None, dualconst=0.0, include=None, 
                 extra_gradZTS_S=None, extra_lin_lags=None, extra_gradZTT=None, extra_quad_lags=None):
    if include is None:
        include = [True]*len(Lags)

    ZTT = zsp.get_ZTT(Lags, O_quad, gradZTT)
    if not extra_quad_lags is None: ZTT += zsp.get_ZTT(extra_quad_lags, sp.csc_array(np.zeros(O_quad.shape, dtype=complex)), extra_gradZTT)
    ZTS_S = zops.get_ZTS_S(Lags, O_lin_S, gradZTS_S)
    if not extra_lin_lags is None: ZTS_S += zops.get_ZTS_S(extra_lin_lags, np.zeros(O_lin_S.shape), extra_gradZTS_S)
    S_ZSS_S = zops.get_S_ZSS_S(Lags, S_gradZSS_S)

    ZTTcho, T = zsp.get_ZTTcho_Tvec(ZTT, ZTS_S, chofac=chofac)
    ZTTfScho = ZTTcho

    quadratic_part = np.real(np.vdot(T, ZTT @ T))
    constant_part = np.real(S_ZSS_S)
    dualval = dualconst + quadratic_part + constant_part

    if len(grad)>0:
        grad[:] = 0
        for i in range(len(Lags)):
            if include[i]:
                grad[i] += -np.real(np.vdot(T, gradZTT[i].dot(T))) + 2*np.real(np.vdot(T, gradZTS_S[i]))

        for _, fS in enumerate(fSlist):
            ZTTfSinv_fS = ZTTfScho.solve_A(fS)
            dualval += np.real(np.vdot(fS, ZTTfSinv_fS))
            for i in range(len(Lags)):
                if include[i]:
                    grad[i] += -np.real(np.vdot(ZTTfSinv_fS, gradZTT[i].dot(ZTTfSinv_fS)))

    else:
        for _, fS in enumerate(fSlist):
            ZTTfSinv_fS = ZTTfScho.solve_A(fS)
            dualval += np.real(np.vdot(fS, ZTTfSinv_fS))

    return dualval


def get_dualgradHess(Lags, grad, Hess, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, fSlist, chofac=None, dualconst=0.0, include=None,
                     extra_gradZTS_S=None, extra_lin_lags=None):
    if include is None:
        include = [True]*len(Lags)
        
    ZTT = zsp.get_ZTT(Lags, O_quad, gradZTT)
    ZTS_S = zops.get_ZTS_S(Lags, O_lin_S, gradZTS_S)
    if not extra_lin_lags is None: ZTS_S += zops.get_ZTS_S(extra_lin_lags, np.zeros(O_lin_S.shape), extra_gradZTS_S)
    S_ZSS_S = zops.get_S_ZSS_S(Lags, S_gradZSS_S)
    grad[:] = 0
    Hess[:,:] = 0

    ZTTcho, T, gradT = zsp.get_ZTTcho_Tvec_gradTvec(ZTT, gradZTT, ZTS_S, gradZTS_S, chofac=chofac)
    ZTTfScho = ZTTcho
    
    dualval = dualconst + np.real(np.vdot(T, ZTT @ T)) + np.real(S_ZSS_S)
    grad[:] = 0
    Hess[:,:] = 0
    
    for i in range(len(Lags)):
        if include[i]:
            grad[i] += -np.real(np.vdot(T, gradZTT[i] @ T)) + 2*np.real(np.vdot(T, gradZTS_S[i]))
            
    for i in range(len(Lags)):
        if not include[i]:
            continue
        for j in range(i,len(Lags)):
            if not include[j]:
                continue
            Hess[i,j] += 2*np.real(np.vdot(gradT[i],-gradZTT[j] @ T + gradZTS_S[j]))
            if i!=j:
                Hess[j,i] = Hess[i,j]

    for _, fS in enumerate(fSlist):
        ZTTfSinv_fS = ZTTfScho.solve_A(fS)
        dualval += np.real(np.vdot(fS, ZTTfSinv_fS))
        ZTTfSinv_gradZTT_ZTTfSinv_fS = []
        for i in range(len(Lags)):
            if include[i]:
                gradZTT_ZTTfSinv_fS = gradZTT[i] @ ZTTfSinv_fS
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
                Hess[i,j] += 2*np.real(np.vdot(ZTTfSinv_fS, gradZTT[i] @ ZTTfSinv_gradZTT_ZTTfSinv_fS[j]))
                if i!=j:
                    Hess[j,i] = Hess[i,j]
                
    return dualval

def get_Lags_from_incLags(incLags, include):
    Lagnum = len(include)
    Lags = np.zeros(Lagnum)
    Lags[include] = incLags
    return Lags

def get_incgrad_from_grad(grad, include):
    return grad[include]

def get_incHess_from_Hess(Hess, include):
    return Hess[np.ix_(include,include)]

def get_dual_and_derivatives(incLags, incgrad, incHess, include, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, fSlist, chofac=None, dualconst=0.0, 
                             get_grad=True, get_Hess=True, extra_gradZTS_S=None, extra_lin_lags=None, extra_gradZTT=None, extra_quad_lags=None):
    Lagnum = len(include)
    Lags = np.zeros(Lagnum)
    Lags[include] = incLags[:]

    if get_Hess:
        grad = np.zeros(Lagnum)
        Hess = np.zeros((Lagnum,Lagnum))
        dualval = get_dualgradHess(Lags, grad, Hess, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, fSlist, chofac=chofac,
                                   dualconst=dualconst, include=include, extra_gradZTS_S=extra_gradZTS_S, extra_lin_lags=extra_lin_lags)
        incgrad[:] = get_incgrad_from_grad(grad, include) # [:] since we are modifying in place
        incHess[:,:] = get_incHess_from_Hess(Hess, include)
    elif get_grad:
        grad = np.zeros(Lagnum)
        dualval = get_dualgrad(Lags, grad, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, fSlist, chofac=chofac,
                               dualconst=dualconst, include=include, extra_gradZTS_S=extra_gradZTS_S, extra_lin_lags=extra_lin_lags,
                               extra_gradZTT=extra_gradZTT, extra_quad_lags=extra_quad_lags)
        incgrad[:] = grad[include]
    else:
        dualval = get_dualgrad(Lags, np.array([]), O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, fSlist, chofac=chofac,
                               dualconst=dualconst, include=include, extra_gradZTS_S=extra_gradZTS_S, extra_lin_lags=extra_lin_lags,
                               extra_gradZTT=extra_gradZTT, extra_quad_lags=extra_quad_lags)
   
    return dualval
