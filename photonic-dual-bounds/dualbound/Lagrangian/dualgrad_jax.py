import numpy as np 
import scipy.linalg as la
from .spatialProjopt_vecs_numpy import get_ZTTcho_Tvec #, get_Tvec, get_ZTTcho_Tvec_gradTvec
from .zops_utils import zops as zops
from .zops_utils import zops_jax as jzops
import jax.numpy as jnp
from jax import jit, vmap
import jax.scipy.linalg as jla

def get_dual(Lags, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, fSlist, dualconst=0.0, include=None):
    if include is None:
        include = [True]*len(Lags)

    ZTT = jzops.get_ZTT(Lags, O_quad, gradZTT)
    ZTS_S = jzops.get_ZTS_S(Lags, O_lin_S, gradZTS_S)
    S_ZSS_S = jzops.get_S_ZSS_S(Lags, S_gradZSS_S)

    ZTTcho, T = jzops.get_ZTTcho_Tvec(ZTT, ZTS_S)
    quadratic_part = jnp.real(np.vdot(T, ZTT @ T))
    constant_part = jnp.real(S_ZSS_S)
    dualval = dualconst + quadratic_part + constant_part

    vmap(lambda i : la.cho_solve(ZTTcho, fSlist[i]))(jnp.arange(fSlist.shape[0]))
    # for _, fS in enumerate(fSlist):
        # ZTTinv_fS = la.cho_solve(ZTTcho, fS)
        # dualval += jnp.real(jnp.vdot(fS, ZTTinv_fS))

    return dualval

def get_dualgrad(Lags, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, fSlist, dualconst=0.0, include=None):
    if include is None:
        include = [True]*len(Lags)

    ZTT = zops.get_ZTT(Lags, O_quad, gradZTT)
    ZTS_S = zops.get_ZTS_S(Lags, O_lin_S, gradZTS_S)
    S_ZSS_S = zops.get_S_ZSS_S(Lags, S_gradZSS_S)

    ZTTcho, T = zops.get_ZTTcho_Tvec(ZTT, ZTS_S)
    quadratic_part = np.real(np.vdot(T, ZTT @ T))
    constant_part = np.real(S_ZSS_S)
    dualval = dualconst + quadratic_part + constant_part

    gradfunc = np.vectorize(lambda i : -np.real(np.vdot(T, gradZTT[i] @ T)) 
                            + 2*np.real(np.vdot(T, gradZTS_S[i])) + np.real(S_gradZSS_S[i]))
    grad = np.where(include, gradfunc(np.arange(len(Lags))), 0)

    for _, fS in enumerate(fSlist):
        ZTTinv_fS = la.cho_solve(ZTTcho, fS)
        dualval += np.real(np.vdot(fS, ZTTinv_fS))
        fs_gradfunc = np.vectorize(lambda i : -np.real(np.vdot(ZTTinv_fS, gradZTT[i] @ ZTTinv_fS)))
        grad += np.where(include, fs_gradfunc(np.arange(len(Lags))), 0)

    return dualval, grad

def get_dualgradhess():
    pass 

############### 

# def get_dualgrad(Lags, grad, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, fSlist, dualconst=0.0, include=None):
#     if include is None:
#         include = [True]*len(Lags)

#     ZTT = zops.get_ZTT(Lags, O_quad, gradZTT)
#     ZTS_S = zops.get_ZTS_S(Lags, O_lin_S, gradZTS_S)
#     S_ZSS_S = zops.get_S_ZSS_S(Lags, S_gradZSS_S)

#     ZTTcho, T = zops.get_ZTTcho_Tvec(ZTT, ZTS_S)
#     quadratic_part = np.real(np.vdot(T, ZTT @ T))
#     constant_part = np.real(S_ZSS_S)
#     dualval = dualconst + quadratic_part + constant_part
#     if len(grad) > 0:
#         gradfunc = np.vectorize(lambda i : -np.real(np.vdot(T, gradZTT[i] @ T)) 
#                                 + 2*np.real(np.vdot(T, gradZTS_S[i])) + np.real(S_gradZSS_S[i]))
#         grad[:] = np.where(include, gradfunc(np.arange(len(Lags))), 0)
    
#         for _, fS in enumerate(fSlist):
#             ZTTinv_fS = la.cho_solve(ZTTcho, fS)
#             dualval += np.real(np.vdot(fS, ZTTinv_fS))
#             fs_gradfunc = np.vectorize(lambda i : -np.real(np.vdot(ZTTinv_fS, gradZTT[i] @ ZTTinv_fS)))
#             grad[:] += np.where(include, fs_gradfunc(np.arange(len(Lags))), 0)
            
#     else:
#         for _, fS in enumerate(fSlist):
#             ZTTinv_fS = la.cho_solve(ZTTcho, fS)
#             dualval += np.real(np.vdot(fS, ZTTinv_fS))

#     return dualval

# def get_dualgradHess(Lags, grad, Hess, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, fSlist, dualconst=0.0, include=None):
#     if include is None:
#         include = [True]*len(Lags)
        
#     ZTT = zops.get_ZTT(Lags, O_quad, gradZTT)
#     ZTS_S = zops.get_ZTS_S(Lags, O_lin_S, gradZTS_S)
#     S_ZSS_S = zops.get_S_ZSS_S(Lags, S_gradZSS_S)
    
#     grad[:] = 0
#     Hess[:,:] = 0
#     ZTTcho, T, gradT = zops.get_ZTTcho_Tvec_gradTvec(ZTT, gradZTT, ZTS_S, gradZTS_S) # Here, constant part? 

#     dualval = dualconst + np.real(np.vdot(T, ZTT @ T)) + np.real(S_ZSS_S)
#     gradfunc = np.vectorize(lambda i : -np.real(np.vdot(T, gradZTT[i] @ T)) 
#                                 + 2*np.real(np.vdot(T, gradZTS_S[i])) + np.real(S_gradZSS_S[i]))
    
#     grad[:] = np.where(include, gradfunc(np.arange(len(Lags))), 0)

#     for i in range(len(Lags)):
#         if not include[i]:
#             continue
#         for j in range(i,len(Lags)):
#             if not include[j]:
#                 continue
#             Hess[i,j] += 2*np.real(np.vdot(gradT[i],-gradZTT[j] @ T + gradZTS_S[j]))
#             if i != j:
#                 Hess[j,i] = Hess[i,j]

#     for _, fS in enumerate(fSlist):
#         ZTTinv_fS = la.cho_solve(ZTTcho, fS)
#         dualval += np.real(np.vdot(fS, ZTTinv_fS))
#         ZTTinv_gradZTT_ZTTinv_fS = []
#         for i in range(len(Lags)):
#             if include[i]:
#                 gradZTT_ZTTinv_fS = gradZTT[i] @ ZTTinv_fS
#                 grad[i] += -np.real(np.vdot(ZTTinv_fS, gradZTT_ZTTinv_fS))
#                 ZTTinv_gradZTT_ZTTinv_fS.append(la.cho_solve(ZTTcho, gradZTT_ZTTinv_fS))
#             else:
#                 ZTTinv_gradZTT_ZTTinv_fS.append(None)

#         for i in range(len(Lags)):
#             if not include[i]:
#                 continue
#             for j in range(i,len(Lags)):
#                 if not include[j]:
#                     continue
#                 Hess[i,j] += 2*np.real(np.vdot(ZTTinv_fS, gradZTT[i] @ ZTTinv_gradZTT_ZTTinv_fS[j]))
#                 if i!=j:
#                     Hess[j,i] = Hess[i,j]
                
#     return dualval

# def get_Lags_from_incLags(incLags, include):
#     Lagnum = len(include)
#     Lags = np.zeros(Lagnum)
#     Lags[include] = incLags
#     return Lags

# def get_incgrad_from_grad(grad, include):
#     return grad[include]

# def get_incHess_from_Hess(Hess, include):
#     return Hess[np.ix_(include,include)]

# def get_dual_and_derivatives(incLags, incgrad, incHess, include, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, fSlist, dualconst=0.0, get_grad=True, get_Hess=True):
#     Lagnum = len(include)
#     Lags = np.zeros(Lagnum)
#     Lags[include] = incLags[:]

#     if get_Hess:
#         grad = np.zeros(Lagnum)
#         Hess = np.zeros((Lagnum,Lagnum))
#         dualval = get_dualgradHess(Lags, grad, Hess, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, fSlist, dualconst=dualconst, include=include)
#         incgrad[:] = get_incgrad_from_grad(grad, include) # [:] since we are modifying in place
#         incHess[:,:] = get_incHess_from_Hess(Hess, include)
#     elif get_grad:
#         grad = np.zeros(Lagnum)
#         dualval = get_dualgrad(Lags, grad, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S,
#                                      fSlist, dualconst=dualconst, include=include)
#         incgrad[:] = grad[include]
#     else:
#         dualval = get_dualgrad(Lags, np.array([]), O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S,
#                                      fSlist, dualconst=dualconst, include=include)
   
#     return dualval
