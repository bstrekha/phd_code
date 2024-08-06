import numpy as np 
import scipy.sparse as sp
from . import zops
import scipy.linalg as la
import sksparse.cholmod as chol
import scipy.sparse.linalg as spla

def density(A):
    return A.nnz / (A.shape[0]*A.shape[1])

def round(A, eps):
    A[A < eps] = 0
    return A 

def get_ZTT(Lags, O, gradZTT):
    ZTT = O.copy()
    for i in range(len(Lags)):
        ZTT += Lags[i] * gradZTT[i]
    return ZTT

def get_ZTS_S(Lags, O_lin_S, gradZTS_S):
    return zops.get_ZTS_S(Lags, O_lin_S, gradZTS_S)

def get_S_ZSS_S(Lags, S_gradZSS_S):
    return zops.get_S_ZSS_S(Lags, S_gradZSS_S)

def get_ZTTcho_Tvec(ZTT, ZTS_S, chofac=None):
    if chofac is None:
        ZTTcho = chol.cholesky(ZTT)
    else:
        ZTTcho = chofac.cholesky(ZTT)
    Tvec = ZTTcho.solve_A(ZTS_S)
    return ZTTcho, Tvec

def get_ZTTcho_Tvec_gradTvec(ZTT, gradZTT, ZTS_S, gradZTS_S, chofac=None):
    if chofac is None:
        ZTTcho = chol.cholesky(ZTT)
    else:
        ZTTcho = chofac.cholesky(ZTT)
    Tvec = ZTTcho.solve_A(ZTS_S)

    gradTvec = []
    for i in range(len(gradZTT)):
        gradTvec.append(ZTTcho.solve_A(-gradZTT[i] @ Tvec + gradZTS_S[i]))

    return ZTTcho, Tvec, gradTvec

def is_Hermitian(A):
    return np.allclose(A.todense(), A.todense().T.conjugate())

# def get_ZTT_mineig(Lags, O, gradZTT, eigvals_only=False, v0=None):
#     ZTT = get_ZTT(Lags, O, gradZTT)
#     if eigvals_only:
#         eigw = spla.eigsh(ZTT, k=1, which='SM', return_eigenvectors=False, v0=v0)
#         return eigw[0]
#     else:
#         eigw, eigv = spla.eigsh(ZTT, k=1, which='SM', return_eigenvectors=True, v0=v0)
#         return eigw[0], eigv[:,0]

def get_ZTT_mineig(Lags, O, gradZTT, eigvals_only=False, v0=None):
    ZTT = get_ZTT(Lags, O, gradZTT)
    if eigvals_only:
        try:
            eigw = spla.eigsh(ZTT, k=1, sigma=0.0, which='LM', return_eigenvectors=False, v0=v0)
        except BaseException as err:
            print('encountered error in sparse eigenvalue evaluation', err)
            eigw = la.eigvalsh(ZTT.todense())
        return eigw[0]
    else:
        try:
            eigw, eigv = spla.eigsh(ZTT, k=1, sigma=0.0, which='LM', return_eigenvectors=True, v0=v0)
        except BaseException as err:
            print('encountered error in sparse eigenvalue evaluation', err)
            eigw, eigv = la.eigh(ZTT.todense())
        return eigw[0], eigv[:,0]
    

def get_inc_ZTT_mineig(incLags, include, O, gradZTT, eigvals_only=False, v0=None):
    Lags = np.zeros(len(include))
    Lags[include] = incLags[:]
    return get_ZTT_mineig(Lags, O, gradZTT, eigvals_only=eigvals_only, v0=v0)

def check_spatialProj_Lags_validity(Lags, O, gradZTT, chofac=None, mineigtol=None):
    ZTT = get_ZTT(Lags, O, gradZTT)
    if not (mineigtol is None):
        ZTT -= mineigtol * sp.eye(ZTT.shape[0], format='csc')
    try:
        if chofac is None:
            ZTTcho = chol.cholesky(ZTT)
            tmp = ZTTcho.L() # necessary to attempt to access raw factor for checking matrix definiteness
        else:
            ZTTcho = chofac.cholesky(ZTT)
            tmp = ZTTcho.L() # see above
    except chol.CholmodNotPositiveDefiniteError:
        return False
    return True

def check_spatialProj_incLags_validity(incLags, include, O, gradZTT, chofac=None, mineigtol=None):
    Lags = np.zeros(len(include))
    Lags[include] = incLags[:]
    return check_spatialProj_Lags_validity(Lags, O, gradZTT, chofac=chofac, mineigtol=mineigtol)

# def Lags_normsqr(Lags):
#     return np.sum(Lags*Lags), 2*Lags

# def Lags_normsqr_Hess_np(Lags):
#     return 2*np.eye(len(Lags))

def Sym(A):
    return (A + A.T.conjugate())/2

def Asym(A):
    return (A - A.T.conjugate())/2j

def Cholesky_analyze_ZTT(O, gradZTT):
    Lags = np.random.rand(len(gradZTT))
    ZTT = get_ZTT(Lags, O, gradZTT)
    print('analyzing ZTT of format and shape', ZTT.format, ZTT.shape, 'and # of nonzero elements', ZTT.count_nonzero())
    return chol.analyze(ZTT)