import numpy as np
import scipy.linalg as la
import scipy.optimize as sopt

def get_dR_and_dI(chiinvPlist, PG, nmat, N):
    dR = np.zeros((nmat*N, nmat*N), dtype=complex)
    dI = np.zeros((nmat*N, nmat*N), dtype=complex)
    for m in range(nmat):
        for mp in range(nmat):
            mpoint = m*N
            mppoint = mp*N
            if m == mp:
                dR[mpoint:mpoint+N, mpoint:mpoint+N] = chiinvPlist[m, :, :].conjugate().T - PG.conjugate().T
                dI[mpoint:mpoint+N, mpoint:mpoint+N] = chiinvPlist[m, :, :].conjugate().T - PG.conjugate().T
            else:
                dR[mpoint:mpoint+N, mppoint:mppoint+N] = - PG.conjugate().T
                dI[mpoint:mpoint+N, mppoint:mppoint+N] = - PG.conjugate().T
    return dR/2, dI/(2j)

def get_dS_and_dA(Pj, m1, m2, nmat, N):
    dS = np.zeros((nmat*N, nmat*N), dtype=complex)
    dA = np.zeros((nmat*N, nmat*N), dtype=complex)

    m1point, nm1point = m1*N, (m1+1)*N # n for next
    m2point, nm2point = m2*N, (m2+1)*N
    dS[m1point:nm1point, m2point:nm2point] = Pj/2
    dS[m2point:nm2point, m1point:nm1point] = Pj/2
    dA[m1point:nm1point, m2point:nm2point] = Pj/(2j)
    dA[m2point:nm2point, m1point:nm1point] = -Pj/(2j)

    return dS, dA

def get_gradZTT(chiinvPlist, PGlist, Plist, nmat, nsource, npixels, N):
    """
    Evaluates dZTT/d\lambda.

    Parameters
    ----------
    chiinvPlist : numpy array
        list of chiinvP matrices
    PGlist : numpy array
        list of PG matrices
    Plist : numpy array
        list of P matrices
    nmat : integer
        number of materials
    nsource : integer
        number of sources
    npixels : integer
        number of projection constraints (lengdth of Plist)
    N : integer
        length of optimization vectors
    """

    gradZTT_1 = np.zeros((npixels, nsource, nsource, 2, nsource*nmat*N, nsource*nmat*N), dtype=complex) # First part of gradient, see SM of multimat publication
    gradZTT_2 = np.zeros((npixels, int(nsource*(nsource+1)/2), int(nmat*(nmat-1)/2), 2, nsource*nmat*N, nsource*nmat*N), dtype=complex)

    # gradZTT_1 part (R and I matrices)
    for j in range(npixels):
        for k1 in range(nsource):
            for k2 in range(nsource):
                # In each part of this loop, we compute the derivative in ZTT with respect to lambda^j,k1,k2
                # dR_j_k1_k2, dI_j_k1_k2 = get_dR_and_dI(chiinvPlist[k2, :, j, :, :], PGlist[k2, j], nmat, N) # Conjugate theory
                dR_j_k1_k2, dI_j_k1_k2 = get_dR_and_dI(chiinvPlist[k1, :, j, :, :], PGlist[k1, j, :, :], nmat, N) # As written in overleaf

                # This includes the diagonal elements, since adding the conjugate gives us sym or asym as we need
                k1_pt, nk1_pt = k1*nmat*N, (k1+1)*nmat*N
                k2_pt, nk2_pt = k2*nmat*N, (k2+1)*nmat*N
                gradZTT_1[j, k1, k2, 0, k1_pt:nk1_pt, k2_pt:nk2_pt] += dR_j_k1_k2
                gradZTT_1[j, k1, k2, 0, k2_pt:nk2_pt, k1_pt:nk1_pt] += dR_j_k1_k2.conjugate().T
                gradZTT_1[j, k1, k2, 1, k1_pt:nk1_pt, k2_pt:nk2_pt] += dI_j_k1_k2
                gradZTT_1[j, k1, k2, 1, k2_pt:nk2_pt, k1_pt:nk1_pt] += dI_j_k1_k2.conjugate().T

    # gradZTT_2 part (S and A orthogonality matrices)
    for j in range(npixels):
        kcounter = 0
        for k1 in range(nsource):
            for k2 in range(k1, nsource):
                mcounter = 0
                k1_pt, nk1_pt = k1*nmat*N, (k1+1)*nmat*N
                k2_pt, nk2_pt = k2*nmat*N, (k2+1)*nmat*N
                for m1 in range(nmat):
                    for m2 in range(m1+1, nmat):
                        dS_j_k1_k2, dA_j_k1_k2 = get_dS_and_dA(Plist[j], m1, m2, nmat, N)
                        gradZTT_2[j, kcounter, mcounter, 0, k1_pt:nk1_pt, k2_pt:nk2_pt] += dS_j_k1_k2
                        gradZTT_2[j, kcounter, mcounter, 0, k2_pt:nk2_pt, k1_pt:nk1_pt] += dS_j_k1_k2.conjugate().T # it is the same as without T or conjugate
                        gradZTT_2[j, kcounter, mcounter, 1, k1_pt:nk1_pt, k2_pt:nk2_pt] += dA_j_k1_k2
                        gradZTT_2[j, kcounter, mcounter, 1, k2_pt:nk2_pt, k1_pt:nk1_pt] += dA_j_k1_k2.conjugate().T
                        mcounter += 1
                kcounter += 1

    # We reshape the matrices and concatenate them into a reasonable order of lagrange multipliers
    gradZTT_1 = np.reshape(gradZTT_1, (npixels*nsource*nsource*2, nsource*nmat*N, nsource*nmat*N), order='C')
    gradZTT_2 = np.reshape(gradZTT_2, (npixels*int(nsource*(nsource+1)/2)*int(nmat*(nmat-1)/2)*2, nsource*nmat*N, nsource*nmat*N), order='C')
    gradZTT = np.append(gradZTT_1, gradZTT_2, axis=0)
    return gradZTT

def get_gradZTS_S_Sym_Asym_k1_k2(k1, k2, S, P, nsource, nmat):
    N = P.shape[0]
    grad_ZTS_Sym_S  = np.zeros((nsource, nmat*N), dtype=np.complex128)
    grad_ZTS_Asym_S = np.zeros((nsource, nmat*N), dtype=np.complex128)
    for m in range(nmat):
        grad_ZTS_Sym_S[k1,  m*N:(m+1)*N] += (P/2)     @ (S[N*k2:N*(k2+1)])
        grad_ZTS_Asym_S[k1, m*N:(m+1)*N] += (-P/(2j)) @ (S[N*k2:N*(k2+1)])

    grad_ZTS_Sym_S = np.reshape(grad_ZTS_Sym_S, (nsource*nmat*N))
    grad_ZTS_Asym_S = np.reshape(grad_ZTS_Asym_S, (nsource*nmat*N))
    return grad_ZTS_Sym_S, grad_ZTS_Asym_S

def get_gradZTS_S(S, Plist, nmat, nsource, npixels, N):
    gradZTS_S = np.zeros((npixels, nsource, nsource, 2, nmat*nsource*N), dtype=np.complex128)
    for j in range(npixels):
        for k1 in range(nsource):
            for k2 in range(nsource):
                grad_ZTS_Sym_S_k1_k2, grad_ZTS_Asym_S_k1_k2 = get_gradZTS_S_Sym_Asym_k1_k2(k2, k1, S, Plist[j], nsource, nmat) # This one is correct (just check gradZTS)
                gradZTS_S[j, k1, k2, 0, :] += grad_ZTS_Sym_S_k1_k2
                gradZTS_S[j, k1, k2, 1, :] += grad_ZTS_Asym_S_k1_k2

    gradZTS_S = np.reshape(gradZTS_S, (npixels*nsource*nsource*2, nsource*nmat*N)) #, order='C')
    otherlags = np.zeros((npixels*int(nsource*(nsource+1)/2)*int(nmat*(nmat-1)/2)*2, nsource*nmat*N), dtype=np.complex128) # The derivative with the rest of the lagrange multipliers is 0
    gradZTS_S = np.append(gradZTS_S, otherlags, axis=0)
    return gradZTS_S

def include_helper(npixels, nsource, nmat, powerCons=True, ortho=True):
    include1 = np.zeros((npixels, nsource, nsource, 2), dtype=bool)
    include2 = np.zeros((npixels, int(nsource*(nsource+1)/2), int(nmat*(nmat-1)/2), 2), dtype=bool)

    if powerCons:
        for j in range(npixels):
            for k1 in range(nsource):
                for k2 in range(nsource):
                    include1[j, k1, k2, 0] = True
                    include1[j, k1, k2, 1] = True

    if ortho:
        for j in range(npixels):
            kcounter = 0
            for k1 in range(nsource):
                for k2 in range(k1, nsource):
                    mcounter = 0
                    for m1 in range(nmat):
                        for m2 in range(m1+1, nmat):
                            include2[j, kcounter, mcounter, 0] = True
                            include2[j, kcounter, mcounter, 1] = True
                            mcounter += 1
                    kcounter += 1

    include1 = np.reshape(include1, (npixels*nsource*nsource*2))
    include2 = np.reshape(include2, (npixels*int(nsource*(nsource+1)/2)*int(nmat*(nmat-1)/2)*2))
    include = np.append(include1, include2, axis=0)
    return include

def get_ZTT(n_S, Lags, O, gradZTT):
    ZTT = O.copy()
    for i in range(len(Lags)):
        ZTT += Lags[i] * gradZTT[i]
    return ZTT

def check_spatialProj_Lags_validity(n_S, Lags, O, gradZTT):
    ZTT = get_ZTT(n_S, Lags, O, gradZTT)
    try:
        _ = la.cholesky(ZTT)
        return 1
    except la.LinAlgError:
        return -1

def check_spatialProj_incLags_validity(n_S, incLags, include, O, gradZTT):
    Lags = np.zeros(len(include), dtype=np.double)
    Lags[include] = incLags[:]
    return check_spatialProj_Lags_validity(n_S, Lags, O, gradZTT)

def get_ZTT_mineig(n_S, Lags, O, gradZTT, eigvals_only=False):
    ZTT = get_ZTT(n_S, Lags, O, gradZTT)
    if eigvals_only:
        eigw = la.eigvalsh(ZTT)
        return eigw[0]
    else:
        eigw, eigv = la.eigh(ZTT)
        return eigw[0], eigv[:,0]

def get_ZTT_mineig_grad(ZTT, gradZTT):
    eigw, eigv = la.eigh(ZTT)
    eiggrad = np.zeros(len(gradZTT))
    for i in range(len(eiggrad)):
        eiggrad[i] = np.real(np.vdot(eigv[:,0], gradZTT[i] @ eigv[:,0]))
    return eiggrad

def get_inc_ZTT_mineig(n_S, incLags, include, O, gradZTT, eigvals_only=False):
    Lags = np.zeros(len(include))
    Lags[include] = incLags[:]
    return get_ZTT_mineig(n_S, Lags, O, gradZTT, eigvals_only=eigvals_only)

def get_ZTT_gradmineig(n_S, incLags, include, O, gradZTT): #Ulist, Plist):
    Lags = np.zeros(len(include))
    Lags[include] = incLags[:]
    ZTT = get_ZTT(n_S, Lags, O, gradZTT)
    mineigJac = np.zeros((1,len(incLags)))
    mineigJac[0,:] = get_ZTT_mineig_grad(ZTT, gradZTT)[include]
    return mineigJac

# def get_ZTT_mineig(n_S, incLags, include, O, Ulist, Plist):
#     Lags = np.zeros(len(include))
#     Lags[include] = incLags
#     ZTT = Z_TT(n_S, Lags, O, Ulist, Plist)
#     eigw = la.eigvalsh(ZTT)
#     if eigw[0]>=0:
#         global feasiblept
#         feasiblept = incLags
#         raise ValueError('found a feasible point')
#     return eigw[0]


def Lags_normsqr(Lags):
    return np.sum(Lags*Lags), 2*Lags

def Lags_normsqr_Hess_np(Lags):
    return 2*np.eye(len(Lags))

def spatialProjopt_find_feasiblept(n_S, Lagnum, include, O, gradZTT, maxiter):
    incLagnum = np.sum(include)
    initincLags = np.random.rand(incLagnum)

    mineigincfunc = lambda incL: get_inc_ZTT_mineig(n_S, incL, include, O, gradZTT, eigvals_only=True)
    Jacmineigincfunc = lambda incL: get_ZTT_gradmineig(n_S, incL, include, O, gradZTT)

    tolcstrt = 1e-4
    cstrt = sopt.NonlinearConstraint(mineigincfunc, tolcstrt, np.inf, jac=Jacmineigincfunc, keep_feasible=False)

    # np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
    lb = -np.inf*np.ones(incLagnum)
    ub = np.inf*np.ones(incLagnum)
    bnds = sopt.Bounds(lb,ub)
    try:
        res = sopt.minimize(Lags_normsqr, initincLags, method='trust-constr', jac=True, hess=Lags_normsqr_Hess_np,
                            bounds=bnds, constraints=cstrt, options={'verbose':2,'maxiter':maxiter})
    except ValueError:
        global feasiblept
        Lags = np.zeros(Lagnum)
        Lags[include] = feasiblept
        return Lags

    Lags = np.zeros(Lagnum)
    Lags[include] = res.x
    Lags[1] = np.abs(Lags[1]) + 0.01
    return Lags
