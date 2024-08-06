import numpy as np
import scipy.linalg as la
import scipy.sparse as sp 

def get_dR_and_dI(P, chilist, Ginv_k, Ginv_kp, nmat, N):
    dR = np.zeros((nmat*N, nmat*N), dtype=complex)
    dI = np.zeros((nmat*N, nmat*N), dtype=complex)
    for m in range(nmat):
        for mp in range(nmat):
            mpoint = m*N
            mppoint = mp*N
            if m == mp:
                dR[mpoint:mpoint+N, mpoint:mpoint+N] = (Ginv_k.T.conj() @ (1/chilist[m].conjugate().T * P) @ Ginv_kp - P @ Ginv_kp ).toarray()
                dI[mpoint:mpoint+N, mpoint:mpoint+N] = (Ginv_k.T.conj() @ (1/chilist[m].conjugate().T * P) @ Ginv_kp - P @ Ginv_kp ).toarray()
            else:
                dR[mpoint:mpoint+N, mppoint:mppoint+N] = - (P @ Ginv_kp ).toarray()
                dI[mpoint:mpoint+N, mppoint:mppoint+N] = - (P @ Ginv_kp ).toarray()
    return dR/2, dI/(2j)

def get_dS_and_dA(P, m1, m2, nmat, N, Ginv_l, Ginv_r):
    dS = np.zeros((nmat*N, nmat*N), dtype=complex)
    dA = np.zeros((nmat*N, nmat*N), dtype=complex)

    m1point, nm1point = m1*N, (m1+1)*N # n for next
    m2point, nm2point = m2*N, (m2+1)*N
    dS[m1point:nm1point, m2point:nm2point] = (Ginv_l.T.conj() @ P/2 @ Ginv_r).toarray()
    dS[m2point:nm2point, m1point:nm1point] = dS[m1point:nm1point, m2point:nm2point].T.conjugate()
    dA[m1point:nm1point, m2point:nm2point] = (Ginv_l.T.conj() @ P/2j @ Ginv_r).toarray()
    dA[m2point:nm2point, m1point:nm1point] = dA[m1point:nm1point, m2point:nm2point].T.conjugate() 

    return dS, dA

def get_gradZTT(Plist, nmat, nsource, chilist, Ginvlist):
    # Notes: Ginvlist is a list of Ginv for each source. 
    # chilist is a list of chis of shape (nsource, nmat)
    num_regions = len(Plist)
    N = Plist[0].shape[1]
    gradZTT_1 = np.zeros((num_regions, nsource, nsource, 2), dtype=object)
    gradZTT_2 = np.zeros((num_regions, int(nsource*(nsource+1)/2), int(nmat*(nmat-1)/2), 2), dtype=object)

    for j in range(num_regions):
        P = Plist[j]
        kcounter = 0 
        for k1 in range(nsource):
            for k2 in range(nsource):
                # second index of chilist is m, we need k1 
                dR_j_k1_k2, dI_j_k1_k2 = get_dR_and_dI(P, chilist[k1, :], Ginvlist[k1], Ginvlist[k2], nmat, N) 
                k1_pt, nk1_pt = k1*nmat*N, (k1+1)*nmat*N
                k2_pt, nk2_pt = k2*nmat*N, (k2+1)*nmat*N
                temp1 = np.zeros((nsource*nmat*N, nsource*nmat*N), dtype=complex)
                temp2 = np.zeros((nsource*nmat*N, nsource*nmat*N), dtype=complex)

                temp1[k1_pt:nk1_pt, k2_pt:nk2_pt] += dR_j_k1_k2
                temp1[k2_pt:nk2_pt, k1_pt:nk1_pt] += dR_j_k1_k2.conjugate().T
                temp2[k1_pt:nk1_pt, k2_pt:nk2_pt] += dI_j_k1_k2
                temp2[k2_pt:nk2_pt, k1_pt:nk1_pt] += dI_j_k1_k2.conjugate().T

                gradZTT_1[j, k1, k2, 0] = sp.coo_array(temp1)
                gradZTT_1[j, k1, k2, 1] = sp.coo_array(temp2)

            # gradZTT_2 part (S and A orthogonality matrices)
            for k2 in range(k1, nsource):
                mcounter = 0
                k1_pt, nk1_pt = k1*nmat*N, (k1+1)*nmat*N
                k2_pt, nk2_pt = k2*nmat*N, (k2+1)*nmat*N
                Gkinv = Ginvlist[k1]
                Gkpinv = Ginvlist[k2] # prime 
                for m1 in range(nmat):
                    for m2 in range(m1+1, nmat):
                        dS_j_k1_k2, dA_j_k1_k2 = get_dS_and_dA(P, m1, m2, nmat, N, Gkinv, Gkpinv)
                        temp1 = np.zeros((nsource*nmat*N, nsource*nmat*N), dtype=complex)
                        temp2 = np.zeros((nsource*nmat*N, nsource*nmat*N), dtype=complex)

                        temp1[k1_pt:nk1_pt, k2_pt:nk2_pt] += dS_j_k1_k2
                        temp1[k2_pt:nk2_pt, k1_pt:nk1_pt] += dS_j_k1_k2.conjugate().T 
                        temp2[k1_pt:nk1_pt, k2_pt:nk2_pt] += dA_j_k1_k2
                        temp2[k2_pt:nk2_pt, k1_pt:nk1_pt] += dA_j_k1_k2.conjugate().T

                        gradZTT_2[j, kcounter, mcounter, 0] = sp.coo_array(temp1)
                        gradZTT_2[j, kcounter, mcounter, 1] = sp.coo_array(temp2)
                        mcounter += 1
                kcounter += 1

    gradZTT_1 = np.reshape(gradZTT_1, (num_regions*nsource*nsource*2))
    gradZTT_2 = np.reshape(gradZTT_2, (num_regions*int(nsource*(nsource+1)/2)*int(nmat*(nmat-1)/2)*2))
    gradZTT = np.append(gradZTT_1, gradZTT_2, axis=0)
    gradZTTlist = [] 
    for i in range(gradZTT.shape[0]):
        gradZTTlist.append(sp.coo_array(gradZTT[i]))

    return gradZTTlist


def get_gradZTS_S_Sym_Asym_k1_k2(k1, k2, Ginvk1, S, P, nsource, nmat):
    N = P.shape[0]
    grad_ZTS_Sym_S  = np.zeros((nsource, nmat*N), dtype=complex)
    grad_ZTS_Asym_S = np.zeros((nsource, nmat*N), dtype=complex)
    for m in range(nmat):
        grad_ZTS_Sym_S[k1,  m*N:(m+1)*N] += Ginvk1.T.conj() @ (P/2).T.conj() @ (S[N*k2:N*(k2+1)])
        grad_ZTS_Asym_S[k1, m*N:(m+1)*N] += Ginvk1.T.conj() @ (P/(2j)).T.conj() @ (S[N*k2:N*(k2+1)])

    grad_ZTS_Sym_S = np.reshape(grad_ZTS_Sym_S, (nsource*nmat*N))
    grad_ZTS_Asym_S = np.reshape(grad_ZTS_Asym_S, (nsource*nmat*N))
    return grad_ZTS_Sym_S, grad_ZTS_Asym_S

def get_gradZTS_S(S, Plist, Ginvlist, nmat, nsource):
    num_regions = len(Plist)
    N = Plist[0].shape[1]
    gradZTS_S = np.zeros((num_regions, nsource, nsource, 2, nmat*nsource*N), dtype=complex)
    for j in range(num_regions):
        for k1 in range(nsource):
            for k2 in range(nsource):
                grad_ZTS_Sym_S_k1_k2, grad_ZTS_Asym_S_k1_k2 = get_gradZTS_S_Sym_Asym_k1_k2(k2, k1, Ginvlist[k1], S, Plist[j], nsource, nmat) # This one is correct (just check gradZTS)
                gradZTS_S[j, k1, k2, 0, :] += grad_ZTS_Sym_S_k1_k2
                gradZTS_S[j, k1, k2, 1, :] += grad_ZTS_Asym_S_k1_k2

    gradZTS_S = np.reshape(gradZTS_S, (num_regions*nsource*nsource*2, nsource*nmat*N)) 
    otherlags = np.zeros((num_regions*int(nsource*(nsource+1)/2)*int(nmat*(nmat-1)/2)*2, nsource*nmat*N), dtype=complex) # The derivative with the rest of the lagrange multipliers is 0
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


# def get_gradZTT_real_mp1(Plist, nmat, nsource, chilist, Ginvlist):
#     # This only works for nmat=1 and nsource=1 for now. (cross source and mat constraints not implemented)
#     assert nmat == 1
#     assert nsource == 1

#     # Notes: Ginvlist is a list of Ginv for each source. 
#     # chilist is a list of chis of shape (nsource, nmat)
#     N = Plist[0].shape[1]
#     gradZTT_1 = np.zeros((nsource, nsource, 1), dtype=object)
#     P = Plist[0] # There is only one type of constraint for the single material problem 

#     for k1 in range(nsource):
#         for k2 in range(nsource):
#             # second index of chilist is m, we need k1 
#             dR_j_k1_k2, dI_j_k1_k2 = get_dR_and_dI(P, chilist[k1, :], Ginvlist[k1], Ginvlist[k2], nmat, N) 
#             k1_pt, nk1_pt = k1*nmat*N, (k1+1)*nmat*N
#             k2_pt, nk2_pt = k2*nmat*N, (k2+1)*nmat*N
#             temp1 = np.zeros((nsource*nmat*N, nsource*nmat*N), dtype=complex)

#             temp1[k1_pt:nk1_pt, k2_pt:nk2_pt] += dR_j_k1_k2
#             temp1[k2_pt:nk2_pt, k1_pt:nk1_pt] += dR_j_k1_k2.conjugate().T

#             gradZTT_1[k1, k2, 0] = sp.csr_array(temp1)

#     gradZTT_1 = np.reshape(gradZTT_1, (nsource*nsource*1))

#     return gradZTT_1

def get_gradZTT_real_mp1(Plist, nmat, nsource, chilist, Ginvlist):
    # This only works for nmat=1 and nsource=1 for now. (cross source and mat constraints not implemented)
    assert nmat == 1
    assert nsource == 1

    # Notes: Ginvlist is a list of Ginv for each source. 
    # chilist is a list of chis of shape (nsource, nmat)
    N = Plist[0].shape[1]
    gradZTT_1 = np.zeros((1), dtype=object)
    P = Plist[0] # There is only one type of constraint for the single material problem 
    Ginv = Ginvlist[0]
    GinvdagPdag = Ginv.conj().T @ P.conj().T
    chi = chilist[0, 0]
    UP = (Ginv.T.conj() @ (1/chi.conjugate().T * P) @ Ginv - P @ Ginv).toarray()

    gradZTT_1[0] = sp.csr_array((UP + UP.T.conjugate())/2)

    return gradZTT_1

def get_gradZTS_S_real_mp1(S, Plist, Ginvlist, nmat, nsource):
    assert nmat == 1
    assert nsource == 1

    N = Plist[0].shape[1]
    P = Plist[0]
    gradZTS_S = np.zeros((nsource, nsource, 1, nmat*nsource*N), dtype=complex)
    for k1 in range(nsource):
        for k2 in range(nsource):
            grad_ZTS_Sym_S_k1_k2, grad_ZTS_Asym_S_k1_k2 = get_gradZTS_S_Sym_Asym_k1_k2(k2, k1, Ginvlist[k1], S, P, nsource, nmat) # This one is correct (just check gradZTS)
            gradZTS_S[k1, k2, 0, :] += grad_ZTS_Sym_S_k1_k2

    gradZTS_S = np.reshape(gradZTS_S, (nsource*nsource*1, nsource*nmat*N)) 
    return gradZTS_S


# no longer takes U1! 
def get_new_vecs_heuristic(Ginv, chi, fSlist, N, cclasses, get_gradZTT, get_gradZTS_S, GT, ZTT, niters, S1):
    T = Ginv @ GT 
    T1 = T[0:N]
    U1conjT = T/chi - GT 
    # violation1 = -(U1.T.conj() @ T1).conj() * T1 + (S1).conj() * T1
    violation1 = -U1conjT * T1 + S1.conj() * T1
    
    for i in range(len(fSlist)): #contribution to violation from fake Source terms
        ZTTinvfS = la.solve(ZTT, fSlist[i])
        GinvZTTinvfS = Ginv @ ZTTinvfS
        violation1 -= (1.0/np.conj(chi)) * (np.conj(GinvZTTinvfS) * GinvZTTinvfS) - np.conj(ZTTinvfS) * GinvZTTinvfS

    violations = [violation1]
    Laggradfac_phase = np.zeros((cclasses, N), dtype=complex)
    for i, v in enumerate(violations):
        Laggradfac_phase[i, :] = -v / np.abs(v)

    eigw, eigv = la.eigh(ZTT)
    eigw = eigw[0]
    eigv = eigv[:,0]
    x1 = Ginv @ eigv

    # mineigfac = outer_sparse_Pstruct(np.conj(Ginveigv), Ginveigv, Pstruct)/np.conj(chi) - outer_sparse_Pstruct(np.conj(eigv), Ginveigv, Pstruct)
    
    mineig_phase = np.zeros((cclasses, N), dtype=complex)
    # minfac1 = (U1.T.conj() @ x1).conj() * x1
    minfac1 = np.conj(x1) * x1 / np.conj(chi) - np.conj(eigv) * x1
    minfac = [minfac1]

    for i, m in enumerate(minfac):
        mineig_phase[i, :] = m / np.abs(m)

    new_vecs = np.zeros((cclasses, N), dtype=complex)
    for i in range(cclasses):
        new_vecs[i, :] = np.conj(Laggradfac_phase[i, :] + mineig_phase[i, :])
        new_vecs[i, :] *= np.abs(np.real(-new_vecs[i, :] * violations[i])) 
        # new_vecs[i, :][np.isnan(new_vecs[i, :])] = 0.0

    return new_vecs
