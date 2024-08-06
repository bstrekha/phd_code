import numpy as np
import scipy.linalg as la 
import matplotlib.pyplot as plt 
import sys
sys.path.append('../../../')
from dualbound.Lagrangian.zops_utils import zops_sparse as zsp
import scipy.sparse as sp 

def get_gradZTT(Plist, G1inv, G2inv, chi1, chi2, alpha, Cc, extra):
    # Construct each matrix with dok and place it in an object array (multidimensional for convenience)
    # Then, flatten the object array 
    # Then, convert it to a list (for use with numba later) and convert each dok matrix to a CSR array
    num_regions = len(Plist)
    N = Plist[0].shape[1]
    gradZTT = np.zeros((num_regions, 6, 2), dtype=object)
    gradZTT2 = np.zeros((num_regions, 1), dtype=object)
    if extra: gradZTT3 = np.zeros((1), dtype=object)

    for j in range(num_regions):
        P = Plist[j]
        U11 = (G1inv.T.conjugate() @ (1/chi1.conjugate() * sp.eye(N)) - sp.eye(N)) @ Plist[j] @ G1inv 
        U22 = (G2inv.T.conjugate() @ (1/chi2.conjugate() * sp.eye(N)) - sp.eye(N)) @ Plist[j] @ G2inv
        U12 = (G1inv.T.conjugate() @ (1/chi1.conjugate() * sp.eye(N)) - sp.eye(N)) @ Plist[j] @ G2inv
        U21 = (G2inv.T.conjugate() @ (1/chi2.conjugate() * sp.eye(N)) - sp.eye(N)) @ Plist[j] @ G1inv 
        # print(f"Density of U11, U22, U12, U21 is {zsp.density(U11)}, {zsp.density(U22)}, {zsp.density(U12)}, {zsp.density(U21)}")
        
        # Constraint 1 
        temp1 = sp.dok_matrix((3*N, 3*N), dtype=complex)
        temp2 = sp.dok_matrix((3*N, 3*N), dtype=complex)
        temp1[0:N, 0:N] += zsp.Asym(U11)
        temp2[0:N, 0:N] += zsp.Sym(U11)
        gradZTT[j, 0, 0] += temp1 
        gradZTT[j, 0, 1] += temp2 

        # Constraint 2 
        temp1 = sp.dok_matrix((3*N, 3*N), dtype=complex)
        temp1[2*N:, 2*N:] += zsp.Asym(U22)
        temp1[N:2*N, 2*N:] += - P @ G2inv / 2j
        temp1[2*N:, N:2*N] += temp1[N:2*N, 2*N:].T.conjugate()
        gradZTT[j, 1, 0] = temp1

        temp2 = sp.dok_matrix((3*N, 3*N), dtype=complex)
        temp2[2*N:, 2*N:] += zsp.Sym(U22)
        temp2[N:2*N, 2*N:] += - P @ G2inv / 2
        temp2[2*N:, N:2*N] += temp2[N:2*N, 2*N:].T.conjugate()
        gradZTT[j, 1, 1] = temp2

        # Constraint 3
        temp1 = sp.dok_matrix((3*N, 3*N), dtype=complex)
        temp1[0:N, 2*N:] += U12 / 2j
        temp1[2*N:, 0:N] += temp1[0:N, 2*N:].T.conjugate()
        gradZTT[j, 2, 0] = temp1

        temp2 = sp.dok_matrix((3*N, 3*N), dtype=complex)
        temp2[0:N, 2*N:] += U12 / 2
        temp2[2*N:, 0:N] += temp2[0:N, 2*N:].T.conjugate()
        gradZTT[j, 2, 1] = temp2

        # Constraint 4 
        temp1 = sp.dok_matrix((3*N, 3*N), dtype=complex)
        temp1[0:N, N:2*N] += U12 / 2j
        temp1[N:2*N, 0:N] += temp1[0:N, N:2*N].T.conjugate()
        gradZTT[j, 3, 0] = temp1

        temp2 = sp.dok_matrix((3*N, 3*N), dtype=complex)
        temp2[0:N, N:2*N] += U12 / 2
        temp2[N:2*N, 0:N] += temp2[0:N, N:2*N].T.conjugate()
        gradZTT[j, 3, 1] = temp2

        # Constraint 5 
        temp1 = sp.dok_matrix((3*N, 3*N), dtype=complex)
        temp1[2*N:, 0:N] += U21 / 2j
        temp1[0:N, 2*N:] += temp1[2*N:, 0:N].T.conjugate()
        temp1[N:2*N, 0:N] += - P @ G1inv / 2j
        temp1[0:N, N:2*N] += temp1[N:2*N, 0:N].T.conjugate()
        gradZTT[j, 4, 0] = temp1

        temp2 = sp.dok_matrix((3*N, 3*N), dtype=complex)
        temp2[2*N:, 0:N] += U21 / 2
        temp2[0:N, 2*N:] += temp2[2*N:, 0:N].T.conjugate()
        temp2[N:2*N, 0:N] += - P @ G1inv / 2
        temp2[0:N, N:2*N] += temp2[N:2*N, 0:N].T.conjugate()
        gradZTT[j, 4, 1] = temp2

        # Constraint 6
        temp1 = sp.dok_matrix((3*N, 3*N), dtype=complex)
        temp1[N:2*N, N:2*N] += - zsp.Asym(P @ G2inv)
        temp1[2*N:, N:2*N] += U22 / 2j
        temp1[N:2*N, 2*N:] += temp1[2*N:, N:2*N].T.conjugate()
        gradZTT[j, 5, 0] = temp1

        temp2 = sp.dok_matrix((3*N, 3*N), dtype=complex)
        temp2[N:2*N, N:2*N] += - zsp.Sym(P @ G2inv)
        temp2[2*N:, N:2*N] += U22 / 2
        temp2[N:2*N, 2*N:] += temp2[2*N:, N:2*N].T.conjugate()
        gradZTT[j, 5, 1] = temp2

        # # J2 J2 
        temp1 = sp.dok_matrix((3*N, 3*N), dtype=complex)
        temp1[N:2*N, N:2*N] += G2inv.T.conjugate() @ Plist[j] @ G2inv 
        temp1[0:N, 0:N] += - Cc * G1inv.T.conjugate() @ (alpha.conjugate() * Plist[j] * alpha) @ G1inv 
        gradZTT2[j, 0] = temp1

    # Convex constraint 
    if extra:
        pass
        # gradZTT3[0, 0:N, 0:N] += zops.Asym(U1)
        # gradZTT3[0, N:2*N, N:2*N] += -zops.Asym(G2.conjugate().T)
        # gradZTT3[0, 2*N:, 2*N:] += zops.Asym(U2)
        # gradZTT3[0, N:2*N, 2*N:] += -zops.Asym(G2.conjugate().T)
        # gradZTT3[0, 2*N:, N:2*N] += -zops.Asym(G2.conjugate().T)
        
    gradZTT = np.reshape(gradZTT, (num_regions*6*2))
    gradZTT2 = np.reshape(gradZTT2, (num_regions*1))

    if extra: 
        gradZTT = np.concatenate((gradZTT, gradZTT2, gradZTT3), axis=0)
    else: 
        gradZTT = np.concatenate((gradZTT, gradZTT2), axis=0)
    gradZTTlist = []
    for i in range(gradZTT.shape[0]):
        gradZTTlist.append(sp.coo_array(gradZTT[i])) # most operations will be nonzero(), coo_array is faster
    return gradZTTlist


def get_gradZTS_S(Plist, S, G1inv, G2inv, extra):
    num_regions = len(Plist)
    N = Plist[0].shape[1]

    gradZTS_S = np.zeros((num_regions, 6, 2, 3*N), dtype=complex)
    gradZTS_S2 = np.zeros((num_regions, 1, 3*N), dtype=complex) 
    if extra: gradZTS_S3 = np.zeros((1, 3*N), dtype=complex)

    for j in range(num_regions):
        gradZTS_S[j, 0, 0, 0:N] += (-G1inv.T.conjugate() @ Plist[j] / 2j) @ S
        gradZTS_S[j, 0, 1, 0:N] += (G1inv.T.conjugate() @ Plist[j] / 2) @ S

        gradZTS_S[j, 2, 0, 2*N:] += (-G2inv.T.conjugate() @ Plist[j] / 2j) @ S
        gradZTS_S[j, 2, 1, 2*N:] += (G2inv.T.conjugate() @ Plist[j] / 2) @ S

        gradZTS_S[j, 3, 0, N:2*N] += (-G2inv.T.conjugate() @ Plist[j] / 2j) @ S
        gradZTS_S[j, 3, 1, N:2*N] += (G2inv.T.conjugate() @ Plist[j] / 2) @ S

    # Convex constraint 
    if extra: pass #gradZTS_S3[0, 0:N] += - S / 2j

    gradZTS_S = np.reshape(gradZTS_S, (num_regions*6*2, 3*N))
    gradZTS_S2 = np.reshape(gradZTS_S2, (num_regions*1, 3*N))

    if extra: 
        gradZTS_S = np.concatenate((gradZTS_S, gradZTS_S2, gradZTS_S3), axis=0)
    else: 
        gradZTS_S = np.concatenate((gradZTS_S, gradZTS_S2), axis=0)

    return gradZTS_S

