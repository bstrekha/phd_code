import numpy as np
import scipy.linalg as la 
import matplotlib.pyplot as plt 
import sys
sys.path.append('../../../')
from dualbound.Lagrangian.zops_utils import zops

def get_gradZTT(Plist, U1, U2, G2, alpha, Cc, extra):
    num_regions = Plist.shape[0]
    N = Plist.shape[1]
    gradZTT = np.zeros((num_regions, 6, 2, 3*N, 3*N), dtype=complex)
    gradZTT2 = np.zeros((num_regions, 1, 3*N, 3*N), dtype=complex)
    if extra: gradZTT3 = np.zeros((1, 3*N, 3*N), dtype=complex)

    for j in range(num_regions):
        # T1 T1
        gradZTT[j, 0, 0, 0:N, 0:N] += zops.Asym(U1 @ Plist[j]) # T1T1
        gradZTT[j, 0, 1, 0:N, 0:N] +=  zops.Sym(U1 @ Plist[j]) # T1T1

        # T2 T2 
        gradZTT[j, 1, 0, 2*N:, 2*N:] += zops.Asym(U2 @ Plist[j])
        gradZTT[j, 1, 0, N:2*N, 2*N:] += -G2.T.conjugate() @ Plist[j] / 2j
        gradZTT[j, 1, 0, 2*N:, N:2*N] += gradZTT[j, 1, 0, N:2*N, 2*N:].T.conjugate()
        
        gradZTT[j, 1, 1, 2*N:, 2*N:] +=  zops.Sym(U2 @ Plist[j])
        gradZTT[j, 1, 1, N:2*N, 2*N:] += -G2.T.conjugate() @ Plist[j] / 2
        gradZTT[j, 1, 1, 2*N:, N:2*N] += gradZTT[j, 1, 1, N:2*N, 2*N:].T.conjugate()

        # T1 T2 
        gradZTT[j, 2, 0, 0:N, 2*N:] += U1 @ Plist[j] / 2j
        gradZTT[j, 2, 0, 2*N:, 0:N] += gradZTT[j, 2, 0, 0:N, 2*N:].T.conjugate()
        gradZTT[j, 2, 1, 0:N, 2*N:] += U1 @ Plist[j] / 2
        gradZTT[j, 2, 1, 2*N:, 0:N] += gradZTT[j, 2, 1, 0:N, 2*N:].T.conjugate()

        gradZTT[j, 4, 0, 2*N:, 0:N] += U2 @ Plist[j] / 2j 
        gradZTT[j, 4, 0, 0:N, 2*N:] += gradZTT[j, 4, 0, 2*N:, 0:N] .T.conjugate()
        gradZTT[j, 4, 1, 2*N:, 0:N] += U2 @ Plist[j] / 2
        gradZTT[j, 4, 1, 0:N, 2*N:] += gradZTT[j, 4, 1, 2*N:, 0:N].T.conjugate()

        # T1 J2
        gradZTT[j, 3, 0, 0:N, N:2*N] += U1 @ Plist[j] / 2j
        gradZTT[j, 3, 0, N:2*N, 0:N] += gradZTT[j, 3, 0, 0:N, N:2*N].T.conjugate()
        gradZTT[j, 3, 1, 0:N, N:2*N] += U1 @ Plist[j] / 2
        gradZTT[j, 3, 1, N:2*N, 0:N] += gradZTT[j, 3, 1, 0:N, N:2*N].T.conjugate()
        
        gradZTT[j, 4, 0, N:2*N, 0:N] += -G2.T.conjugate() @ Plist[j] / 2j
        gradZTT[j, 4, 0, 0:N, N:2*N] += gradZTT[j, 4, 0, N:2*N, 0:N].T.conjugate()
        gradZTT[j, 4, 1, N:2*N, 0:N] += -G2.T.conjugate() @ Plist[j] / 2
        gradZTT[j, 4, 1, 0:N, N:2*N] += gradZTT[j, 4, 1, N:2*N, 0:N].T.conjugate()

        gradZTT[j, 5, 0, N:2*N, N:2*N] += - zops.Asym(G2.T.conjugate() @ Plist[j])
        gradZTT[j, 5, 0, 2*N: , N:2*N] += U2 @ Plist[j] / 2j
        gradZTT[j, 5, 0, N:2*N, 2*N:] += gradZTT[j, 5, 0, 2*N:, N:2*N].T.conjugate()
        gradZTT[j, 5, 1, N:2*N, N:2*N] += - zops.Sym(G2.T.conjugate() @ Plist[j])
        gradZTT[j, 5, 1, 2*N: , N:2*N] += U2 @ Plist[j] / 2
        gradZTT[j, 5, 1, N:2*N, 2*N:] += gradZTT[j, 5, 1, 2*N:, N:2*N].T.conjugate()

        # J2 J2 
        gradZTT2[j, 0, N:2*N, N:2*N] += Plist[j] 
        gradZTT2[j, 0, 0:N, 0:N] += - Cc * alpha.conjugate() * Plist[j] * alpha

    # Convex constraint 
    if extra:
        gradZTT3[0, 0:N, 0:N] += zops.Asym(U1)
        gradZTT3[0, N:2*N, N:2*N] += -zops.Asym(G2.conjugate().T)
        gradZTT3[0, 2*N:, 2*N:] += zops.Asym(U2)
        gradZTT3[0, N:2*N, 2*N:] += -zops.Asym(G2.conjugate().T)
        gradZTT3[0, 2*N:, N:2*N] += -zops.Asym(G2.conjugate().T)
        
    gradZTT = np.reshape(gradZTT, (num_regions*6*2, 3*N, 3*N))
    gradZTT2 = np.reshape(gradZTT2, (num_regions*1, 3*N, 3*N))

    if extra: 
        gradZTT = np.concatenate((gradZTT, gradZTT2, gradZTT3), axis=0)
    else: 
        gradZTT = np.concatenate((gradZTT, gradZTT2), axis=0)
    return gradZTT


def get_gradZTS_S(Plist, S, extra):
    num_regions = Plist.shape[0]
    N = Plist.shape[1]

    gradZTS_S = np.zeros((num_regions, 6, 2, 3*N), dtype=complex)
    gradZTS_S2 = np.zeros((num_regions, 1, 3*N), dtype=complex) 
    if extra: gradZTS_S3 = np.zeros((1, 3*N), dtype=complex)

    for j in range(num_regions):
        gradZTS_S[j, 0, 0, 0:N] += (-Plist[j] / 2j) @ S
        gradZTS_S[j, 0, 1, 0:N] += (Plist[j] / 2) @ S

        gradZTS_S[j, 2, 0, 2*N:] += (-Plist[j] / 2j) @ S
        gradZTS_S[j, 2, 1, 2*N:] += (Plist[j] / 2) @ S

        gradZTS_S[j, 3, 0, N:2*N] += (-Plist[j] / 2j) @ S
        gradZTS_S[j, 3, 1, N:2*N] += (Plist[j] / 2) @ S

    # Convex constraint 
    if extra: gradZTS_S3[0, 0:N] += - S / 2j

    gradZTS_S = np.reshape(gradZTS_S, (num_regions*6*2, 3*N))
    gradZTS_S2 = np.reshape(gradZTS_S2, (num_regions*1, 3*N))

    if extra: 
        gradZTS_S = np.concatenate((gradZTS_S, gradZTS_S2, gradZTS_S3), axis=0)
    else: 
        gradZTS_S = np.concatenate((gradZTS_S, gradZTS_S2), axis=0)

    return gradZTS_S

