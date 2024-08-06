import numpy as np
import scipy.linalg as la
import sys 
import scipy.sparse as sp
sys.path.append('../../')
from dualbound.Lagrangian.zops_utils import zops

def get_gradZTT(chi_m, chi_b, G, Plist, npixels, N):
    """
    Evaluates dZTT/d\lambda. 
    Order of Lagrange multipliers is bS, bA, mS, mA, nS, nA, Os
    (background, material, new, orthogonality) / (Symmetric, Asymmetric)
    See notes on Bounds formalism for non-vacuum background for details on these Lagrange multipliers

    Parameters
    ----------
    chib_inv : np.ndarray
        Inverse of the background susceptibility
    chi_inv : np.ndarray
        Inverse of the material susceptibility
    G : np.ndarray
        Green's function
    Plist : list
        List of projection matrices
    npixels : int
        Number of projections 
    N : int
        Number of pixels in design region (length of T vector)
    """
    print("This theory needs to be re-written, do not use this! ")
    exit()
    gradZTT = np.zeros((npixels, 3, 2, 2*N, 2*N), dtype=complex) # First part of gradient

    # chi_inv_minus_G = (1/chi_m + 1/chi_b)*np.eye(N) - G
    # A = chi_inv_minus_G # just a shorter name for it while maintaining clarity 

    chi_inv_b_minus_G = (1/chi_b)*np.eye(N) - G
    C = chi_inv_b_minus_G

    chi_inv_m_minus_G = (1/chi_m)*np.eye(N) - G
    B = chi_inv_m_minus_G

    # chiInv_minus_G = chi_inv*np.eye(N) - G
    # chibInv_minus_G = chib_inv*np.eye(N) - G

    for j in range(npixels):
        P = Plist[j].astype(complex)

        # lambda_q
        # Correct and agrees with multimat code 
        gradZTT[j, 0, 0, 0:N, 0:N] = zops.Sym(P @ B.T.conjugate())
        gradZTT[j, 0, 1, 0:N, 0:N] = zops.Asym(P @ B.T.conjugate())
        gradZTT[j, 0, 0, N:, N:] = zops.Sym(P @ C.T.conjugate())
        gradZTT[j, 0, 1, N:, N:] = zops.Asym(P @ C.T.conjugate())
        gradZTT[j, 0, 0, 0:N, N:] = -zops.Sym(P @ G.T.conjugate())
        gradZTT[j, 0, 0, N:, 0:N] = -zops.Sym(P @ G.T.conjugate())
        gradZTT[j, 0, 1, 0:N, N:] = -zops.Asym(P @ G.T.conjugate())
        gradZTT[j, 0, 1, N:, 0:N] = -zops.Asym(P @ G.T.conjugate())
        
        # lambda_n
        gradZTT[j, 1, 0, N:, 0:N] = -(1/2)*P @ B.T.conjugate() @ C
        gradZTT[j, 1, 0, 0:N, N:] = gradZTT[j, 1, 0, N:, 0:N].T.conjugate()
        gradZTT[j, 1, 1, N:, 0:N] = -(1/2j)*P @ B.T.conjugate() @ C
        gradZTT[j, 1, 1, 0:N, N:] = gradZTT[j, 1, 1, N:, 0:N].T.conjugate()

        # lambda_O
        gradZTT[j, 2, 0, N:, 0:N] = -((1/2)*P).todense()
        gradZTT[j, 2, 0, 0:N, N:] = gradZTT[j, 2, 0, N:, 0:N].T.conjugate()
        gradZTT[j, 2, 1, N:, 0:N] = -((1/2j)*P).todense()
        gradZTT[j, 2, 1, 0:N, N:] = gradZTT[j, 2, 1, N:, 0:N].T.conjugate()

    gradZTT = gradZTT.reshape((npixels*3*2, 2*N, 2*N), order='C')
    return gradZTT

def get_gradZTS_S(S, chi_m, chi_b, G, Plist, npixels, N):
    gradZTS_S = np.zeros((npixels, 3, 2, 2*N), dtype=complex)
    G_minus_chi_b_inv = G - (1/chi_b)*np.eye(N)
    B = G_minus_chi_b_inv
    G_minus_chi_m_inv = G - (1/chi_m)*np.eye(N)
    C = G_minus_chi_m_inv
    for j in range(npixels):
        P = Plist[j].astype(complex)

        # Z^ST_b
        gradZTS_S[j, 0, 0, N:] = (1/2)   * P @ S # lambda_q
        gradZTS_S[j, 0, 1, N:] = (-1/2j) * P @ S # lambda_q
        gradZTS_S[j, 1, 0, N:] = ((1/2)* B @ P).T.conjugate() @ S # lambda_n
        gradZTS_S[j, 1, 1, N:] = (-(1/2j)* B @ P).T.conjugate() @ S # lambda_n

        # Z^ST
        gradZTS_S[j, 0, 0, 0:N] = (1/2) * P @ S # lambda_q
        gradZTS_S[j, 0, 1, 0:N] = (-1/2j)*P @ S # lambda_q
        gradZTS_S[j, 1, 0, 0:N] = ((1/2)*  P @ C).T.conjugate() @ S # lambda_n
        gradZTS_S[j, 1, 1, 0:N] = ((1/2j)* P @ C).T.conjugate() @ S # lambda_n

    gradZTS_S = gradZTS_S.reshape((npixels*3*2, 2*N), order='C')
    return gradZTS_S 

def get_S_gradZSS_S(S, Plist, npixels):
    grad = np.zeros((npixels, 6), dtype=complex) # There are 6 lags for this problem
    grad[:, 2] = np.vectorize(lambda j: np.conjugate(S) @ Plist[j] @ S)(range(npixels))  # 5 is the location of lambda_n,S
    return np.reshape(grad, (npixels*6))

if __name__ == '__main__':
    import time 
    print('Lazy profiling of Zops_novac.py')
    N = 20
    npixels = 1
    Plist = np.zeros((npixels), dtype=object)
    for i in range(npixels):
        Plist[i] = sp.csr_matrix(np.eye(N))

    chi_inv = 1/(3+1j)
    chib_inv = 1/(4+1j)
    G = np.random.rand(N,N) + 1j*np.random.rand(N,N)
    G = G + G.T.conjugate()
    S = np.random.rand(N) + 1j*np.random.rand(N)

    # Test get_gradZTT
    t1 = time.time()
    gradZTT = get_gradZTT(chi_inv, chib_inv, G, Plist, npixels, N)
    print(f"Time for get_gradZTT: {time.time() - t1}")

    # Test get_gradZTS_S
    t1 = time.time()
    gradZTS_S = get_gradZTS_S(S, chi_inv, chib_inv, G, Plist, npixels, N)
    print(f"Time for get_gradZTS_S: {time.time() - t1}")

    # Test get_gradZTS_S
    t1 = time.time()
    vectorized = get_S_gradZSS_S(S, Plist, npixels)
    print(f"Time for get_S_gradZSS_S: {time.time() - t1}")
    

# Graveyard. These functions used 8 Lagrange multipliers with the erroneous theory maybe 
# old with A matrix 
# def get_gradZTT(chi_m, chi_b, G, Plist, npixels, N):
#     """
#     Evaluates dZTT/d\lambda. 
#     Order of Lagrange multipliers is bS, bA, mS, mA, nS, nA, Os
#     (background, material, new, orthogonality) / (Symmetric, Asymmetric)
#     See notes on Bounds formalism for non-vacuum background for details on these Lagrange multipliers

#     Parameters
#     ----------
#     chib_inv : np.ndarray
#         Inverse of the background susceptibility
#     chi_inv : np.ndarray
#         Inverse of the material susceptibility
#     G : np.ndarray
#         Green's function
#     Plist : list
#         List of projection matrices
#     npixels : int
#         Number of projections 
#     N : int
#         Number of pixels in design region (length of T vector)
#     """

#     gradZTT = np.zeros((npixels, 3, 2, 2*N, 2*N), dtype=complex) # First part of gradient

#     chi_inv_minus_G = (1/chi_m + 1/chi_b)*np.eye(N) - G
#     A = chi_inv_minus_G # just a shorter name for it while maintaining clarity 

#     chi_inv_b_minus_G = (1/chi_b)*np.eye(N) - G
#     B = chi_inv_b_minus_G

#     chi_inv_m_minus_G = (1/chi_m)*np.eye(N) - G
#     C = chi_inv_m_minus_G

#     # chiInv_minus_G = chi_inv*np.eye(N) - G
#     # chibInv_minus_G = chib_inv*np.eye(N) - G

#     for j in range(npixels):
#         P = Plist[j].astype(complex)

#         # lambda_q
#         # Maybe would be good to type this up and check this through 
#         gradZTT[j, 0, 0, 0:N, 0:N] = zops.Sym(P @ A.T.conjugate())
#         gradZTT[j, 0, 1, 0:N, 0:N] = zops.Asym(P @ A.T.conjugate())
#         gradZTT[j, 0, 0, N:, N:] = zops.Sym(P @ A.T.conjugate())
#         gradZTT[j, 0, 1, N:, N:] = zops.Asym(P @ A.T.conjugate())
#         gradZTT[j, 0, 0, 0:N, N:] = zops.Sym(P @ A)
#         gradZTT[j, 0, 0, N:, 0:N] = zops.Sym(P @ A)
#         gradZTT[j, 0, 1, 0:N, N:] = zops.Asym(P @ A.T.conjugate())
#         gradZTT[j, 0, 1, N:, 0:N] = zops.Asym(P @ A.T.conjugate()) 
        
#         # lambda_n
#         gradZTT[j, 1, 0, N:, 0:N] = -(1/2)*P @ B.T.conjugate() @ C
#         gradZTT[j, 1, 0, 0:N, N:] = gradZTT[j, 1, 0, N:, 0:N].T.conjugate()
#         gradZTT[j, 1, 1, N:, 0:N] = -(1/2j)*P @ B.T.conjugate() @ C
#         gradZTT[j, 1, 1, 0:N, N:] = gradZTT[j, 1, 1, N:, 0:N].T.conjugate()

#         # lambda_O
#         gradZTT[j, 2, 0, N:, 0:N] = -((1/2)*P).todense()
#         gradZTT[j, 2, 0, 0:N, N:] = gradZTT[j, 2, 0, N:, 0:N].T.conjugate()
#         gradZTT[j, 2, 1, N:, 0:N] = -((1/2j)*P).todense()
#         gradZTT[j, 2, 1, 0:N, N:] = gradZTT[j, 2, 1, N:, 0:N].T.conjugate()

#     gradZTT = gradZTT.reshape((npixels*3*2, 2*N, 2*N), order='C')
#     return gradZTT

# old old:
# def get_gradZTT(chi_inv, chib_inv, G, Plist, npixels, N):
#     """
#     Evaluates dZTT/d\lambda. 
#     Order of Lagrange multipliers is bS, bA, mS, mA, nS, nA, Os
#     (background, material, new, orthogonality) / (Symmetric, Asymmetric)
#     See notes on Bounds formalism for non-vacuum background for details on these Lagrange multipliers

#     Parameters
#     ----------
#     chib_inv : np.ndarray
#         Inverse of the background susceptibility
#     chi_inv : np.ndarray
#         Inverse of the material susceptibility
#     G : np.ndarray
#         Green's function
#     Plist : list
#         List of projection matrices
#     npixels : int
#         Number of projections 
#     N : int
#         Number of pixels in design region (length of T vector)
#     """

#     gradZTT = np.zeros((npixels, 4, 2, 2*N, 2*N), dtype=complex) # First part of gradient

#     chiInv_minus_G = chi_inv*np.eye(N) - G
#     chibInv_minus_G = chib_inv*np.eye(N) - G

#     for j in range(npixels):
#         P = Plist[j].astype(complex)
#         # lambda_b 
#         gradZTT[j, 0, 0, N:, N:] = -zops.Sym(P @ chibInv_minus_G.T.conjugate())
#         gradZTT[j, 0, 1, N:, N:] = -zops.Asym(P @ chibInv_minus_G.T.conjugate())
        
#         # lambda_m
#         gradZTT[j, 1, 0, 0:N, 0:N] = -zops.Sym(P @ chiInv_minus_G.T.conjugate())
#         gradZTT[j, 1, 1, 0:N, 0:N] = -zops.Asym(P @ chiInv_minus_G.T.conjugate())
        
#         # lambda_n
#         gradZTT[j, 2, 0, N:, 0:N] = 0 #(1/2)*P @ chibInv_minus_G.T.conjugate() @ chiInv_minus_G
#         gradZTT[j, 2, 0, 0:N, N:] = gradZTT[j, 2, 0, N:, 0:N].T.conjugate()
#         gradZTT[j, 2, 1, N:, 0:N] = 0 #(1/2j)*P @ chibInv_minus_G.T.conjugate() @ chiInv_minus_G
#         gradZTT[j, 2, 1, 0:N, N:] = gradZTT[j, 2, 1, N:, 0:N].T.conjugate()

#         # lambda_O
#         gradZTT[j, 3, 0, N:, 0:N] = ((1/2)*P).todense()
#         gradZTT[j, 3, 0, 0:N, N:] = gradZTT[j, 3, 0, N:, 0:N].T.conjugate()
#         gradZTT[j, 3, 1, N:, 0:N] = ((1/2j)*P).todense()
#         gradZTT[j, 3, 1, 0:N, N:] = gradZTT[j, 3, 1, N:, 0:N].T.conjugate()

#     gradZTT = gradZTT.reshape((npixels*8, 2*N, 2*N), order='C')
#     return -gradZTT

# def get_gradZTS_S(S, chi_inv, chib_inv, G, Plist, npixels, N):
#     gradZTS_S = np.zeros((npixels, 4, 2, 2*N), dtype=complex)
#     G_minus_chiinv = G - chi_inv*np.eye(N)
#     G_minus_chibinv = G - chib_inv*np.eye(N)
#     print(chi_inv, chib_inv)
#     for j in range(npixels):
#         P = Plist[j].astype(complex)

#         # Z^ST_b
#         gradZTS_S[j, 0, 0, N:] = ((1/2)*P).T.conjugate() @ S # lambda_b
#         gradZTS_S[j, 0, 1, N:] = ((1/2j)*P).T.conjugate() @ S # lambda_b
#         gradZTS_S[j, 2, 0, N:] = 0 #((1/2)*(G_minus_chibinv) @ P).T.conjugate() @ S # lambda_n
#         gradZTS_S[j, 2, 1, N:] = 0 #(-(1/2j)*(G_minus_chibinv) @ P).T.conjugate() @ S # lambda_n

#         # Z^ST
#         gradZTS_S[j, 1, 0, 0:N] = ((1/2)* P).T.conjugate() @ S # lambda_m 
#         gradZTS_S[j, 1, 1, 0:N] = ((1/2j)*P).T.conjugate() @ S # lambda_m 
#         gradZTS_S[j, 2, 0, 0:N] = 0 #((1/2)*  P @ G_minus_chiinv).T.conjugate() @ S # lambda_n
#         gradZTS_S[j, 2, 1, 0:N] = 0 #((1/2j)* P @ G_minus_chiinv).T.conjugate() @ S # lambda_n

#     gradZTS_S = gradZTS_S.reshape((npixels*8, 2*N), order='C')
#     return gradZTS_S 

# def get_S_gradZSS_S(S, Plist, npixels):
#     grad = np.zeros((npixels, 8), dtype=complex) # There are 8 lags for this problem
#     # grad[:, 4] = np.vectorize(lambda j: np.conjugate(S) @ Plist[j] @ S)(range(npixels))  # 5 is the location of lambda_n,S
#     return np.reshape(grad, (npixels*8))