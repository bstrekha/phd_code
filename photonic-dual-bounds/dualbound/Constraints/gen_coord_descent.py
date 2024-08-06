import numpy as np
import scipy.linalg as la
import scipy.sparse as sp 
import sys 
sys.path.append('../../')
import dualbound.Lagrangian.zops_utils.zops as zops
import dualbound.Lagrangian.zops_utils.zops_sparse as zops_sp
import dualbound.Lagrangian.dualgrad as dualgrad
import dualbound.Lagrangian.dualgrad_sparse as dualgrad_sp

from dualbound.Lagrangian.spatialProjopt_Zops_Msparse import get_ZTT, get_Msparse_gradZTT, get_Msparse_gradZTS_S, check_Msparse_spatialProj_Lags_validity, check_Msparse_spatialProj_incLags_validity, get_Msparse_inc_PD_ZTT_mineig
from dualbound.Lagrangian.spatialProjopt_dualgradHess_fakeS_Msparse import get_inc_spatialProj_dualgradHess_fakeS_Msparse
from dualbound.Optimization.fakeSource_with_restart_singlematrix import fakeS_with_restart_singlematrix, fakeS_single_restart_Newton
from dualbound.Constraints.mask_iterative_splitting import reduce_Plist

# import scipy.optimize as scopt
# import matplotlib.pyplot as plt 

'''
This file describes code for the coordinate descent algorithm for solving the 
Lagrange dual problem. It is written for the Raman problem, since writing a general one is hard 
TODO(alessio) : write a general case 
Author: alessio 
Theory by pengning 
'''

# This is the general idea: 
# 1. Start with lags from global optimization
# 2. get gradZTT real only, with a custom P made of lambda_R + i*lambda_I, set new multipliers to 1. You will end up with 4 constraints. 
# 3. Compute maximum violation and direction to avoid semi-definite boundary. This will always use the global gradZTT parts, since there is no P_j in the derivation
# 4. Calculate the new component of gradZTT (and gradZTS) with the new custom P (Plist really, one for each constraint)
# 5. Append the new gradZTT to the old one
# 6. Redefine the lambda functions, and run the optimization with the new Lagrange multipliers set to 0 
# 7. Save the new run, will need a new savefunc and new csv that lists number of iterations and final number of constraints 
# 8. If you've exceeded the number of constraints you're willing to use, condense them using reduction 
# 9. Repeat from step 3. 

def get_Plist_from_vecs(vecs):
    return list(map(lambda v: np.diag(v), vecs))

def get_vecs_from_sparse_Plist(Plist):
    return list(map(lambda p: np.diag(p.todense()), Plist))

def get_Plist_from_vecs_sparse(vecs):
    return list(map(lambda v: sp.diags(v), vecs))

def get_new_vecs_random(fSlist, N, cclasses, get_gradZTT, get_gradZTS_S, T, ZTT, niters):
    eigw, eigv = la.eigh(ZTT)
    eigw = eigw[0]
    x = eigv[:,0]

    res = -np.inf 
    for i in range(niters):    
        vopt = np.exp(1j*np.random.random((cclasses, N)))
        Plist = get_Plist_from_vecs(vopt)
        boundary = 0
        violation = 0
        Plist = get_Plist_from_vecs(vopt)
        gradZTT = get_gradZTT(Plist)
        gradZTS_S = get_gradZTS_S(Plist)

        for i in range(cclasses):
            boundary += np.abs(x.conjugate() @ (gradZTT[i] @ x))
            violation += np.real(T.conjugate() @ gradZTS_S[i]) + T.conjugate() @ gradZTT[i] @ T 

        result = np.real(boundary + violation)
        if result > res:
            res = result
    return vopt

def get_new_vecs_opt(N, cclasses, get_gradZTT, get_gradZTS_S, T, ZTT, niters):
    eigw, eigv = la.eigh(ZTT)
    eigw = eigw[0]
    x = eigv[:,0]

    def f(phases):
        p = np.reshape(phases, (cclasses, N))
        vopt = np.exp(1j*p)
        Plist = get_Plist_from_vecs(vopt)
        gradZTT = get_gradZTT(Plist)
        gradZTS_S = get_gradZTS_S(Plist)

        boundary = 0
        violation = 0
        for i in range(cclasses):
            boundary += np.abs(x.conjugate() @ (gradZTT[i] @ x))
            violation += np.real(T.conjugate() @ gradZTS_S[i]) + T.conjugate() @ gradZTT[i] @ T

        result = -np.real(boundary + violation)
        print(result)
        return result 

    return np.reshape(scopt.minimize(f, np.random.random((cclasses*N))), (cclasses, N))

def orthogonalize_vecs(cclasses, Plist, vecs_new):
    vecs_final = vecs_new.copy()
    for i in range(cclasses):
        for j in range(cclasses):
            v = np.diag(Plist[j])
            vecs_final[i] -= (vecs_final[i].T.conjugate() @ v / (v.conjugate() @ v)) * v
        vecs_final[i] /= np.sqrt(vecs_final[i].conjugate() @ vecs_final[i])
    return vecs_final

def orthogonalize_vecs_sparse(cclasses, Plist, vecs_new):
    vecs_final = vecs_new.copy()
    for i in range(cclasses):
        for j in range(cclasses):
            v = np.diag(Plist[j].todense())
            vecs_final[i] -= (vecs_final[i].T.conjugate() @ v / (v.conjugate() @ v)) * v
        vecs_final[i] /= np.sqrt(vecs_final[i].conjugate() @ vecs_final[i])
    return vecs_final
            
def gcd_dense(N, cclasses, get_gradZTT, get_gradZTS_S, dualconst, opt_func, global_lags, O_quad, O_lin_S, saveFunc, Pnum, maxiternum=10, saveint=1, get_new_vecs=get_new_vecs_random):
    '''
    N: Number of pixels in region (length of optimization vector)
    get_gradZTT_t: 
        tuple of form (get_gradZTT_m1, get_gradZTT_args)
        where get_gradZTT is a function that gets the m+1-th component of gradZTT by specifying a list of projectors 
    opt_args is of the form 
        [dualconst, ]
    '''
    
    Pstruct = np.eye(N, dtype=bool)
    include = np.ones(cclasses, dtype=bool)
    Lags = np.ones(cclasses)
    fSlist = [] 
    optLags = Lags[include]

    # Evaluate mindual, minLags, mingrad, minobj 
    # Need to convert global with imag, real to global real with one Lagrange multiplier. That is, take 8 multipliers -> 4
    alpha = -global_lags[0::2]*1j + global_lags[1::2]
    Plist = [np.eye(N, dtype=complex) * alpha[i] for i in range(len(Lags))]

    gradZTT = get_gradZTT(Plist)
    gradZTS_S = get_gradZTS_S(Plist)
    S_gradZSS_S = np.zeros(cclasses) # Any <S|gradZSS|S> terms should have been optimized away 

    validityfunc = lambda dof: zops.check_spatialProj_incLags_validity(dof, include, O_quad, gradZTT)
    mineigfunc = lambda dof: zops.get_inc_ZTT_mineig(dof, include, O_quad, gradZTT)
    dgHfunc = lambda dof, grad, dofHess, fSl, get_grad=True, get_Hess=True: (
        dualgrad.get_dual_and_derivatives(dof, grad, dofHess, include, O_lin_S, O_quad, gradZTS_S, gradZTT, 
                                          S_gradZSS_S, fSl, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess))
    
    assert(validityfunc(Lags))

    optgrad = np.zeros(len(optLags))
    dofHess = np.zeros((len(optLags), len(optLags)))
    optdual = dgHfunc(optLags, optgrad, dofHess, [], True, True)
    optobj = optdual - optLags @ optgrad 
    mindual, minobj, minLags, mingrad = optdual, optobj, optLags.copy(), optgrad.copy()
    iternum = 0 

    print(f'Calculated inital bound: {optdual/dualconst}')
    print()

    while True: 
        iternum += 1
        print(f'At dimension reduction iteration #{iternum}, bound is {mindual/dualconst}')
        print(f'number of projection constraints {len(Plist)}')

        if iternum == maxiternum:
            saveFunc(iternum, optdual, dualconst, optLags, optgrad, get_Plist_from_vecs(Plist))
            break 

        ZTT = zops.get_ZTT(optLags, O_quad, gradZTT)
        ZTS_S = zops.get_ZTS_S(optLags, O_lin_S, gradZTS_S)
        S_ZSS_S = np.zeros(len(ZTT))
        T = la.solve(ZTT, ZTS_S)

        vecs_new = get_new_vecs(N, cclasses, get_gradZTT, get_gradZTS_S, T, ZTT, 20)
        vecs_new = orthogonalize_vecs(cclasses, Plist, vecs_new)
        Plist_new = get_Plist_from_vecs(vecs_new)

        for i in range(len(Plist_new)):
            Plist.append(Plist_new[i])

        minLags = np.append(optLags, np.zeros(len(Plist_new)))
        gradZTT = np.append(gradZTT, get_gradZTT(Plist_new), axis=0)
        gradZTS_S = np.append(gradZTS_S, get_gradZTS_S(Plist_new), axis=0)
        S_gradZSS_S = np.zeros(len(Plist))
        include = np.ones(len(Plist), dtype=bool)

        validityfunc = lambda dof: zops.check_spatialProj_incLags_validity(dof, include, O_quad, gradZTT)
        mineigfunc = lambda dof: zops.get_inc_ZTT_mineig(dof, include, O_quad, gradZTT)
        dgHfunc = lambda dof, grad, dofHess, fSl, get_grad=True, get_Hess=True: (
            dualgrad.get_dual_and_derivatives(dof, grad, dofHess, include, O_lin_S, O_quad, gradZTS_S, gradZTT, 
                                          S_gradZSS_S, fSl, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess))
    

        # Run Newton iteration 
        optLags, optgrad, optdual, optval, fSlist = opt_func(minLags, dgHfunc, validityfunc, mineigfunc, fSlist)
        print(f'New bound: {optdual/dualconst}')
        print(f'number of projection constraints {len(Plist)}')

        if (not saveFunc is None) and (iternum % saveint == 0):
            saveFunc(iternum, optdual, dualconst, optLags, optgrad, get_Plist_from_vecs(Plist))
        
        if len(Plist) > Pnum * cclasses: 
            print('reducing Plist')
            print('len optLags', len(optLags), 'len gradZTT, gradZTS_S', len(gradZTT), len(gradZTS_S))
            new_Plist = []
            for i in range(cclasses):
                new_Plist.append(np.sum(optLags[i::cclasses, None, None] * np.array(Plist[i::cclasses]), axis=0))   

            Plist = new_Plist
            gradZTT = get_gradZTT(Plist)
            gradZTS_S = get_gradZTS_S(Plist)
            S_gradZSS_S = np.zeros(len(Plist))
            include = np.ones(len(Plist), dtype=bool)
            optLags = np.ones(len(include))

            print(f'New bound: {optdual/dualconst}')

    return 0

           
def gcd_sparse(N, cclasses, get_gradZTT, get_gradZTS_S, dualconst, opt_func, global_lags, O_quad, O_lin_S, saveFunc, Pnum, maxiternum=10, saveint=1, get_new_vecs=get_new_vecs_random):
    Pstruct = np.eye(N, dtype=bool)
    include = np.ones(cclasses, dtype=bool)
    Lags = np.ones(cclasses)
    fSlist = [] 
    optLags = Lags[include]

    # Evaluate mindual, minLags, mingrad, minobj 
    # Need to convert global with imag, real to global real with one Lagrange multiplier. That is, take 2 multipliers -> 1

    # alpha = global_lags[0::2]*1j + global_lags[1::2]
    # alpha = global_lags[0::2] + global_lags[1::2]*1j
    # alpha = -global_lags[0::2]*1j + global_lags[1::2] # original for Raman
    # alpha = -global_lags[0::2] + global_lags[1::2]*1j
    # alpha = global_lags[0::2]*1j - global_lags[1::2]
    alpha = global_lags[0::2] - global_lags[1::2]*1j # This can be chosen depending on the ORDER of your constraints (sym, asym vs asym, sym) and the sign 

    print(alpha)
    Plist = [sp.eye(N, dtype=complex) * alpha[i] for i in range(len(Lags))] 
    # Note that this is length of cclasses (number of constraint classes)
    # For "normal" constraints, cclasses = 1 so there will just be one P

    gradZTT = get_gradZTT(Plist)
    gradZTS_S = get_gradZTS_S(Plist)
    S_gradZSS_S = np.zeros(cclasses) # Any <S|gradZSS|S> terms should have been optimized away 

    validityfunc = lambda dof: zops_sp.check_spatialProj_incLags_validity(dof, include, O_quad, gradZTT)
    mineigfunc = lambda dof: zops_sp.get_inc_ZTT_mineig(dof, include, O_quad, gradZTT)
    dgHfunc = lambda dof, grad, dofHess, fSl, get_grad=True, get_Hess=True: (
        dualgrad_sp.get_dual_and_derivatives(dof, grad, dofHess, include, O_lin_S, O_quad, gradZTS_S, gradZTT, 
                                          S_gradZSS_S, fSl, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess))
    
    assert validityfunc(optLags) == 1
    # exit()

    optgrad = np.zeros(len(optLags))
    dofHess = np.zeros((len(optLags), len(optLags)))
    optdual = dgHfunc(optLags, optgrad, dofHess, [], True, True)
    optobj = optdual - optLags @ optgrad 
    mindual, minobj, minLags, mingrad = optdual, optobj, optLags.copy(), optgrad.copy()
    iternum = 0 

    # a bit hacky, if dualconst is just a const or the normalization factor... 
    # TODO(alessio): fix this more elegantly 
    dualconstnorm = dualconst if dualconst != 0 else 1 
    print(f'Calculated inital bound: {optdual/dualconstnorm}')
    print()

    while True: 
        iternum += 1
        print(f'At dimension reduction iteration #{iternum}, bound is {optdual/dualconstnorm}')
        print(f'number of projection constraints {len(Plist)}')

        if iternum == maxiternum and not saveFunc is None:
            saveFunc(iternum, optdual, dualconst, optLags, optgrad, get_Plist_from_vecs_sparse(Plist))
            break 

        ZTT = zops_sp.get_ZTT(optLags, O_quad, gradZTT)
        ZTS_S = zops_sp.get_ZTS_S(optLags, O_lin_S, gradZTS_S)
        S_ZSS_S = np.zeros(len(ZTT))
        GT = la.solve(ZTT, ZTS_S)
        # T = G1inv @ GT

        vecs_new = get_new_vecs(fSlist, N, cclasses, get_gradZTT, get_gradZTS_S, GT, ZTT, 20)
        vecs_new = orthogonalize_vecs_sparse(cclasses, Plist, vecs_new)
        Plist_new = get_Plist_from_vecs_sparse(vecs_new)

        for i in range(len(Plist_new)):
            Plist.append(Plist_new[i])

        minLags = np.append(optLags, np.zeros(len(Plist_new)))
        gradZTT = np.append(gradZTT, get_gradZTT(Plist_new), axis=0)
        gradZTS_S = np.append(gradZTS_S, get_gradZTS_S(Plist_new), axis=0)
        S_gradZSS_S = np.zeros(len(Plist))
        include = np.ones(len(Plist), dtype=bool)

        validityfunc = lambda dof: zops_sp.check_spatialProj_incLags_validity(dof, include, O_quad, gradZTT)
        mineigfunc = lambda dof: zops_sp.get_inc_ZTT_mineig(dof, include, O_quad, gradZTT)
        dgHfunc = lambda dof, grad, dofHess, fSl, get_grad=True, get_Hess=True: (
            dualgrad_sp.get_dual_and_derivatives(dof, grad, dofHess, include, O_lin_S, O_quad, gradZTS_S, gradZTT, 
                                          S_gradZSS_S, fSl, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess))
    

        # Run Newton iteration 
        optLags, optgrad, optdual, optval, fSlist = opt_func(minLags, dgHfunc, validityfunc, mineigfunc, fSlist)
        print(f'New bound: {optdual/dualconstnorm}')
        print(f'number of projection constraints {len(Plist)}')

        if (not saveFunc is None) and (iternum % saveint == 0):
            saveFunc(iternum, optdual, dualconst, optLags, optgrad, get_vecs_from_sparse_Plist(Plist))
        
        if len(Plist) > Pnum * cclasses: 
            print('reducing Plist')
            print('len optLags', len(optLags), 'len gradZTT, gradZTS_S', len(gradZTT), len(gradZTS_S))
            new_Plist = []
            Plist = np.array(Plist, dtype=object)

            assert cclasses == 1 # TODO(alessio): generalize this 
            newP = np.zeros((N,N), dtype=complex) #sp.csr_matrix((N, N), dtype=complex)
            for i in range(len(Plist)):
                newP += optLags[i] * Plist[i]
            newP = sp.csr_matrix(newP)

            new_Plist = [newP]
            # for i in range(cclasses):
                # new_Plist.append(np.sum(optLags[i::cclasses, None, None] * np.array(Plist[i::cclasses]), axis=0))   
            print(new_Plist)
            Plist = new_Plist 
            gradZTT = get_gradZTT(Plist) 
            gradZTS_S = get_gradZTS_S(Plist)
            S_gradZSS_S = np.zeros(len(Plist))
            include = np.ones(len(Plist), dtype=bool)
            optLags = np.ones(len(include))

            print(f'New bound: {optdual/dualconstnorm}')
            print(f'New lags: {optLags}')
            # input()

    return 0


# import sksparse.cholmod as chol
# def Cholesky_analyze_ZTT(O, gradZTT):
#     Lags = np.random.rand(len(gradZTT))
#     ZTT = get_ZTT(Lags, O, gradZTT)
#     # print('analyzing ZTT of format and shape', ZTT.format, ZTT.shape, 'and # of nonzero elements', ZTT.count_nonzero())
#     return chol.analyze(ZTT)


def outer_sparse_Pstruct(a, b, Pstruct):
    """
    equivalent to np.outer(a,b) * Pstruct where Pstruct is a boolean mask
    here Pstruct is represented as a sparse coo array with 1s on supported entries
    """
    outer_data = a[Pstruct.row] * b[Pstruct.col]
    return outer_data


def dual_space_reduction_iteration_Msparse_align_mineig_maxviol(chi, Si, Ginv, O_lin, O_quad, Pstruct=None, P0phase=1.0+0j, Pnum=1, dualconst=0.0, opttol=1e-2, fakeSratio=1e-2, gradConverge=False, singlefS=False, iter_period=20, outputFunc=None):
    """
    dual space reduction for sparse off-diagonal P constraints
    Pstruct is a sparse correlation matrix indicating structure of P
    Pstruct[i,j] = 1 if P can have a non-zero (i,j) entry and 0 otherwise
    generate new projection constraints to both have large Lagrangian gradient
    and have the Lagrangian gradient point in a direction that increases the minimum eigenvalue of ZTT
    """

    if Pstruct is None: #for sparse off-diagonal P, default assume off-diagonals are 0
        Pstruct = sp.eye(Ginv.shape[0], format='coo')
    Pstruct = Pstruct.tocoo() #for use later with outer_sparse_Pstruct

    P0 = sp.eye(Ginv.shape[0], dtype=complex, format='csc') * P0phase
    P0 /= sp.linalg.norm(P0)

    P0_data = np.zeros_like(Pstruct.row, dtype=complex)
    for i in range(len(Pstruct.row)):
        P0_data[i] = P0[Pstruct.row[i],Pstruct.col[i]]
    Pdatalist = [P0_data]
    GinvdagPdag = Ginv.conj().T @ P0.conj().T
    GinvdagPdaglist = [GinvdagPdag]
    UPlist = [(Ginv.conj().T @ GinvdagPdag.conj().T)/np.conj(chi) - GinvdagPdag.conj().T]

    print('check AsymUM definiteness')
    # t = time.time()
    UM = UPlist[0].todense()
    AsymUM = (UM-UM.conj().T)/2j
    print(la.eigvalsh(AsymUM)[0])
    # print('time taken to evaluate eigenvalues of dense AsymUM:', time.time()-t, flush=True)

    AsymUM_sp = (UPlist[0] - UPlist[0].conj().T)/2j
    # t = time.time()
    eig0_AsymUM = sp.linalg.eigsh(AsymUM_sp, k=1, sigma=0.0, which='LM', return_eigenvectors=False)
    print('using sparse ARPACK method, mineig for AsymUM is', eig0_AsymUM)
    # print('time used', time.time()-t, flush=True)
    
    gradZTT = get_Msparse_gradZTT(UPlist)
    gradZTS_S = get_Msparse_gradZTS_S(Si, GinvdagPdaglist)
    # ZTTchofac = Cholesky_analyze_ZTT(O_quad, gradZTT)
    
    include = np.ones(len(gradZTT), dtype=bool)

    validityfunc = lambda dof: zops_sp.check_spatialProj_incLags_validity(dof, include, O_quad, gradZTT)
    # validityfunc = lambda dof: check_Msparse_spatialProj_incLags_validity(dof, include, O_quad, gradZTT, chofac=ZTTchofac)
    
    mineigfunc = lambda dof: get_Msparse_inc_PD_ZTT_mineig(dof, include, O_quad, gradZTT, eigvals_only=False)

    S_gradZSS_S = np.zeros(2)
    # dgHfunc = lambda dof, dofgrad, dofHess, fSl, get_grad=True, get_Hess=True: get_inc_spatialProj_dualgradHess_fakeS_Msparse(dof, dofgrad, dofHess, include, O_lin, O_quad, gradZTS_S, gradZTT, fSl, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess)
    dgHfunc = lambda dof, grad, dofHess, fSl, get_grad=True, get_Hess=True: (
        dualgrad_sp.get_dual_and_derivatives(dof, grad, dofHess, include, O_lin, O_quad, gradZTS_S, gradZTT, 
                                          S_gradZSS_S, fSl, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess))


    Lags = np.zeros(len(gradZTT))

    Lags[1] = 1.0

    # while True:
    #     tmp = check_Msparse_spatialProj_Lags_validity(Lags, O_quad, gradZTT, chofac=ZTTchofac)
    #     if tmp>0:
    #         break
    #     print(tmp, flush=True)
    #     print('zeta', Lags[1])
    #     Lags[1] *= 1.5

    fSlist = []
    optLags = Lags[include]

    iternum = 0
    mindual = np.inf
    minLags = mingrad = minobj = None
    
    while True: #perhaps put in a convergence criterion later
        iternum += 1

        optLags, optgrad, optdual, optobj, fSlist = fakeS_single_restart_Newton(optLags, dgHfunc, validityfunc, mineigfunc, fSlist=fSlist, opttol=opttol, gradConverge=gradConverge, singlefS=singlefS, fakeSratio=fakeSratio, iter_period=iter_period)

        print('at dimension reduction iteration #', iternum, 'optdual and norm(optgrad) are', optdual, la.norm(optgrad))
        if optdual<mindual:
            mindual = optdual
            minobj = optobj
            minLags = optLags.copy()
            mingrad = optgrad.copy()
        print('the tightest dual bound found so far is', mindual)
        print('number of projection constraints', len(Pdatalist))
        if not (outputFunc is None):
            outputFunc(optLags, optgrad, optdual, optobj)
            print('minimum found so far')
            outputFunc(minLags, mingrad, mindual, minobj)

        ####adjust list of projection constraints
        ZTT = O_quad.copy()
        ZTS_S = O_lin.copy()
        for i in range(len(optLags)):
            ZTT += optLags[i] * gradZTT[i]
            ZTS_S += optLags[i] * gradZTS_S[i]

        if len(Pdatalist)-1>Pnum: #reduce Plist down to global constraint + one general constraint
            print('reduce Pdatalist')
            print('len optLags', len(optLags), 'len gradZTT, gradZTS_S', len(gradZTT), len(gradZTS_S))
            
            Pdatalist_new, optLags_new = reduce_Plist(Pdatalist, optLags) #dual space reduction
            while len(Pdatalist_new)>2:
                Pdatalist_new, optLags_new = reduce_Plist(Pdatalist_new, optLags_new)

            optLags = optLags_new
            Pdatalist = Pdatalist_new
            GinvdagPdaglist = []
            UPlist = []
            for i in range(len(Pdatalist)):
                P = sp.coo_matrix((Pdatalist[i], (Pstruct.row,Pstruct.col))).tocsc()
                GinvdagPdag = Ginv.conj().T @ P.conj().T
                GinvdagPdaglist.append(GinvdagPdag)
                UPlist.append(Ginv.conj().T @ GinvdagPdag.conj().T / np.conj(chi) - GinvdagPdag.conj().T)

            gradZTT = get_Msparse_gradZTT(UPlist)
            gradZTS_S = get_Msparse_gradZTS_S(Si, GinvdagPdaglist)

        ##################get new projection constraint#####################
        optGT = sp.linalg.spsolve(ZTT, ZTS_S)
        optT = Ginv @ optGT
        UdagT = optT/chi - optGT
        
        violation = outer_sparse_Pstruct(np.conj(Si), optT, Pstruct) - outer_sparse_Pstruct(np.conj(UdagT), optT, Pstruct) #off-diagonal P, violation generalized from point-wise multiplication to outer product

        for i in range(len(fSlist)): #contribution to violation from fake Source terms
            ZTTinvfS = sp.linalg.spsolve(ZTT, fSlist[i])
            GinvZTTinvfS = Ginv @ ZTTinvfS
            violation -= (1.0/np.conj(chi)) * outer_sparse_Pstruct(np.conj(GinvZTTinvfS), GinvZTTinvfS, Pstruct) - outer_sparse_Pstruct(np.conj(ZTTinvfS), GinvZTTinvfS, Pstruct)

        ###evaluate direction for increasing mineig of ZTT
        try:
            eigw, eigv = sp.linalg.eigsh(ZTT, k=1, sigma=0.0, which='LM', return_eigenvectors=True)
        except BaseException as err:
            print('encountered error in sparse eigenvalue evaluation', err)
            eigw, eigv = la.eigh(ZTT.toarray())

        eigw = eigw[0]
        eigv = eigv[:,0]

        Ginveigv = Ginv @ eigv
        mineigfac = outer_sparse_Pstruct(np.conj(Ginveigv), Ginveigv, Pstruct)/np.conj(chi) - outer_sparse_Pstruct(np.conj(eigv), Ginveigv, Pstruct)

        
        ###set the entries of the new projection matrix Pij so we align the gradients of mineig ZTT and the Sym(UP) constraint

        mineigfac_phase = mineigfac / (np.abs(mineigfac))
        Laggradfac_phase = -violation / np.abs(violation) #minus sign since we are doing dual minimization

        Pdata_new = np.conj(mineigfac_phase + Laggradfac_phase)
        Pdata_new *= np.abs(np.real(-Pdata_new * violation)) #scale with real part of Lagrangian gradient after rotation in phase

        print('norm of new projection matrix before orthogonalization', la.norm(Pdata_new))

        #orthogonalize against prior projections
        for i in range(len(Pdatalist)):
            Pdata_new -= np.vdot(Pdatalist[i], Pdata_new) * Pdatalist[i]

        print('norm of new projection matrix after orthogonalization', la.norm(Pdata_new))

        #update all of the lists and dofs for next iteration round
        Pdata_new /= la.norm(Pdata_new) #normalize
        P_new = sp.coo_matrix((Pdata_new, (Pstruct.row,Pstruct.col))).tocsc()
        GinvdagPdag = Ginv.conj().T @ P_new.conj().T
        UP_new = Ginv.conj().T @ GinvdagPdag.conj().T / np.conj(chi) - GinvdagPdag.conj().T

    
        #update the dual optimization parameters
        Pdatalist.append(Pdata_new)
        GinvdagPdaglist.append(GinvdagPdag)
        UPlist.append(UP_new)

        gradZTT.extend([(UP_new+UP_new.conj().T)/2, (UP_new-UP_new.conj().T)/2j])
        gradZTS_S.append(GinvdagPdag @ Si / 2)
        gradZTS_S.append(1j*gradZTS_S[-1])
        S_gradZSS_S = np.zeros(len(gradZTT))

        include = np.ones(len(gradZTT), dtype=bool)
        
        optLags_new = np.zeros(len(optLags)+2)
        optLags_new[:-2] = optLags[:]
        optLags = optLags_new

        # ZTTchofac = Cholesky_analyze_ZTT(O_quad, gradZTT)
        validityfunc = lambda dof: zops_sp.check_spatialProj_incLags_validity(dof, include, O_quad, gradZTT)
        # validityfunc = lambda dof: check_Msparse_spatialProj_incLags_validity(dof, include, O_quad, gradZTT, chofac=ZTTchofac)
        #validityfunc = lambda dof: check_Msparse_spatialProj_incLags_validity(dof, include, O_quad, gradZTT, chofac=None)
    
        mineigfunc = lambda dof: get_Msparse_inc_PD_ZTT_mineig(dof, include, O_quad, gradZTT, eigvals_only=False)

    
        # dgHfunc = lambda dof, dofgrad, dofHess, fSl, get_grad=True, get_Hess=True: get_inc_spatialProj_dualgradHess_fakeS_Msparse(dof, dofgrad, dofHess, include, O_lin, O_quad, gradZTS_S, gradZTT, fSl, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess)
        dgHfunc = lambda dof, grad, dofHess, fSl, get_grad=True, get_Hess=True: (
            dualgrad_sp.get_dual_and_derivatives(dof, grad, dofHess, include, O_lin_S, O_quad, gradZTS_S, gradZTT, 
                                          S_gradZSS_S, fSl, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess))

        

if __name__ == '__main__':
    import sys 
    import pandas as pd 
    sys.path.append('../')
    sys.path.append('../../')

    from Optimization.fakeSource_with_restart_singlematrix import fakeS_single_restart_Newton as Newton

    # Example for single material (sparse, using multimat code)
    # This does not yet support multimat constraints! A bit tricky to do. 
    import dualbound.Lagrangian.multimat.zops_multimat_sp as zmultimat
    from Examples.MultiMat.general_multimat.DipoleOutside.sp_mmms_do import sp_get_multimat_multifreq_DO_bound
    
    # We need to first load the lags from the global run (it needs to be run first!)
    # This is because gcd_sparse expects a starting point and doesn't know how to get it
    # This is done because every application has a different way to get the positive definite point 
    # so this general method just assumes you have it ahead of time.
    # I've specified a few example parameters. Check out Raman/TM/raman_center/bounds/run_raman_gcd.py 
    # for how you would integrate this into submission scripts. That file also has an example savefunc
    # that saves npys of lags. In this example, I just don't save any of the results.

    bounds = pd.read_csv('../../Examples/MultiMat//general_multimat/DipoleOutside/results/sp_do_results.csv', sep=';', index_col=False, on_bad_lines='warn')
    bounds = bounds[(bounds['wavelengths'] == '[1.0]') & (bounds['nprojx'] == 1.0) & (bounds['nprojy'] == 1.0) & 
                    (bounds['chis'] == f'[(4+0.1j)]') & (bounds['dx'] == 0.4) & (bounds['dy'] == 0.4) &  
                    (bounds['dist'] == 0.4) & (bounds['gpr'] == 40)]
    global_lags = np.array(bounds['lags'].values[0].strip('[]').split(', '), dtype=float)
    # This part is just to get the parameters and use the real code for calculating the Green's functions
    # You have to go to sp_mmms_do and uncomment the indicated line to return the right things 
    # and not actually run the optimization.
    # The parameters below are just for the example, they would be defined individually in your run.
    # In reality you would do something like Raman/TM/raman_center/bounds/run_raman_gcd.py
    chiList = [4+0.1j]
    wvlgthList = [1.0]
    nmat = 1 
    cons, type_o, obj, dist, design_x, design_y, gpr = [1,1], 'rect', 'ABS', 0.4, 0.2, 0.2, 40
    NProjx, NProjy, pml_sep, pml_thick = 1, 1, 0.5, 0.5 

    # Don't run the optimization! Just return these things manually for this example. 
    Ginvlist, chilist, nsource, nmat, S, dualconst, O_quad, O_lin_S = sp_get_multimat_multifreq_DO_bound(chiList, wvlgthList, nmat, cons, type_o, obj, dist, design_x, design_y, gpr, 
                                       NProjx, NProjy, pml_sep, pml_thick)
    G1inv = Ginvlist[0]
    # We need a get_gradZTT function that isn't like the previous ones. It ONLY takes a Plist, and it actually
    # just returns the next component of gradZTT given a new Plist (not the whole Plist). 
    # This is why I wrote a new function to do this, which is simpler than the overall get_gradZTT function
    get_gradZTT = lambda Plist : zmultimat.get_gradZTT_real_mp1(Plist, nmat, nsource, chilist, Ginvlist)

    # same for get_gradZTS 
    get_gradZTS_S = lambda Plist : zmultimat.get_gradZTS_S_real_mp1(S, Plist, Ginvlist, nmat, nsource)

    # Now we need a heuristic to choose the next vectors. I've written an example one already, 
    # cloaking should just work out of the box. 
    get_new_vecs = lambda fSlist, N, cclasses, get_gradZTT, get_gradZTS_S, GT, ZTT, niters : zmultimat.get_new_vecs_heuristic(G1inv, chiList[0], fSlist, N, cclasses, get_gradZTT, get_gradZTS_S, GT, ZTT, niters, S)

    # define the optfunc it'll use
    opt_func = lambda Lags_include, dgfunc, validityfunc, mineigfunc, fSlist : Newton(Lags_include, dgfunc, validityfunc, mineigfunc, fSlist, opttol=1e-4, fakeSratio=1e-3, iter_period=80, gradConverge=False)
    saveFunc = None 
    
    # This parameter is imoprtant! The higher it is, the slower your calculation will be but 
    # you will likely converge faster. It is basically the parameter that mediates how local the constraints
    # will get before a reduction. 10-20 is an ok number, but bring this as high as you can. 
    # Pnum should always be less than the number of constraints you can run with the old algorithm
    # This is because each iteration of GCD will add constraints up to Pnum number, so for example if nproj=100 is too slow, don't set Pnum=100.
    Pnum = 2
    N = Ginvlist[0].shape[0]
    cclasses=1
    # gcd_sparse(N, cclasses, get_gradZTT, get_gradZTS_S, dualconst, opt_func, global_lags, O_quad, O_lin_S, saveFunc, Pnum, maxiternum=40, saveint=1, get_new_vecs=get_new_vecs)
    # exit()

    chi = chiList[0]
    G1inv = sp.csc_array(G1inv)
    dual_space_reduction_iteration_Msparse_align_mineig_maxviol(chi, S, G1inv, O_lin_S, O_quad, Pstruct=None, P0phase=1.0+0j, Pnum=1, dualconst=0.0, opttol=1e-2, fakeSratio=1e-2, gradConverge=False, singlefS=False, iter_period=20, outputFunc=None)



    # Example for Raman
    # import dualbound.Lagrangian.raman.zops_raman_TM as zraman
    # from Examples.raman.TM.raman_center.bounds.raman_bounds_TM import get_raman_bound as rb 

    # # Load data and parameters for run 
    # chi = -10+0.1j
    # alpha = 1 
    # wv1 = 1
    # wv2 = 1
    # r_x = 0.4
    # r_y = 0.4
    # r_i = 0.1
    # gpr = 40
    # nproj = 1

    # bounds = pd.read_csv('../../Examples/raman/TM/raman_center/bounds/results/raman.csv', sep=';', index_col=False, on_bad_lines='warn')
    # bounds = bounds[(bounds['wavelengths'] == '[1.0, 1.0]') & (bounds['nproj'] == 1.0) & (
    #                  bounds['rx'] == r_x) & (bounds['ry'] == r_y) & (bounds['ri'] == 0.1) & (
    #                  bounds['chis'] == f'[{chi}, {chi}]') & (bounds['alpha'] == 1) & (bounds['gpr'] == 40) & (bounds['include_code'] == 11111111)]
    
    # Lags = np.array(bounds['lags'].values[0].strip('[]').split(', '), dtype=float)
    # print(f"bound: {bounds['bound'].values[0]}")
    # print(f"lags: {Lags}")

    # Ac, U1, U2, G1rd, G2dr, alpha, p, delta, S1, O_quad, O_lin_S, dualconst, gradZTT, gradZTS_S = rb(chi, chi, alpha, wv1, wv2, r_x, r_y, r_i, gpr, nproj, 
    #                 opttol=1e-4, fakeSratio=1e-3, iter_period=80, opttype='newton', 
    #                 pml_sep=0.5, pml_thick=0.5, circle=True, include_code=11111111, TESTS={'plotting': False, 'one_pixel': False, 'just_mask': False}, init_lags=None)

    # N = U1.shape[0]
    # cclasses = 4
    # get_gradZTT = lambda Plist : zraman.get_gradZTT_real_mp1(Ac, U1, U2, G1rd, G2dr, alpha, Plist)
    # get_gradZTS_S = lambda Plist : zraman.get_gradZTS_S_real_mp1(Ac, p, delta, alpha, G2dr, S1, Plist)
    # optfunc = lambda Lags_include, dgfunc, validityfunc, mineigfunc, fSlist : Newton(Lags_include, dgfunc, validityfunc, mineigfunc, fSlist, opttol=1e-3, fakeSratio=1e-3, iter_period=80, gradConverge=False)

    # gcd_dense(N, cclasses, get_gradZTT, get_gradZTS_S, dualconst, optfunc, Lags, O_quad, O_lin_S, None, 1)


