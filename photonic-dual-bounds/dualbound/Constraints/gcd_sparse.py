import numpy as np 
import scipy.sparse as sp
import sys 
sys.path.append('../../')
import dualbound.Lagrangian.zops_utils.zops_sparse as zops_sp
import dualbound.Lagrangian.dualgrad_sparse as dualgrad_sp
import dualbound.Lagrangian.zops_utils.zops_sparse_all as zops_sp_all
import dualbound.Lagrangian.dualgrad_sparse_all as dualgrad_sp_all
from contextlib import suppress
import scipy.linalg as la
import scipy.sparse.linalg as spla
           
def orthogonalize_vecs_get_new_P(Plist, new_vec):
    nv = new_vec.copy()
    for i in range(len(Plist)):
        v = Plist[i].diagonal()
        nv -= (new_vec.T.conjugate() @ v / (v.conjugate() @ v)) * v
    return sp.diags(nv, 0)

# next task: this function generates its own Plist, but we need to be able to pass it and lags (or not)
def gcd_sparse_all_with_inequality(N, get_init_gradZTT, get_gradZTT, get_init_gradZTS_S, get_gradZTS_S, dualconst, opt_func, starting, O_quad, O_lin_S, 
                   saveFunc, Pnum, maxiternum, saveint, get_new_vecs, extra_gradZTS_S=None, extra_lin_lags=None):
    fSlist = [] 

    if starting[0] == False:
        start_lags = starting[1]
        non_opt_lags = starting[2] # Can pass an empty array if no non_opt_lags
        total_lags = np.append(start_lags, non_opt_lags)
        Plist = [-1j*sp.eye(N, dtype=complex), sp.eye(N, dtype=complex)] 
        include = np.ones(len(Plist)+len(non_opt_lags), dtype=bool)

        # Check if non_opt_lags are inequality constraints
        inequality = np.append(np.zeros(len(start_lags), dtype=bool), np.ones(len(non_opt_lags), dtype=bool)) if starting[5] else None 
        optLags = total_lags[include]

    elif starting[0] == True:
        Lags = starting[1]
        non_opt_lags = [0]*starting[2]
        include = starting[3]
        Plist = starting[4]
        optLags = Lags[include]
        inequality = np.append(np.zeros(len(include)-len(non_opt_lags), dtype=bool), np.ones(len(non_opt_lags), dtype=bool)) if starting[5] else None 
        
    optgrad = np.zeros(len(optLags))    
    dofHess = np.zeros((len(optLags), len(optLags)))
    iternum = 0 
    gradZTT = get_init_gradZTT(Plist) # if we want to include any extra constraints that will be optimized but will not be involved in GCD 
                                      # iteration, then get_init_gradZTT \neq get_gradZTT
    gradZTS_S = get_init_gradZTS_S(Plist)
    S_gradZSS_S = np.zeros(len(optLags)) # Any <S|gradZSS|S> terms should have been optimized away 
    ZTTchofac = zops_sp_all.Cholesky_analyze_ZTT(O_quad, gradZTT)
    
    validityfunc = lambda dof: zops_sp_all.check_spatialProj_incLags_validity(dof, include, O_quad, gradZTT, ZTTchofac)
    mineigfunc = lambda dof, v0=None: zops_sp_all.get_inc_ZTT_mineig(dof, include, O_quad, gradZTT, eigvals_only=False, v0=v0)
    dgHfunc = lambda dof, grad, dofHess, fSl, get_grad=True, get_Hess=True: (
            dualgrad_sp_all.get_dual_and_derivatives(dof, grad, dofHess, include, O_lin_S, O_quad, gradZTS_S, gradZTT, 
                                S_gradZSS_S, fSl, ZTTchofac, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess,
                                extra_gradZTS_S=extra_gradZTS_S, extra_lin_lags=extra_lin_lags)) 

    if starting[0] == False:
        while not validityfunc(optLags) and np.sum(optLags) < 1e6:
            optLags *= 1.5
            print(f'1.5* optLags: {optLags}')
    else: 
        assert(validityfunc(optLags))

    optdual = dgHfunc(optLags, optgrad, dofHess, [], True, True)
    print(f'initial dualval: {optdual}')
    with suppress(ZeroDivisionError): print(f'initial dualval/dualconst: {optdual/dualconst}')
    
    while True:
        iternum += 1
        print(f'GCD Iteration #{iternum}')
        print(optLags)
        # input()
        optLags, optgrad, optdual, optval, fSlist = opt_func(optLags, dgHfunc, validityfunc, mineigfunc, fSlist, inequality) # run opt
        # fSlist = []
        # dualgrad_sp_all.dual_debugtool(optLags, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, dualconst, ZTTchofac)
        if iternum % saveint == 0 and not saveFunc is None:
            saveFunc(iternum, optdual, dualconst, optLags, optgrad, Plist)

        if iternum >= maxiternum: 
            print('Max iterations reached')
            break

        print(f'dualval: {optdual}')
        with suppress(ZeroDivisionError): print(f'dualval/dualconst: {optdual/dualconst}')
        print(f'number of projection constraints {len(Plist)}')

        ZTT = zops_sp_all.get_ZTT(optLags, O_quad, gradZTT)
        ZTS_S = zops_sp_all.get_ZTS_S(optLags, O_lin_S, gradZTS_S)
        S_ZSS_S = np.zeros(len(gradZTT))
        GT = spla.spsolve(ZTT, ZTS_S)

        new_vec = get_new_vecs(GT, fSlist, ZTT)
        newP = orthogonalize_vecs_get_new_P(Plist, new_vec) 
        Plist = np.append([newP], Plist)

        optLags = np.append([0], optLags) # add newP lagrange multiplier (0)
        gradZTT = np.append(get_gradZTT([newP]), gradZTT, axis=0)
        gradZTS_S = np.append(get_gradZTS_S([newP]), gradZTS_S, axis=0)
        S_gradZSS_S = np.zeros(len(optLags))
        include = np.ones(len(optLags), dtype=bool)
        optgrad = np.zeros(len(optLags))    
        dofHess = np.zeros((len(optLags), len(optLags)))
        inequality = np.append([False], inequality) if starting[5] else None 

        # dualgrad_sp_all.dual_debugtool(optLags, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, dualconst, ZTTchofac)

        if len(Plist) > Pnum: 
            print('reducing Plist')
            print(f'len optLags: {len(optLags)}, len gradZTT, gradZTS_S: {len(gradZTT)}, {len(gradZTS_S)}')

            newP = sp.dia_matrix((N,N), dtype=complex)
            for i in range(len(Plist)):
                newP += optLags[i] * Plist[i]

            old_Plist_length = len(Plist)
            Plist = [newP]
            gradZTT = get_init_gradZTT(Plist) 
            gradZTS_S = get_init_gradZTS_S(Plist)
            S_gradZSS_S = np.zeros(len(Plist)+len(non_opt_lags))
            include = np.ones(len(Plist)+len(non_opt_lags), dtype=bool)
            inequality = np.append(np.zeros(len(Plist), dtype=bool), np.ones(len(non_opt_lags), dtype=bool)) if starting[5] else None 
            optLags = np.append(np.ones(len(Plist)), optLags[old_Plist_length:])

        validityfunc = lambda dof: zops_sp_all.check_spatialProj_incLags_validity(dof, include, O_quad, gradZTT, ZTTchofac)
        mineigfunc = lambda dof, v0=None: zops_sp_all.get_inc_ZTT_mineig(dof, include, O_quad, gradZTT, eigvals_only=False, v0=v0)
        dgHfunc = lambda dof, grad, dofHess, fSl, get_grad=True, get_Hess=True: (
                dualgrad_sp_all.get_dual_and_derivatives(dof, grad, dofHess, include, O_lin_S, O_quad, gradZTS_S, gradZTT, 
                                S_gradZSS_S, fSl, ZTTchofac, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess,
                                extra_gradZTS_S=extra_gradZTS_S, extra_lin_lags=extra_lin_lags)) 
    
    return Plist, optLags, include, optgrad, optdual, gradZTT, gradZTS_S, ZTTchofac

def gcd_sparse_all(N, get_gradZTT, get_gradZTS_S, dualconst, opt_func, starting, O_quad, O_lin_S, 
                   saveFunc, Pnum, maxiternum, saveint, get_new_vecs, extra_gradZTS_S=None, extra_lin_lags=None):
    fSlist = [] 

    if starting[0] == False:
        start_lags = starting[1]
        # alpha = -start_lags[0::2]*1j + start_lags[1::2] # may need to change this depending on the global_lags format, Asym, Sym order etc
        
        # Plist = [sp.eye(N, dtype=complex) * alpha[i] for i in range(Ngcd)]
        Plist = [-1j*sp.eye(N, dtype=complex), sp.eye(N, dtype=complex)] 
        include = np.ones(len(Plist), dtype=bool)
        optLags = start_lags[include]

    elif starting[0] == True:
        Lags = starting[1]
        include = starting[2]
        Plist = starting[3]
        optLags = Lags[include]
        
    optgrad = np.zeros(len(optLags))    
    dofHess = np.zeros((len(optLags), len(optLags)))
    iternum = 0 
    gradZTT = get_gradZTT(Plist)
    gradZTS_S = get_gradZTS_S(Plist)
    S_gradZSS_S = np.zeros(len(optLags)) # Any <S|gradZSS|S> terms should have been optimized away 
    ZTTchofac = zops_sp_all.Cholesky_analyze_ZTT(O_quad, gradZTT)
    
    validityfunc = lambda dof: zops_sp_all.check_spatialProj_incLags_validity(dof, include, O_quad, gradZTT, ZTTchofac)
    mineigfunc = lambda dof, v0=None: zops_sp_all.get_inc_ZTT_mineig(dof, include, O_quad, gradZTT, eigvals_only=False, v0=v0)
    dgHfunc = lambda dof, grad, dofHess, fSl, get_grad=True, get_Hess=True: (
            dualgrad_sp_all.get_dual_and_derivatives(dof, grad, dofHess, include, O_lin_S, O_quad, gradZTS_S, gradZTT, 
                                S_gradZSS_S, fSl, ZTTchofac, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess,
                                extra_gradZTS_S=extra_gradZTS_S, extra_lin_lags=extra_lin_lags)) 

    if starting[0] == False:
        while not validityfunc(optLags) and np.sum(optLags) < 1e6:
            optLags *= 1.5
            print(f'1.5* optLags: {optLags}')
    else: 
        assert(validityfunc(optLags))
    
    optdual = dgHfunc(optLags, optgrad, dofHess, [], True, True)
    print(f'initial dualval: {optdual}')
    with suppress(ZeroDivisionError): print(f'initial dualval/dualconst: {optdual/dualconst}')
    
    while True:
        iternum += 1
        print(f'GCD Iteration #{iternum}')
        optLags, optgrad, optdual, optval, fSlist = opt_func(optLags, dgHfunc, validityfunc, mineigfunc, fSlist) # run opt
        # fSlist = []
        dualval, dualconst, constant_part, Lag_quadratic_part, O_quad_part, partial_quadratic_lags, partial_constant, T = dualgrad_sp_all.dual_debugtool(optLags, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, dualconst, ZTTchofac)
        # return Plist, optLags, include, optgrad, optdual, gradZTT, gradZTS_S, ZTTchofac
        if iternum % saveint == 0 and not saveFunc is None:
            saveFunc(iternum, dualval, dualconst, optLags, optgrad, Plist)

        if iternum >= maxiternum: 
            print('Max iterations reached')
            break

        print(f'dualval: {optdual}')
        with suppress(ZeroDivisionError): print(f'dualval/dualconst: {optdual/dualconst}')
        print(f'number of projection constraints {len(Plist)}')

        ZTT = zops_sp_all.get_ZTT(optLags, O_quad, gradZTT)
        ZTS_S = zops_sp_all.get_ZTS_S(optLags, O_lin_S, gradZTS_S)
        S_ZSS_S = np.zeros(len(gradZTT))
        GT = spla.spsolve(ZTT, ZTS_S)

        new_vec = get_new_vecs(GT, fSlist, ZTT)
        newP = orthogonalize_vecs_get_new_P(Plist, new_vec) 
        Plist = np.append([newP], Plist)

        optLags = np.append([0], optLags) # add newP lagrange multiplier (0)
        gradZTT = np.append(get_gradZTT([newP]), gradZTT, axis=0)
        gradZTS_S = np.append(get_gradZTS_S([newP]), gradZTS_S, axis=0)
        S_gradZSS_S = np.zeros(len(optLags))
        include = np.ones(len(optLags), dtype=bool)
        optgrad = np.zeros(len(optLags))    
        dofHess = np.zeros((len(optLags), len(optLags)))

        # dualgrad_sp_all.dual_debugtool(optLags, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, dualconst, ZTTchofac)

        if len(Plist) > Pnum: 
            print('reducing Plist')
            print(f'len optLags: {len(optLags)}, len gradZTT, gradZTS_S: {len(gradZTT)}, {len(gradZTS_S)}')

            newP = sp.dia_matrix((N,N), dtype=complex)
            for i in range(len(Plist)):
                newP += optLags[i] * Plist[i]

            Plist = [newP]
            gradZTT = get_gradZTT(Plist) 
            gradZTS_S = get_gradZTS_S(Plist)
            S_gradZSS_S = np.zeros(len(Plist))
            include = np.ones(len(Plist), dtype=bool)
            optLags = np.ones(len(include))


        validityfunc = lambda dof: zops_sp_all.check_spatialProj_incLags_validity(dof, include, O_quad, gradZTT, ZTTchofac)
        mineigfunc = lambda dof, v0=None: zops_sp_all.get_inc_ZTT_mineig(dof, include, O_quad, gradZTT, eigvals_only=False, v0=v0)
        dgHfunc = lambda dof, grad, dofHess, fSl, get_grad=True, get_Hess=True: (
                dualgrad_sp_all.get_dual_and_derivatives(dof, grad, dofHess, include, O_lin_S, O_quad, gradZTS_S, gradZTT, 
                                S_gradZSS_S, fSl, ZTTchofac, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess,
                                extra_gradZTS_S=extra_gradZTS_S, extra_lin_lags=extra_lin_lags)) 
        

    # dualgrad_sp_all.dual_debugtool(optLags, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, dualconst, ZTTchofac)
    return Plist, optLags, include, optgrad, optdual, gradZTT, gradZTS_S, ZTTchofac

def gcd_sparse(N, Nconstr, get_init_gradZTT, get_init_gradZTS_S, get_gradZTT, get_gradZTS_S, dualconst, opt_func, 
               global_lags, O_quad, O_lin_S, saveFunc, Pnum, maxiternum, saveint, get_new_vecs):
    Ngcd, Nextra = Nconstr['gcd_constraints'], Nconstr['extra_constraints']
    include = np.ones(Ngcd + Nextra, dtype=bool)
    Lags = np.append(np.ones(Ngcd), np.zeros(Nextra))
    fSlist = [] 
    optLags = Lags[include]

    # Need to convert global with imag, real to global real with one Lagrange multiplier. That is, take 2 multipliers -> 1
    alpha = -global_lags[0::2]*1j + global_lags[1::2] # may need to change this depending on the global_lags format, Asym, Sym order etc
    Plist = [sp.eye(N, dtype=complex) * alpha[i] for i in range(Ngcd)] 
    
    optgrad = np.zeros(len(optLags))    
    dofHess = np.zeros((len(optLags), len(optLags)))
    iternum = 0 
    gradZTT = get_init_gradZTT(Plist[0])
    gradZTS_S = get_init_gradZTS_S(Plist[0])
    S_gradZSS_S = np.zeros(len(optLags)) # Any <S|gradZSS|S> terms should have been optimized away 
    print(f'Density of gradZTT[0]: {zops_sp.density(gradZTT[0])}')
    
    validityfunc = lambda dof: zops_sp.check_spatialProj_incLags_validity(dof, include, O_quad, gradZTT)
    mineigfunc = lambda dof: zops_sp.get_inc_ZTT_mineig(dof, include, O_quad, gradZTT)
    dgHfunc = lambda dof, grad, dofHess, fSl, get_grad=True, get_Hess=True: (
        dualgrad_sp.get_dual_and_derivatives(dof, grad, dofHess, include, O_lin_S, O_quad, gradZTS_S, gradZTT, 
                                        S_gradZSS_S, fSl, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess))
    
    optdual = dgHfunc(optLags, optgrad, dofHess, [], True, True)

    print(f'initial dualval: {optdual}')
    with suppress(ZeroDivisionError): print(f'initial dualval/dualconst: {optdual/dualconst}')
    
    while True:
        iternum += 1
        print(f'GCD Iteration #{iternum}')

        optLags, optgrad, optdual, optval, fSlist = opt_func(optLags, dgHfunc, validityfunc, mineigfunc, fSlist) # run opt
        
        if iternum % saveint == 0 and not saveFunc is None:
            # saveFunc(iternum, optdual, dualconst, optLags, optgrad, get_Plist_from_vecs_sparse(Plist))
            pass 
        if iternum >= maxiternum: 
            print('Max iterations reached')
            break
        # dualgrad_sp.dual_debugtool(optLags, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, dualconst)
        # return optdual

        print(f'dualval: {optdual}')
        with suppress(ZeroDivisionError): print(f'dualval/dualconst: {optdual/dualconst}')
        print(f'number of projection constraints {len(Plist)}')

        ZTT = zops_sp.get_ZTT(optLags, O_quad, gradZTT)
        ZTS_S = zops_sp.get_ZTS_S(optLags, O_lin_S, gradZTS_S)
        S_ZSS_S = np.zeros(len(ZTT))
        GT = la.solve(ZTT, ZTS_S)

        new_vec = get_new_vecs(GT, fSlist, ZTT)
        newP = orthogonalize_vecs_get_new_P(Plist, new_vec) 
        Plist.extend([newP])

        optLags = np.append([0], optLags) # add newP lagrange multiplier (0)
        gradZTT = np.append(get_gradZTT(newP), gradZTT, axis=0)
        gradZTS_S = np.append(get_gradZTS_S(newP), gradZTS_S, axis=0)
        S_gradZSS_S = np.zeros(len(optLags))
        include = np.ones(len(optLags), dtype=bool)
        optgrad = np.zeros(len(optLags))    
        dofHess = np.zeros((len(optLags), len(optLags)))

        iternum += 1

        validityfunc = lambda dof: zops_sp.check_spatialProj_incLags_validity(dof, include, O_quad, gradZTT)
        mineigfunc = lambda dof: zops_sp.get_inc_ZTT_mineig(dof, include, O_quad, gradZTT)
        dgHfunc = lambda dof, grad, dofHess, fSl, get_grad=True, get_Hess=True: (
            dualgrad_sp.get_dual_and_derivatives(dof, grad, dofHess, include, O_lin_S, O_quad, gradZTS_S, gradZTT, 
                                            S_gradZSS_S, fSl, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess))
        

        # if len(Plist) > Pnum * cclasses: 
        #     print('reducing Plist')
        #     print('len optLags', len(optLags), 'len gradZTT, gradZTS_S', len(gradZTT), len(gradZTS_S))
        #     new_Plist = []
        #     Plist = np.array(Plist, dtype=object)

        #     assert cclasses == 1 # TODO(alessio): generalize this 
        #     newP = np.zeros((N,N), dtype=complex) #sp.csr_matrix((N, N), dtype=complex)
        #     for i in range(len(Plist)):
        #         newP += optLags[i] * Plist[i]
        #     newP = sp.csr_matrix(newP)

        #     new
        #     _Plist = [newP]
        #     # for i in range(cclasses):
        #         # new_Plist.append(np.sum(optLags[i::cclasses, None, None] * np.array(Plist[i::cclasses]), axis=0))   
        #     print(new_Plist)
        #     Plist = new_Plist 
        #     gradZTT = get_gradZTT(Plist) 
        #     gradZTS_S = get_gradZTS_S(Plist)
        #     S_gradZSS_S = np.zeros(len(Plist))
        #     include = np.ones(len(Plist), dtype=bool)
        #     optLags = np.ones(len(include))

        #     print(f'New bound: {optdual/dualconstnorm}')
        #     print(f'New lags: {optLags}')
        #     # input()

    dualgrad_sp.dual_debugtool(optLags, O_lin_S, O_quad, gradZTS_S, gradZTT, S_gradZSS_S, dualconst)
    return 0


# if zerod_inequality:
#     # It is not good to optimize like this, because adding the unoptimized inequality constraint 
#     # multiplier to the optimized values doesn't guarantee the result is positive definite
#     # In practice it is for some reason? But then the optimization breaks and doesn't move. 
#     # It might make sense to always have a fake source that adds a big value to the dual problem if lambda > 0
#     # then do not even bother with two stage optimization 
#     short_gradZTT = gradZTT[:-1]
#     short_gradZTS_S = gradZTS_S[:-1]
#     short_optLags = optLags[:-1]

#     validityfunc = lambda dof: zops_sp_all.check_spatialProj_incLags_validity(dof, include[:-1], O_quad, short_gradZTT, ZTTchofac)
#     mineigfunc = lambda dof, v0=None: zops_sp_all.get_inc_ZTT_mineig(dof, include[:-1], O_quad, short_gradZTT, eigvals_only=False, v0=v0)
#     dgHfunc = lambda dof, grad, dofHess, fSl, get_grad=True, get_Hess=True: (
#             dualgrad_sp_all.get_dual_and_derivatives(dof, grad, dofHess, include[:-1], O_lin_S, O_quad, short_gradZTS_S, short_gradZTT, 
#                                 S_gradZSS_S[:-1], fSl, ZTTchofac, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess,
#                                 extra_gradZTS_S=extra_gradZTS_S, extra_lin_lags=extra_lin_lags)) 
    
#     # optimize without the inequality constraint 
#     new_optLags, new_optgrad, optdual, optval, fSlist, zerod_inequality = opt_func(short_optLags, dgHfunc, 
#                                                                                     validityfunc, mineigfunc, fSlist, inequality=None)

#     # Add it back in to optLags so it can be optimized next loops 
#     optLags = np.append(new_optLags, optLags[-1])
#     optgrad = np.append(new_optgrad, optgrad[-1])