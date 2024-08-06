import numpy as np
import autograd.numpy as npa
import matplotlib.pyplot as plt
import ceviche
from ceviche import jacobian
from ceviche.constants import ETA_0, C_0, MU_0, EPSILON_0

def source_ldos(source, Ezs, dl, omegas, omega0, Qabs):
    ldos = 0
    Polefactor = 0
    for nn in range(len(Ezs)):
        Polefactor += -1j*np.exp(1j*(np.pi+nn*2.*np.pi)/(2.*len(Ezs)))
    for ii in range(len(Ezs)):
        ldos += dl**2 * 0.5 * npa.real(-1j*(np.exp(1j*(np.pi+2.*ii*np.pi)/(2.*len(Ezs)))/Polefactor*npa.sum(npa.conj(source) * Ezs[ii]))/omegas[ii] * omega0*(1+(1/2./Qabs)**2))
    return ldos

def source_abs(source, Ez, dl, omega_Qabs, eps):
    return -dl**2 * 0.5 * omega_Qabs * npa.sum(np.imag(eps) * npa.abs(Ez)**2)  * EPSILON_0

def scale_dof_to_eps(dof, epsmin, epsmax):
    return epsmin + dof * (epsmax-epsmin)

def eps_parametrization(dof, epsmin, epsmax, designMask, epsBkg):
    eps = scale_dof_to_eps(dof, epsmin, epsmax) * designMask + epsBkg * (1-designMask)
    return eps

def ldos_objective(dof, epsval, designMask, dl, source, sims, omegas, omega0, Qabs):
    Nx,Ny = source.shape
    dof = dof.reshape((Nx,Ny))
    epsBkg = np.ones((Nx,Ny), dtype=complex)
    eps = eps_parametrization(dof, 1.0, epsval, designMask, epsBkg)
    Ezs = []
    for ii in range(len(sims)):
        sims[ii].eps_r = eps
        _,_,Ez1 = sims[ii].solve(source)
        Ezs += [Ez1[:]]
    return source_ldos(source, Ezs, dl, omegas, omega0, Qabs)

def abs_objective(dof, epsval, designMask, dl, source, sim,omega_Qabs):
    Nx,Ny = source.shape
    dof = dof.reshape((Nx,Ny))
    epsBkg = np.ones((Nx,Ny), dtype=complex)
    eps = eps_parametrization(dof, 1.0, epsval, designMask, epsBkg)
    sim.eps_r = eps

    _,_,Ez = sim.solve(source)
    return source_abs(source, Ez, dl, omega_Qabs, eps)

def designdof_ldos_objective(designdof, designgrad, epsval, designMask, dl, source, omega, omega0, Num_Poles, Qabs, epsVac, Npml, opt_data):
    """
    optimization objective to be used with NLOPT
    opt_data is dictionary with auxiliary info such as current iteration number and output base
    """
    omegas = []
    sims = []
    for nn in range(Num_Poles):
        omegas += [omega * (1-np.exp(1j*(np.pi+nn*2.*np.pi)/(2.*Num_Poles))/2./Qabs)]
        sims += [ceviche.fdfd_ez(omegas[nn], dl, epsVac, [Npml,Npml])]
        
    Nx,Ny = source.shape
    dof = np.zeros((Nx,Ny))
    dof[designMask] = designdof[:]

    objfunc = lambda d: ldos_objective(d, epsval, designMask, dl, source, sims,omegas, omega0, Qabs)
    obj = objfunc(dof.flatten())

    opt_data['count'] += 1
    print('at iteration #', opt_data['count'], 'the ldos value is', obj, ' with enhancement', obj/opt_data['vac_ldos_Q'], flush=True)
    if opt_data['count'] % opt_data['output_base'] == 0:
        np.savetxt(opt_data['name']+'_dof'+str(opt_data['count'])+'.txt', designdof[:])

    if len(designgrad)>0:
        jac_objfunc = jacobian(objfunc, mode='reverse')
        fullgrad = jac_objfunc(dof.flatten())
        designgrad[:] = np.reshape(fullgrad, (Nx,Ny))[designMask]

    return obj

