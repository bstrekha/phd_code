import numpy as np
import random
import autograd.numpy as npa
import matplotlib.pyplot as plt
import ceviche
from ceviche import jacobian
from ceviche.constants import ETA_0, C_0, MU_0, EPSILON_0

def source_abs(source, Ez, dl, omega_Qabs, eps):
    return -dl**2 * 0.5 * omega_Qabs * npa.sum(np.imag(eps) * npa.abs(Ez)**2)  * EPSILON_0

def scale_dof_to_eps(dof, epsmin, epsmax):
    return epsmin + dof * (epsmax-epsmin)

def eps_parametrization(dof, epsmin, epsmax, designMask, epsBkg):
    eps = scale_dof_to_eps(dof, epsmin, epsmax) * designMask + epsBkg * (1-designMask)
    return eps

def ldos_sca_objective(dof, epsval, designMask, dl, source, omega0, Qabs, Npml):
    Nx,Ny = source.shape
    dof = dof.reshape((Nx,Ny))
    epsVac = np.ones((Nx,Ny), dtype=complex)
    eps = eps_parametrization(dof, 1.0, epsval, designMask, epsVac)
    
    omega_Qabs = omega0*(1 - 1j/2/Qabs) #ceviche has +iwt time convention
    
    simTot = ceviche.fdfd_ez(omega_Qabs, dl, epsVac, [Npml,Npml])
    simTot.eps_r = eps
    _,_,Eztot = simTot.solve(source)
    
    simVac = ceviche.fdfd_ez(omega_Qabs, dl, epsVac, [Npml,Npml])
    simVac.eps_r = epsVac
    _,_,Ezvac = simVac.solve(source)
    
    ldos_sca = dl**2 * 0.5 * npa.real(npa.sum(npa.conj(source) * (Eztot - Ezvac))/omega_Qabs * omega0*(1+(1/2./Qabs)**2))
#     ldos_vac_test = np.real(np.sum(np.conj(source)*Ezvac) / omega_Qabs / omega0 * (omega0**2+(omega0**2)*(1/2./Qabs)**2)) * 0.5 * dl**2
#     exact_vac_ldos = 1/4/np.pi/omega0*(omega0**2+(omega0**2)*(1/2./Qabs)**2)*np.arctan(2*Qabs)*np.sqrt(1.25663706e-6/ 8.85418782e-12)*np.sqrt(1.25663706e-6 * 8.85418782e-12)
#     print('INTERMEDIATE TEST: ')
#     print('numerical vacuum LDOS', ldos_vac_test)
#     print('exact vacuum LDOS', exact_vac_ldos)
    return ldos_sca

def ldos_tot_objective(dof, epsval, designMask, dl, source, omega0, Qabs, Npml):
    Nx,Ny = source.shape
    dof = dof.reshape((Nx,Ny))
    epsVac = np.ones((Nx,Ny), dtype=complex)
    eps = eps_parametrization(dof, 1.0, epsval, designMask, epsVac)
    
    omega_Qabs = omega0*(1 - 1j/2/Qabs)
    
    simTot = ceviche.fdfd_ez(omega_Qabs, dl, epsVac, [Npml,Npml])
    simTot.eps_r = eps
    _,_,Eztot = simTot.solve(source)

    ldos_tot = dl**2 * 0.5 * npa.real(npa.sum(npa.conj(source) * Eztot)/omega_Qabs * omega0*(1+(1/2./Qabs)**2))
    return ldos_tot

def abs_objective(dof, epsval, designMask, dl, source, sim,omega_Qabs):
    Nx,Ny = source.shape
    dof = dof.reshape((Nx,Ny))
    epsBkg = np.ones((Nx,Ny), dtype=complex)
    eps = eps_parametrization(dof, 1.0, epsval, designMask, epsBkg)
    sim.eps_r = eps

    _,_,Ez = sim.solve(source)
    return source_abs(source, Ez, dl, omega_Qabs, eps)

def designdof_ldos_objective(designdof, designgrad, epsval, designMask, dl, source, omega0, Qabs, epsVac, Npml, opt_data):
    """
    optimization objective to be used with NLOPT
    opt_data is dictionary with auxiliary info such as current iteration number and output base
    """
    
    Nx,Ny = source.shape
    dof = np.zeros((Nx,Ny))
    dof[designMask] = designdof[:]
    
    only_sca = False
    if only_sca:
        objfunc = lambda d: ldos_sca_objective(d, epsval, designMask, dl, source, omega0, Qabs, Npml)
        obj = objfunc(dof.flatten())
        obj = obj + opt_data['vac_ldos_Q']
    else:
        objfunc = lambda d: ldos_tot_objective(d, epsval, designMask, dl, source, omega0, Qabs, Npml)
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

def get_supercell_dof(supercell_dof, designdof, designMask, desxL, desyL, numCellsX, numCellsY, period_xPixels, period_yPixels):
    dof = designdof.reshape((period_xPixels, period_yPixels))
    for i in range(numCellsX):
        for j in range(numCellsY):
            supercell_dof[desxL + period_xPixels*i:desxL + period_xPixels*(i+1), desyL + period_yPixels*j:desyL + period_yPixels*(j+1)] = dof[:, :]
            
def ldos_tot_objective_periodic(dof, epsval, designMask, dl, source, omega0, Qabs, Npml, cx, cy, desxL, desyL, numCellsX, numCellsY, period_xPixels, period_yPixels):
#     plt.figure()
#     plt.imshow(dof)
#     plt.savefig('dof_in_ldos_tot_check.png')
    Nx,Ny = source.shape
    epsVac = np.ones((Nx,Ny), dtype=complex)
    eps = eps_parametrization(dof, 1.0, epsval, designMask, epsVac)
    
    omega_Qabs = omega0*(1 - 1j/2/Qabs)
    
    simTot = ceviche.fdfd_ez(omega_Qabs, dl, epsVac, [Npml,Npml])
    simTot.eps_r = eps
    _,_,Eztot = simTot.solve(source)

    ldos_tot = dl**2 * 0.5 * npa.real(npa.sum(npa.conj(source) * Eztot)/omega_Qabs * omega0*(1+(1/2./Qabs)**2))
    return ldos_tot, Eztot

def designdof_ldos_objective_periodic(designdof, designgrad, wavelength, epsval, designMask, dl, source, omega0, Qabs, epsVac, Npml, opt_data, cx, cy, desxL, desyL, numCellsX, numCellsY, period_xPixels, period_yPixels):
    """
    optimization objective to be used with NLOPT
    opt_data is dictionary with auxiliary info such as current iteration number and output base
    """
    
    Nx,Ny = source.shape
    dof = np.zeros((Nx,Ny))
    get_supercell_dof(dof, designdof, designMask, desxL, desyL, numCellsX, numCellsY, period_xPixels, period_yPixels)
    
    objfunc = lambda d: ldos_tot_objective_periodic(d, epsval, designMask, dl, source, omega0, Qabs, Npml, cx, cy, desxL, desyL, numCellsX, numCellsY, period_xPixels, period_yPixels)

    obj, Ez = objfunc(dof)
        
    opt_data['count'] += 1
    print('at iteration #', opt_data['count'], 'the ldos value is', obj, ' with enhancement', obj/opt_data['vac_ldos_Q'], flush=True)
    if opt_data['count'] % opt_data['output_base'] == 0:
        np.savetxt(opt_data['name']+'_dof'+str(opt_data['count'])+'.txt', dof[desxL: desxL + numCellsX*period_xPixels, desyL: desyL + numCellsY*period_yPixels].flatten()) #save only design part

#     if len(designgrad)>0:
#         jac_objfunc = jacobian(objfunc, mode='reverse')
#         fullgrad = jac_objfunc(designdof.flatten())
#         designgrad[:] = fullgrad[:]

    if len(designgrad)>0:
        epsBkg = np.ones((Nx,Ny), dtype=complex)
        fullgrad = -0.5 * dl**2 * np.real(1j * (epsval - 1) * (Ez**2) * omega0*(1+(1/2./Qabs)**2))/(MU_0*C_0**2)
        reducedgrad = np.zeros((period_xPixels, period_yPixels))
        for i in range(numCellsX):
            for j in range(numCellsY):
                reducedgrad[:, :] += fullgrad[desxL + period_xPixels*i:desxL + period_xPixels*(i+1), desyL + period_yPixels*j:desyL + period_yPixels*(j+1)]
        designgrad[:] = reducedgrad.flatten()
        #gradient checks
        printGradChecks = False
        if printGradChecks:
            #theoretical gradient
            randIndX = round(random.uniform(0,period_xPixels - 1))
            randIndY = round(random.uniform(0,period_yPixels - 1))
            print('rand x index:', randIndX)
            print('rand y index:', randIndY)
            theograd0 = reducedgrad[randIndX, randIndY]
            print("theograd", theograd0,flush=True)
            #numerical grad test
            deltadof = np.zeros((period_xPixels, period_yPixels))
            deltaChiMag = (random.random() + 1e-5)*1e-4
            deltadof[randIndX, randIndY] = deltaChiMag
            supercell_deltadof = np.zeros((Nx,Ny))
            get_supercell_dof(supercell_deltadof, deltadof, designMask, desxL, desyL, numCellsX, numCellsY, period_xPixels, period_yPixels)
            obj2, _ = objfunc(dof + supercell_deltadof)
            obj3, _ = objfunc(dof - supercell_deltadof)
            numgrad0 = (obj2-obj3)/deltaChiMag/2
            print("numgrad",numgrad0,flush=True)
            print("|theo-num|/|theo|", abs((theograd0 - numgrad0)/theograd0),flush=True) 
            #relative error should be small
    return obj