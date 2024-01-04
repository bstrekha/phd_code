import numpy as np
import random
import autograd.numpy as npa
import matplotlib.pyplot as plt
import ceviche
from ceviche.constants import ETA_0, C_0
from dualbound.Maxwell.Yee_TE_FDFD_master import get_TE_dipole_field, get_TE_adjoint_field
from TE_ES_FD import get_ES_TE_dipole_field
import sys

def source_ldos(source, Ex, dl):
    return -dl**2 * 0.5 * np.real(np.sum(np.conj(source) * Ex))
    #minus sign due to e^{-iwt} convention for TE code

def scale_dof_to_eps(dof, epsmin, epsmax):
    return epsmin + dof * (epsmax-epsmin)

def eps_parametrization(dof, epsmin, epsmax, designMask, epsBkg):
    eps = scale_dof_to_eps(dof, epsmin, epsmax) * designMask + epsBkg * (1-designMask)
    return eps

def ldos_tot_objective(dof, epsval, designMask, dl, source, wavelength, Nx, Ny, Npmlx, Npmly, dx, dy, cxmin, cxmax, cymin, cymax, orient, amp, omega_Qabs, Qabs):
    Nx,Ny = source.shape
    try:
        dof = np.array(dof._value).reshape((Nx,Ny))
    except:
        dof = dof.reshape((Nx,Ny))
    epsBkg = np.ones((Nx,Ny), dtype=np.complex)
    eps = eps_parametrization(dof, 1.0, epsval, designMask, epsBkg)
    _, Extot, Eytot = get_TE_dipole_field(wavelength, Nx, Ny, Npmlx, Npmly, dx, dy, cxmin, cxmax, cymin, cymax, orient, amp, chigrid=eps-1, omega_Qabs=omega_Qabs)
    omega0 = 2*np.pi/wavelength
    if orient == 1:
        ldos_tot = -dl**2 * 0.5 * npa.real(npa.sum(npa.conj(source) * (Extot))/omega_Qabs * omega0*(1+(1/2./Qabs)**2))
    else:
        ldos_tot = -dl**2 * 0.5 * npa.real(npa.sum(npa.conj(source) * (Eytot))/omega_Qabs * omega0*(1+(1/2./Qabs)**2))
    return ldos_tot, Extot, Eytot

def ldos_sca_objective(dof, epsval, designMask, dl, source, wavelength, Nx, Ny, Npmlx, Npmly, dx, dy, cxmin, cxmax, cymin, cymax, orient, amp, omega_Qabs, Qabs):
    #This needs to be rethought so the gradient for this obj is calcualted correctly
    #it works for Q > 1e4. For smaller Q, I think the Evac term at complex freq causes issues for TE case
    #due to faster divergence than TM case.
    #Right now, this is essentially the same as ldos_tot_objective just with additional steps.
    Nx,Ny = source.shape
    try:
        dof = np.array(dof._value).reshape((Nx,Ny))
    except:
        dof = dof.reshape((Nx,Ny))
    epsBkg = np.ones((Nx,Ny), dtype=np.complex)
    eps = eps_parametrization(dof, 1.0, epsval, designMask, epsBkg)
    _, Extot, Eytot = get_TE_dipole_field(wavelength, Nx, Ny, Npmlx, Npmly, dx, dy, cxmin, cxmax, cymin, cymax, orient, amp, chigrid=eps-1, omega_Qabs=omega_Qabs)
    _, Exvac, Eyvac = get_TE_dipole_field(wavelength, Nx, Ny, Npmlx, Npmly, dx, dy, cxmin, cxmax, cymin, cymax, orient, amp, chigrid=None, omega_Qabs=omega_Qabs)
    Exsca = Extot - Exvac
    Eysca = Eytot - Eyvac
    omega0 = 2*np.pi/wavelength
    
    if orient == 1:
        ldos_sca = -dl**2 * 0.5 * np.real(np.sum(np.conj(source) * (Extot - Exvac))/omega_Qabs * omega0*(1+(1/2./Qabs)**2))
    else:
        ldos_sca = -dl**2 * 0.5 * np.real(np.sum(np.conj(source) * (Eytot - Eyvac))/omega_Qabs * omega0*(1+(1/2./Qabs)**2))
    if orient == 1:
        cur_ldos_vac = -dl**2 * 0.5 * np.real(np.sum(np.conj(source) * (Exvac))/omega_Qabs * omega0*(1+(1/2./Qabs)**2))
    else:
        cur_ldos_vac = -dl**2 * 0.5 * np.real(np.sum(np.conj(source) * (Eyvac))/omega_Qabs * omega0*(1+(1/2./Qabs)**2))
    print('ldos_sca', ldos_sca)
    print('num_ldos_vac', cur_ldos_vac)
    exact_vac_ldos_Q = np.arctan(2*Qabs)*omega0*(1 + (1.0/2/Qabs)**2)/(8*np.pi)
    print('exact_ldos_vac', exact_vac_ldos_Q)
    print('exact + sca', exact_vac_ldos_Q + ldos_sca)
    print('num + sca', cur_ldos_vac+ ldos_sca)
        
    #get electrostatic terms
    #need bigger grid for electrostatic solver which uses Dirichlet conditions
    Nxpad = 200
    Nypad = 200
    bigchigrid = np.zeros((Nx+2*Nxpad, Ny+2*Nypad))
    bigchigrid[Nxpad:Nxpad+Nx, Nypad:Nypad+Ny] = eps[:,:] - 1.0
#     plt.imshow(np.real(bigchigrid), cmap='Greys')
#     plt.savefig('check_bigchigrid_config.png')
    Extot0, Eytot0 = get_ES_TE_dipole_field(Nx + 2*Nxpad, Ny + 2*Nypad, dl, dl, cxmin + Nxpad, cxmax + Nxpad, cymin + Nypad, cymax + Nypad, pol=orient, amp=amp, chigrid=bigchigrid)
    Extot0 = Extot0[Nxpad:Nxpad+Nx, Nypad:Nypad+Ny] 
    Eytot0 = Eytot0[Nxpad:Nxpad+Nx, Nypad:Nypad+Ny]
    
    Exvac0, Eyvac0 = get_ES_TE_dipole_field(Nx + 2*Nxpad, Ny + 2*Nypad, dl, dl, cxmin + Nxpad, cxmax + Nxpad, cymin + Nypad, cymax + Nypad, pol=orient, amp=amp, chigrid=None)
    Exvac0 = Exvac0[Nxpad:Nxpad+Nx, Nypad:Nypad+Ny] 
    Eyvac0 = Eyvac0[Nxpad:Nxpad+Nx, Nypad:Nypad+Ny]
    
    prefactor = np.imag(omega_Qabs)/(np.real(omega_Qabs)**2 + np.imag(omega_Qabs)**2)
    sourceES = np.zeros_like(source)
    sourceESAmp = amp*dl*dl*(cxmax-cxmin)*(cymax-cymin) #the static dipole should have same amp as non-static
    if orient == 1:
        sourceES[cxmin:Nx//2, cymin:cymax] = sourceESAmp/dl/dl/(cymax-cymin)/(Nx//2 - cxmin)/2
        sourceES[Nx//2:cxmax, cymin:cymax] = -sourceESAmp/dl/dl/(cymax-cymin)/(cxmax - Nx//2)/2
    else:
        sourceES[cxmin:cxmax, cymin:Ny//2] = sourceESAmp/dl/dl/(cxmax-cxmin)/(Nx//2 - cxmin)/2
        sourceES[cxmin:cxmax, Ny//2:cymax] = -sourceESAmp/dl/dl/(cxmax-cxmin)/(cymax - Ny//2)/2
    
    if orient == 1:
        alpha_sca_ES = prefactor*np.real(np.sum(np.conj(sourceES) * Extot0) * dl * dl)
        alpha_vac_ES = prefactor*np.real(np.sum(np.conj(sourceES) * Exvac0) * dl * dl)
    else:
        alpha_sca_ES = prefactor*np.real(np.sum(np.conj(sourceES) * Eytot0) * dl * dl)
        alpha_vac_ES = prefactor*np.real(np.sum(np.conj(sourceES) * Eyvac0) * dl * dl)
        
    testSourceES = False
    if testSourceES:
        print('sourceES dipole signed:', np.sum(np.conj(sourceES)*np.abs(np.sign(sourceES))) * dl * dl, flush=True)
        print('sourceES dipole magnitude:', np.sum(np.conj(sourceES)*np.sign(sourceES)) * dl * dl, flush=True)
    
#     plt.imshow(np.real(Extot0), cmap='RdBu')
#     plt.savefig('checkEx_ES_check_config.png')
#     plt.imshow(np.real(Eytot0), cmap='RdBu')
#     plt.savefig('checkEy_ES_check_config.png')
    print('scaES', alpha_sca_ES)
    print('vacES', alpha_vac_ES)
    print('exact + sca + scaES + vacES', exact_vac_ldos_Q + ldos_sca + alpha_sca_ES + alpha_vac_ES)
    
    return ldos_sca + alpha_sca_ES + alpha_vac_ES, Extot, Eytot


def designdof_ldos_objective(designdof, designgrad, epsval, designMask, dl, source, opt_data, wavelength, Nx, Ny, Npmlx, Npmly, dx, dy, cxmin, cxmax, cymin, cymax, orient, amp, vac_ldos, omega_Qabs, Qabs):
    """
    optimization objective to be used with NLOPT
    opt_data is dictionary with auxiliary info such as current iteration number and output base
    """
    epsBkg = np.ones((Nx,Ny), dtype=np.complex)
    Nx,Ny = source.shape
    dof = np.zeros((Nx,Ny))
    dof[designMask] = designdof[:]
    
    only_sca = True
    if only_sca:
        objfunc = lambda d: ldos_sca_objective(d, epsval, designMask, dl, source, wavelength, Nx, Ny, Npmlx, Npmly, dx, dy, cxmin, cxmax, cymin, cymax, orient, amp, omega_Qabs, Qabs)
        obj, Ex, Ey = objfunc(dof.flatten())
        obj = obj + opt_data['vac_ldos_Q'] #add known analytical avg vac to the sca term
    else:
        objfunc = lambda d: ldos_tot_objective(d, epsval, designMask, dl, source, wavelength, Nx, Ny, Npmlx, Npmly, dx, dy, cxmin, cxmax, cymin, cymax, orient, amp, omega_Qabs, Qabs)
        obj, Ex, Ey = objfunc(dof.flatten())

    opt_data['count'] += 1
    print('at iteration #', opt_data['count'], 'the ldos value is', obj, ' with enhancement', obj/opt_data['vac_ldos_Q'], flush=True)
    if opt_data['count'] % opt_data['output_base'] == 0:
        np.savetxt(opt_data['name']+'_dof'+str(opt_data['count'])+'.txt', designdof[:])

    if len(designgrad)>0:
        epsBkg = np.ones((Nx,Ny), dtype=np.complex)
        omega0 = 2*np.pi/wavelength
        fullgrad = 0.5 * dl**2 * np.real(1j * (epsval - 1) * (Ex**2+Ey**2) * omega0*(1+(1/2./Qabs)**2))
        designgrad[:] = np.reshape(fullgrad, (Nx,Ny))[designMask]
        #gradient checks
        printGradChecks = False
        if printGradChecks:
            #theoretical gradient
            randInd = round(random.uniform(0,np.sum(designMask.flatten()) - 1))
            print('rand index:', randInd)
            theograd0 = fullgrad[designMask].flatten()[randInd]
            print("theograd", theograd0,flush=True)
            #numerical grad test
            deltadof = np.zeros((Nx,Ny)).flatten()
            designdeltadof = np.zeros(np.sum(designMask.flatten()))
            designdeltadof[randInd] = 1e-4
            deltadof[designMask.flatten()] = designdeltadof
            obj2, _, _ = objfunc(dof.flatten()+deltadof.flatten())
            obj3, _, _ = objfunc(dof.flatten()-deltadof.flatten())
            numgrad0 = (obj2-obj3)/np.linalg.norm(deltadof)/2
            print("numgrad",numgrad0,flush=True)
            print("|theo-num|/|theo|", abs((theograd0 - numgrad0)/theograd0),flush=True)
    return obj