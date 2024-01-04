import numpy as np
import random
import matplotlib.pyplot as plt
from dualbound.Maxwell import TM_FDFD as TM
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def ext_objective(dof, chif, background_mask, chid, design_mask, Npml, dl, cx, omega0, Qabs, returngrad=False):
    nonpmlNx,nonpmlNy = design_mask.shape
    Nx = nonpmlNx + 2*Npml
    Ny = nonpmlNy + 2*Npml
    wv = 2*np.pi/omega0
    
    tot_chi = np.zeros((Nx, Ny), dtype=complex)
    dof_chi_nopml = np.zeros((nonpmlNx,nonpmlNy), dtype=complex)
    dof_chi_nopml[design_mask] = dof[:]*chid
    
    do_checks = False
    if do_checks:
        config = np.zeros((nonpmlNx,nonpmlNy))
        config[design_mask] = dof[:]
        config[background_mask] += 2.0
        plt.figure()
        plt.imshow(config)
        plt.savefig('check_design+background_in_ext_objective.png')
    
    tot_chi[Npml:-Npml, Npml:-Npml] = (background_mask[:,:]*chif) + dof_chi_nopml[:,:]
    Etot = TM.get_TM_linesource_field(wv, dl, Nx, Ny, cx, Npml, bloch_x=0.0, bloch_y=0.0, amp=1.0, Qabs=Qabs, chigrid=tot_chi)
    Z = 1.0
    Jg = -1j*omega0/Z*tot_chi*Etot
    Ei = TM.get_TM_linesource_field(wv, dl, Nx, Ny, cx, Npml, bloch_x=0.0, bloch_y=0.0, amp=1.0, Qabs=Qabs, chigrid=None)
    
    Pext = np.real(np.vdot(Ei, Jg)/2) * dl**2
    
    if returngrad:
        chid_mask = design_mask*chid
        
        fullgrad1 = np.conj(Ei[Npml:-Npml, Npml:-Npml]) * chid_mask * Etot[Npml:-Npml, Npml:-Npml] * omega0/1j/Z
        
        tildeJ = tot_chi*np.conj(Ei)*omega0/1j/Z
        G0inv = TM.get_TM_MaxwellOp(wv, dl, Nx, Ny, Npml, bloch_x=0, bloch_y=0, Qabs=Qabs)
        omega = omega0*(1 + 1j/2/Qabs)
        Ginv = G0inv + TM.get_diagM_from_chigrid(omega, tot_chi)
        E2 = spla.spsolve(Ginv, tildeJ.flatten()*1j*Z/omega0)
        Etil = np.reshape(np.conj(E2), (Nx,Ny))
        fullgrad2 = np.conj(Etil[Npml:-Npml, Npml:-Npml]) * chid_mask * Etot[Npml:-Npml, Npml:-Npml] * omega0**2 * omega0/1j/Z
        
        fullgrad = np.real(fullgrad1[:,:] + fullgrad2[:,:])/2 * dl**2 #dl**2 from inner products
        designgrad = fullgrad[design_mask].flatten()
        #gradient checks
        printGradChecks = False
        if printGradChecks:
            objfunc = lambda d: ext_objective(d, chif, background_mask, chid, design_mask, Npml, dl, cx, omega0, Qabs)
            #theoretical gradient
            randInd = round(random.uniform(0,np.sum(design_mask.flatten()) - 1))
            print('rand index:', randInd)
            theograd0 = designgrad[randInd]
            print("theograd", theograd0,flush=True)
            #numerical grad test
            deltadof = np.zeros_like(dof)
            deltadof[randInd] = 1e-4
            obj2 = objfunc(dof + deltadof)
            obj3 = objfunc(dof - deltadof)
            numgrad0 = (obj2-obj3)/np.linalg.norm(deltadof)/2
            print("numgrad",numgrad0,flush=True)
            if abs(theograd0) != 0:
                print("|theo-num|/|theo|", abs((theograd0 - numgrad0)/theograd0),flush=True)
            else:
                print("|theo-num|", abs((theograd0 - numgrad0)),flush=True)
    
        return Pext, designgrad
    else:
        return Pext

def designdof_ext_objective(dof, dofgrad, chif, background_mask, chid, design_mask, Npml, dl, source_cx, omega0, Qabs, opt_data):
    """
    optimization objective to be used with NLOPT
    opt_data is dictionary with auxiliary info such as current iteration number and output base
    """
    Z = 1.0
    objfunc = lambda d: ext_objective(dof, chif, background_mask, chid, design_mask, Npml, dl, source_cx, omega0, Qabs, returngrad=True)
    obj, grad = objfunc(dof)
    if len(dofgrad) > 0:
        dofgrad[:] = grad[:]
        
    opt_data['count'] += 1
    print('at iteration #', opt_data['count'], 'the ext value is', obj, ' with enhancement', obj/opt_data['background_term'], flush=True)
    if opt_data['count'] % opt_data['output_base'] == 0:
        nonpmlNx,nonpmlNy = design_mask.shape
        fulldof = np.zeros_like(design_mask)
        fulldof[design_mask] = dof[:]
        np.save(opt_data['name']+'_fulldof'+str(opt_data['count']), fulldof)

    return obj