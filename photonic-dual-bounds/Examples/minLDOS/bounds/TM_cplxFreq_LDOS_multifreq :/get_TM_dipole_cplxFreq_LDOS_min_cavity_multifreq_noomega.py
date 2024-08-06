import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import time,sys
sys.path.append('../../')

from dualbound.Rect.rect_domains import get_rect_proj_in_rect_region, divide_rect_region

from dualbound.Maxwell.TM_FDFD import get_TM_dipole_field, get_TM_dipole_field_omega, get_Gddinv, get_Gddinv_omega

from get_Msparse_bounds import get_Msparse_bound_multipole

from get_dense_bounds import get_multiSource_dense_bound

from dualbound.Rect.rect_iterative_splitting import dualopt_iterative_splitting, dualopt_Msparse_iterative_splitting_Lag_input, dualopt_Msparse_iterative_splitting_vacuum_multipole


def get_TM_dipole_oneside_ldos_Msparse_iterative_splitting(chi, wvlgth, Qabs, design_x, design_y, vacuum_x, vacuum_y, emitter_x, emitter_y, dist, pml_sep, pml_thick, gpr, opttol=1e-2, fakeSratio=1e-2, reductFactor=0.1, iter_period=20, alg='Newton', Lag_input='Lags_input.txt', Qabstol=1e2, initialize=False, Geometry='Cavity', Num_Poles=1):

    zinv = np.imag(chi) / np.real(chi*np.conj(chi))
    
    domain_x = 2*pml_sep + design_x
    domain_y = 2*pml_sep + design_y

    dl = 1.0/gpr #uniform discretization

    Mx = int(np.round(design_x/dl))
    My = int(np.round(design_y/dl))
    Dx = int(np.round(dist/dl))

    Npmlx = int(np.round(pml_thick/dl))
    Npmly = int(np.round(pml_thick/dl))
    Npmlsepx = int(np.round(pml_sep/dl))
    Npmlsepy = int(np.round(pml_sep/dl))
    nonpmlNx = Mx + Dx + 2*Npmlsepx
    nonpmlNy = My + 2*Npmlsepy
    Nx = nonpmlNx + 2*Npmlx
    Ny = nonpmlNy + 2*Npmly
    emitterx = int(np.round(emitter_x / dl))
    emittery = int(np.round(emitter_y / dl))
    vacuumx = int(np.round(vacuum_x / dl))
    vacuumy = int(np.round(vacuum_y / dl))
    
    ###############get Green's function for design region#############
    design_mask = np.zeros((Nx,Ny), dtype=np.bool)
    design_mask[Npmlx+Npmlsepx+Dx:Npmlx+Npmlsepx+Dx+Mx , Npmly+Npmlsepy:Npmly+Npmlsepy+My] = True
    design_mask2 = np.zeros((Mx,My), dtype=np.bool)
    design_mask2[:,:] = True
    if Geometry[0].lower() == 'c':
        print("Computing cavity bounds for square window...",flush=True)
        design_mask2[(Mx-vacuumx)//2:(Mx-vacuumx)//2+vacuumx,(My-vacuumy)//2:(My-vacuumy)//2+vacuumy] = False
        design_mask[Npmlx+Npmlsepx+Dx+(Mx-vacuumx)//2:Npmlx+Npmlsepx+Dx+(Mx-vacuumx)//2+vacuumx,Npmly+Npmlsepy+(My-vacuumy)//2:Npmly+Npmlsepy+(My-vacuumy)//2+vacuumy] = False
    else:
        print("Computing halfspace bounds for square window...",flush=True)
        
    k0_0 = (2*np.pi / wvlgth)
    omegas = []
    Polefactor = 0
    for ii in range(Num_Poles):
        omegas += [k0_0 * (1 + np.exp(1j*(np.pi+ii*2.0*np.pi)/(2*Num_Poles))/2./Qabs)]
        Polefactor += 1j*np.exp(1j*(np.pi+ii*2.0*np.pi)/(2*Num_Poles))
    print('getting Gddinv', flush=True)
    Gddinv = []
    for ii in range(Num_Poles):
        Gddinv1, _ = get_Gddinv_omega(wvlgth, dl, Nx, Ny, Npmlx, design_mask, omegas[ii], Num_Poles,ii)
        Gddinv += [Gddinv1.copy()]
        
    print('finished computing Gddinv', flush=True)

    #UM = [(Gddinv[0].conj().T @ Gddinv[0])/np.conj(chi) - Gddinv[0]]
    #AsymUM = [(UM[0] - UM[0].conj().T) / 2j]
    #for ii in range(1,Num_Poles):
    #    UM += [(Gddinv[ii].conj().T @ Gddinv[ii])/np.conj(chi) - Gddinv[ii]]
    #    AsymUM += [(UM - UM.conj().T) / 2j]
    #print('UM format', UM.format, 'AsymUM format', AsymUM.format, flush=True)

    #Id = sp.eye(Gddinv[0].shape[0], format="csc")
    #Plist = [[Id.copy()]]
    #for ii in range(1,Num_Poles):
    #    Plist += [[Id.copy()]]
    
    #get vacuum field
    Z = 1.0 #dimensionless units
    Ezfields = []
    vacPrad = 0
    if Geometry[0].lower() == 'c':
        cxmin = Nx//2-emitterx//2
    else:
        cxmin = Npmlx + Npmlsepx
    cymin = Ny//2-emittery//2 #position of dipole in entire computational domain
    for ii in range(len(omegas)):
        Ezfields += [get_TM_dipole_field_omega(wvlgth, dl, Nx, Ny, cxmin, cxmin+emitterx, cymin, cymin+emittery, Npmlx, omegas[ii])]
    source = np.zeros((Nx,Ny),dtype=np.complex)
    source[cxmin:cxmin+emitterx,cymin:cymin+emittery] = 1.0/dl/dl/emitterx/emittery
    for ii in range(len(omegas)):
        vacPrad += -0.5 * dl**2 * np.real(1j*((np.exp(1j*(np.pi+ii*2.0*np.pi)/(2*Num_Poles)))/Polefactor*np.sum(np.conj(source) * Ezfields[ii]))/omegas[ii] * 1./ k0_0 * (k0_0**2+(k0_0**2)*(1/2./Qabs)**2)) #default unit amplitude dipole
    S1 = [] 
    S2 = []
    for ii in range(len(omegas)):
        S1 += [(omegas[ii]/1j/Z) * 1j * ((np.exp(1j*(np.pi+ii*2.0*np.pi)/(2*Num_Poles)))/Polefactor * Ezfields[ii][design_mask])]  #S1 = G @ J
        S2 += [np.conj((omegas[ii]/1j/Z) * 1j * ((np.exp(1j*(np.pi+ii*2.0*np.pi)/(2*Num_Poles)))/Polefactor * Ezfields[ii][design_mask]))] #S2 = G* @ J = S1*
    
    print('num vacPrad', vacPrad, flush=True)
    k0_0 = (2.0*np.pi / wvlgth) #real part of wavevector
    wtil = k0_0 * (1.0 + 1j/2.0/Qabs) #the complex wavevector/frequency
    Ntil = 1.0/k0_0 * 1.0/(1.0 + (1.0/2.0/Qabs)**2) #normalization factor introduced in W weight function   
    exactvacPrad = np.arctan2(np.real(wtil), np.imag(wtil)) / (4*np.pi*Ntil)
    print('exact vacPrad', exactvacPrad, flush=True)
    #######################set up optimization##################
    O_quad = -sp.csc_matrix(Gddinv[0].shape, dtype=complex) #for LDOS, no quadratic term


    O_lin = -(((1./ k0_0 * (k0_0**2+(k0_0**2)*(1/2./Qabs)**2))*Z/2/np.conj(omegas[0])**2) * (1j/2) * (Gddinv[0].conj().T @ S2[0]) * dl**2).flatten()
    for ii in range(1,Num_Poles):
        O_lin += -(((1./ k0_0 * (k0_0**2+(k0_0**2)*(1/2./Qabs)**2))*Z/2/np.conj(omegas[ii])**2) * (1j/2) * (Gddinv[ii].conj().T @ S2[ii]) * dl**2).flatten()


    print('starting iterative splitting', flush=True)

    def Prad_outputfunc(optLags, optgrad, dualval, objval, vacPrad):
        Prad_enh = -dualval / vacPrad
        print('Prad_enh', Prad_enh, flush=True)
        print('vacPrad', vacPrad, flush=True)
    outputFunc = lambda L,G,D,O: Prad_outputfunc(L,G,D,O, vacPrad)

    dualoptFunc = lambda nS, initL, GinvdagPdagl, UPl, inc: get_Msparse_bound_multipole(S1, O_lin, O_quad, GinvdagPdagl, UPl, inc, Num_Poles=Num_Poles, initLags=initL, dualconst=-vacPrad, opttol=opttol, fakeSratio=fakeSratio, reductFactor=reductFactor, iter_period=iter_period, alg=alg)
    if Qabs > Qabstol and initialize==False:
        print('Initializing...',flush=True)
        get_TM_dipole_oneside_ldos_Msparse_iterative_splitting(chi, wvlgth, Qabstol, design_x, design_y, vacuum_x, vacuum_y, emitter_x, emitter_y, dist, pml_sep, pml_thick, gpr, opttol=opttol, fakeSratio=fakeSratio, reductFactor=reductFactor, iter_period=iter_period, alg=alg, Lag_input=Lag_input, Qabstol=Qabstol, initialize=True)

    dualopt_Msparse_iterative_splitting_vacuum_multipole(1, Mx, My, design_mask2, chi, Gddinv, dualoptFunc, outputFunc, Num_Poles=Num_Poles, pol='TM', Lag_input=Lag_input, initialize=initialize)

    

    
