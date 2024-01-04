import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
def build_div_curl_op_Dirichlet(Nx, Ny, dx, dy, chigrid=None):
    """
    construct TE FDFD system matrix 
    the ordering of the indices goes:
    (x,y,Ex), (x,y+1,Ex),  (x,y+2,Ex) , ... , (x+1, y, Ex), (x+1,y+1, Ex), ...
    (x,y,Ey), (x,y+1,Ey),  (x,y+2,Ey) , ... , (x+1, y, Ey), (x+1,y+1, Ey), ...
    for now default periodic boundary conditions with no phase shift
    """
    
    A1_data = []
    A1_i = []
    A1_j = []  #prepare to construct matrix in COO format
    
    A2_data = []
    A2_i = []
    A2_j = []  #prepare to construct matrix in COO format
    
    #construct div to act on [Ex; Ey].
    #first Nx*Ny elements of vector are Ex on grid
    #last Nx*Ny elements of vector are Ey on grid
    #operator will be of size (2*Nx*Ny, 2*Nx*Ny)
    #the [0:Nx*Ny, :] part is the div part
    #the [Nx*Ny:, :] part is the curl part
    for cx in range(1, Nx-1):
        for cy in range(1, Ny-1):
            #construct div to act on [Ex; Ey]
            #dEx/dx using central difference
            i = cx*Ny + cy
            j = (((cx + 1) % Nx) * Ny) + cy
            A1_i.append(i); A1_j.append(j); A1_data.append(1.0/2/dx)

            i = cx*Ny + cy
            j = (((cx - 1) % Nx) * Ny) + cy
            A1_i.append(i); A1_j.append(j); A1_data.append(-1.0/2/dx)
            
            #dEy/dy using central difference
            i = cx*Ny + cy
            j = cx*Ny + ((cy + 1) % Ny) + Nx*Ny
            A1_i.append(i); A1_j.append(j); A1_data.append(1.0/2/dy)

            i = cx*Ny + cy
            j = cx*Ny + ((cy - 1) % Ny) + Nx*Ny
            A1_i.append(i); A1_j.append(j); A1_data.append(-1.0/2/dy)
            
    for cx in range(1, Nx-1):
        for cy in range(1, Ny-1):
            #construct curl to act on [Ex; Ey]
            #-dEx/dy using central difference with PBC
            i = cx*Ny + cy
            j = cx*Ny + ((cy + 1) % Ny)
            A2_i.append(i); A2_j.append(j); A2_data.append(-1.0/2/dy)

            i = cx*Ny + cy
            j = cx*Ny + ((cy - 1) % Ny) 
            A2_i.append(i); A2_j.append(j); A2_data.append(1.0/2/dy)
            
            #dEy/dx using central difference with PBC
            i = cx*Ny + cy
            j = (((cx + 1) % Nx) * Ny) + cy + Nx*Ny
            A2_i.append(i); A2_j.append(j); A2_data.append(1.0/2/dx)

            i = cx*Ny + cy
            j = (((cx - 1) % Nx) * Ny) + cy + Nx*Ny
            A2_i.append(i); A2_j.append(j); A2_data.append(-1.0/2/dx)
            
    #Set Dirichlet conditions
    #set Dirichlet on the x-edges
    for cy in range(Ny):
        #Dirichlet on the div part
        #Ex part
        i = 0*Ny + cy
        j = 0*Ny + cy
        A1_i.append(i); A1_j.append(j); A1_data.append(1.0)
        
        i = (Nx-1)*Ny + cy
        j = (Nx-1)*Ny + cy
        A1_i.append(i); A1_j.append(j); A1_data.append(1.0)
        #Ey part
        i = 0*Ny + cy
        j = 0*Ny + cy + Nx*Ny
        A2_i.append(i); A2_j.append(j); A2_data.append(1.0)
        
        i = (Nx-1)*Ny + cy
        j = (Nx-1)*Ny + cy + Nx*Ny
        A2_i.append(i); A2_j.append(j); A2_data.append(1.0)
    #set Dirichlet on the y-edges
    for cx in range(Nx):
        #Dirichlet on the div part
        #Ex part
        i = cx*Ny + 0
        j = cx*Ny + 0
        A1_i.append(i); A1_j.append(j); A1_data.append(1.0)
        
        i = cx*Nx + (Ny-1)
        j = cx*Nx + (Ny-1)
        A1_i.append(i); A1_j.append(j); A1_data.append(1.0)
        #Ey part
        i = cx*Ny + 0 
        j = cx*Ny + 0 + Nx*Ny
        A2_i.append(i); A2_j.append(j); A2_data.append(1.0)
        
        i = cx*Nx + (Ny-1)
        j = cx*Nx + (Ny-1) + Nx*Ny
        A2_i.append(i); A2_j.append(j); A2_data.append(1.0)
                
    A1 = sp.coo_matrix((A1_data, (A1_i, A1_j)), shape=(Nx*Ny, 2*Nx*Ny))
    A1 = A1.tocsr()
    
    if not (chigrid is None):
        eps = sp.coo_matrix((chigrid.flatten() + 1, (np.arange(Nx*Ny), np.arange(Nx*Ny))), shape=(2*Nx*Ny,2*Nx*Ny))
        eps += sp.coo_matrix((chigrid.flatten() + 1, (np.arange(Nx*Ny, 2*Nx*Ny), np.arange(Nx*Ny, 2*Nx*Ny))), shape=(2*Nx*Ny,2*Nx*Ny))
        A1 = A1 @ eps
    
    A2 = sp.coo_matrix((A2_data, (A2_i, A2_j)), shape=(Nx*Ny, 2*Nx*Ny))
    A2 = A2.tocsr()
    A = sp.vstack([A1,A2])
    return A.tocsr()

def get_ES_TE_dipole_field(Nx, Ny, dx, dy, exmin, exmax, eymin, eymax, pol=1, amp=1.0, chigrid=None):
    A = build_div_curl_op_Dirichlet(Nx, Ny, dx, dy, chigrid)
    b = np.zeros(A.shape[0], dtype=complex)

#     for jj in range(exmin, exmax):
#         for kk in range(eymin, eymax):
#             xyind = jj*Ny + kk
#             b[xyind] = amp #electrostatic source
    ESamp = amp * dx*(exmax-exmin) * dy*(eymax-eymin) #the ES pole should have same amp as the non-static case
    if pol == 1:
        for jj in range(exmin, Nx//2):
            for kk in range(eymin, eymax):
                xyind = jj*Ny + kk
                b[xyind] = ESamp/dx/dy/(eymax-eymin)/(Nx//2 - exmin)/2
        for jj in range(Nx//2, exmax):
            for kk in range(eymin, eymax):
                xyind = jj*Ny + kk
                b[xyind] = -ESamp/dx/dy/(eymax-eymin)/(exmax - Nx//2)/2
    else:
        for jj in range(exmin, exmax):
            for kk in range(eymin, Ny//2):
                xyind = jj*Ny + kk
                b[xyind] = ESamp/dx/dy/(exmax-exmin)/(Ny//2 - eymin)/2
        for jj in range(exmin, exmax):
            for kk in range(Ny//2, eymax):
                xyind = jj*Ny + kk
                b[xyind] = -ESamp/dx/dy/(exmax-exmin)/(eymax - Ny//2)/2
    testSourceES = False
    if testSourceES:
        print('ESamp:', ESamp)
        print('b dipole signed:', np.sum(np.conj(b)*np.abs(np.sign(b))) * dx * dy, flush=True)
        print('b dipole magnitude:', np.sum(np.conj(b)*np.sign(b)) * dx * dy, flush=True)
#         plt.imshow(np.real(np.reshape(b[0:Nx*Ny], (Nx,Ny))), cmap='Greys')
#         plt.title('source')
    x = spla.spsolve(A, b)
    Exfield = np.reshape(x[:Nx*Ny], (Nx,Ny))
    Eyfield = np.reshape(x[Nx*Ny:], (Nx,Ny))
    
    return Exfield, Eyfield