import jax.numpy as jnp
import jax.experimental.sparse as jsp
from jax import jit 

@jit
def kron(A, B):
    output_shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

    if A.nse == 0 or B.nse == 0:
        return jsp.empty(output_shape)
    
    row = jnp.reshape(jnp.reshape(jnp.repeat(A.indices[:, 0], B.nse)*B.shape[0], (-1, B.nse)) + B.indices[:, 0], -1)
    col = jnp.reshape(jnp.reshape(jnp.repeat(A.indices[:, 1], B.nse)*B.shape[1], (-1, B.nse)) + B.indices[:, 1], -1)
    data = jnp.reshape(jnp.reshape(jnp.repeat(A.data, B.nse), (-1, B.nse)) * B.data, -1)

    indices = jnp.stack((row, col), axis=1)
    return jsp.BCOO((data, indices), shape=output_shape)

@jit
def diag(x):
    N = x.shape[0]
    indices = jnp.stack((jnp.arange(N), jnp.arange(N)), axis=1)
    return jsp.BCOO((x, indices), shape=(N, N))
    
@jit
def diag_m1(x, topval):
    x2 = jnp.append(topval, x)
    N = x.shape[0]

    indices = jnp.stack((jnp.arange(N+1), jnp.roll(jnp.arange(N+1), 1)), axis=1)
    # indices = jnp.append(jnp.stack((jnp.arange(N)+1, jnp.arange(N)), axis=1), jnp.array([0, N]), axis=0)
    return jsp.BCOO((x2, indices), shape=(N+1, N+1))

@jit 
def diag_p1(x, botval):
    x2 = jnp.append(botval, x)
    N = x.shape[0]

    indices = jnp.stack((jnp.roll(jnp.arange(N+1), 1), jnp.arange(N+1)), axis=1)
    return jsp.BCOO((x2, indices), shape=(N+1, N+1))

@jit
def eliminate_zeros(A, mask):
    row = A.indices[:, 0][mask]
    col = A.indices[:, 1][mask]
    data = A.data[mask]
    indices = jnp.stack((row, col), axis=1)
    return jsp.BCOO((data, indices), shape=A.shape)



if __name__ == '__main__':
    from jax import random
    import scipy.sparse as sp
    import numpy as np
    import time 

    key = random.PRNGKey(0)
    key, subkey = random.split(key)

    Nx, Ny = 5, 5
    A = jsp.random_bcoo(key, (Nx, Ny), nse=0.1)
    B = jsp.random_bcoo(subkey, (Nx, Ny), nse=0.1)

    KRON = 0
    DIAGM1 = 1
    if KRON:
        jaxsol = kron(A, B)

        Anp = sp.coo_matrix(A.todense())
        Bnp = sp.coo_matrix(B.todense())

        t1 = time.time()
        sol = sp.kron(Anp, Bnp)
        print(f"scipy: {time.time() - t1}")

        t2 = time.time()
        jaxsol = kron(A, B)
        print(f"jax: {time.time() - t2}")

        jaxsol_d = np.array(jaxsol.todense())
        sol_d = np.array(sol.todense())

        print(np.allclose(sol_d, jaxsol_d))

    if DIAGM1:
        x = np.arange(Nx)
        jaxsol = diag_m1(x, 1)
        npsol = sp.diags(x, offsets=-1).todense()
        npsol[0, -1] = 1
        jaxsol2 = diag_p1(x, 1)
        
        npsol2 = sp.diags(x, offsets=1).todense()
        npsol2[-1, 0] = 1

        print(np.allclose(np.array(jaxsol.todense()), npsol))
        print(np.allclose(np.array(jaxsol2.todense()), npsol2))

    C = A @ B
    print(C.nse / C.shape[0]**2)
    print(C.data, C.indices[:, 0], C.indices[:, 1])

    C = sp.coo_matrix(A.todense()) @ sp.coo_matrix(B.todense())
    print(C.nnz / C.shape[0]**2)
    print(C.data, type(C))

    # mask = C.data != 0
    # C = eliminate_zeros(C, mask)
    # print(C.nse / C.shape[0]**2)
    # print(C.data, C.indices[:, 0], C.indices[:, 1])

    # A = jsp.BCSR.from_bcoo(A)
    # B = jsp.BCSR.from_bcoo(B)
    # C = A @ B
    # print(C.nse / C.shape[0]**2)