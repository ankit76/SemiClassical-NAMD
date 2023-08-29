import os
import numpy as np
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'
from jax import jit, vmap, lax, numpy as jnp
from flax import struct
from typing import Callable, Any
from dataclasses import dataclass
from functools import partial

def rotate():
    theta = np.pi / 16
    gamma_1 = 0.00001837465
    gamma_2 = gamma_1 / 2
    j_exc = 100 * 4.2629194e-8
    ev = 27.211385
    #e_ct1 = -0.1 / ev
    #e_ct2 = -0.25 / ev
    lambda_1 = 0.1
    lambda_2 = 0.2
    e_ct1 = -0.1 / ev + lambda_1 / ev
    e_ct2 = -0.35 / ev + (np.sqrt(lambda_1 / ev) + np.sqrt(lambda_2 / ev))**2
    Vij = np.zeros((5, 5), dtype=complex)
    
    Vij[1, 1] = e_ct1 + j_exc
    Vij[2, 2] = e_ct1 - j_exc
    Vij[3, 3] = e_ct2 + j_exc
    Vij[4, 4] = e_ct2 + j_exc
     
    Vij[0, 1], Vij[1, 0] = gamma_1 * np.cos(theta), gamma_1 * np.cos(theta)
    Vij[0, 2], Vij[2, 0] = -1.j * gamma_1 * np.sin(theta), 1.j * gamma_1 * np.sin(theta)
    Vij[1, 3], Vij[3, 1] = gamma_2, gamma_2
    Vij[2, 4], Vij[4, 2] = gamma_2, gamma_2
 
    e, v = np.linalg.eigh(Vij)
    b1 = np.zeros((5,5))
    b1[1,1] = 1
    b1[2,2] = 1
    b2 = 0. * b1
    b2[3,3] = 1
    b2[4,4] = 1
    b1_p = v.conj().T @ b1 @ v
    b2_p = v.conj().T @ b2 @ v
    vi = np.diag(e)
    return vi.real, b1_p.real, b2_p.real, v


@struct.dataclass
class parameters():
    dtN: float = 100.0
    dtE: float = dtN / 20
    time: int = 1000
    ps: int = 41342
    NSteps: int = int(time * ps / dtN)  # int(2*10**6)
    NESteps: int = int(dtN / dtE) // 2
    dtE: float = 2 * dtN / NESteps
    NTraj: int = 10
    NStates: int = 5
    M: int = 1
    initState: int = 0
    nskip: int = 10
    beta: float = 1 / 0.00095
    theta: float = jnp.pi / 16
    gamma_1: float = 0.00001837465
    j_exc: float = 4.2629194e-6
    ev: float = 27.211385
    gamma_2: float = gamma_1 / 2
    bath: Any = jnp.array(np.loadtxt('bath.txt'))
    omega: Any = bath[:, 0]
    c_1: Any = bath[:, 1]
    lambda_1: float = 0.1
    lambda_2: float = 0.2
    e_ct1: float = -0.1 / ev + lambda_1 / ev
    e_ct2: float = -0.35 / ev + (np.sqrt(lambda_1 / ev) + np.sqrt(lambda_2 / ev))**2
    c_2: float = (np.sqrt(lambda_2) + np.sqrt(lambda_1)) * c_1 / np.sqrt(lambda_1)
    ndof: int = c_1.size
    ham_diag, b_1, b_2, u_rot = rotate()
    dhel_1 = jnp.einsum('k,ij->ijk', c_1, b_1) + jnp.einsum('k,ij->ijk', c_2, b_2)
    
    #@partial(jit, static_argnums=(0,))
    @jit
    def Hel(self, R):
        Vij = self.ham_diag
        Vij += self.b_1 * jnp.sum(self.c_1 * R) + self.b_2 * jnp.sum(self.c_2 * R)
        return Vij
    
    #@partial(jit, static_argnums=(0,))
    @jit
    def dHel0(self, R):
        omega = self.omega
        dH0 = omega**2 * R
        return dH0
    
    #@partial(jit, static_argnums=(0,))
    @jit
    def dHel(self, R):
        return self.dhel_1
    
    #@partial(jit, static_argnums=(0,))
    def initR(self):
        R0 = 0.0
        P0 = 0.0
        beta = self.beta
        omega = self.omega
        ndof = self.ndof

        sigP = jnp.sqrt(omega / (2 * jnp.tanh(0.5*beta*omega)))
        sigR = sigP/omega

        R = jnp.zeros((ndof))
        P = jnp.zeros((ndof))
        for d in range(ndof):
            R = R.at[d].set(np.random.normal()*sigR[d])
            P = P.at[d].set(np.random.normal()*sigP[d])
        return R, P
    
    def __hash__(self):
        return hash((self.dtN, self.time, self.ps, self.NTraj, self.NStates, self.M, self.initState, self.nskip, self.beta, self.theta, self.gamma_1, self.j_exc, self.ev, self.NSteps, self.dtE, self.gamma_2, self.bath, self.omega, self.c_1, self.lambda_1, self.lambda_2, self.e_ct1, self.e_ct2, self.c_2, self.ndof, self.ham_diag, self.b_1, self.b_2, self.u_rot, self.dhel_1))

def initR():
    R0 = 0.0
    P0 = 0.0
    beta = parameters.beta
    omega = parameters.omega
    ndof = parameters.ndof

    sigP = np.sqrt(omega / (2 * np.tanh(0.5*beta*omega)))
    sigR = sigP/omega

    R = np.zeros((ndof))
    P = np.zeros((ndof))
    for d in range(ndof):
        R[d] = np.random.normal()*sigR[d]
        P[d] = np.random.normal()*sigP[d]
    return R, P

if __name__ == '__main__':
    par = parameters()
    R, P = par.initR()
    print(R.shape)
    print(P.shape)
    print(par.dHel(R).shape)