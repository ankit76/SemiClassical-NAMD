import os
import numpy as np
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'
from jax import jit, vmap, lax, numpy as jnp
from flax import struct
from typing import Callable, Any
from dataclasses import dataclass
from functools import partial

@struct.dataclass
class parameters():
    dtN: float = 100.0
    dtE: float = dtN / 20
    time: int = 1000
    ps: int = 41342
    NSteps: int = int(time * ps / dtN)  # int(2*10**6)
    NESteps: int = int(dtN / dtE) // 2
    dtE: float = 2 * dtN / NESteps
    NTraj: int = 5
    NStates: int = 2
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
    
    #@partial(jit, static_argnums=(0,))
    @jit
    def Hel(self, R):
        Vij = jnp.zeros((2, 2))
        Vij = Vij.at[1, 1].set(self.e_ct1 + np.sum(self.c_1 * R))
        Vij = Vij.at[0, 1].set(self.gamma_1)
        Vij = Vij.at[1, 0].set(self.gamma_1)
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
        c_1 = self.c_1
        dHij = jnp.zeros((2, 2, R.size))
        dHij = dHij.at[1, 1, :].set(c_1)
        return dHij
    
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
        return hash((self.dtN, self.time, self.ps, self.NTraj, self.NStates, self.M, self.initState, self.nskip, self.beta, self.theta, self.gamma_1, self.j_exc, self.ev, self.NSteps, self.dtE, self.gamma_2, self.bath, self.omega, self.c_1, self.lambda_1, self.lambda_2, self.e_ct1, self.e_ct2, self.c_2, self.ndof))

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