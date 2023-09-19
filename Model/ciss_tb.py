import os
import numpy as np
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'
from jax import jit, vmap, lax, numpy as jnp, scipy as jsp
from flax import struct
from typing import Callable, Any, Sequence
from dataclasses import dataclass
from functools import partial

@struct.dataclass
class parameters():
    dtN: float = 100.0
    dtE: float = dtN / 200
    time: int = 0.5
    ps: int = 41342
    NSteps: int = int(time * ps / dtN)  # int(2*10**6)
    NESteps: int = int(dtN / dtE) // 2
    dtE: float = 2 * dtN / NESteps
    NTraj: int = 10
    NSites: int = 100
    NStates: int = 2 * NSites
    beta: float = 1 / 0.00095
    M: int = 1
    initState: int = NSites // 2 - 1 
    nskip: int = 10
    shr: float = 1.
    omega: float = 0.05 / 27.211385
    c: float = (2 * shr * omega**3)**0.5
    tc: float = 0.02 / 27.211385
    NUnit: int = 10
    dphi: float = 2 * np.pi / NUnit
    theta: float = np.pi / 20
    p: float = 1.
    kappa: float = np.cos(theta)
    tau: float = p * np.sin(theta)
    lambda_soc: float = 1. * tc   
    
    @partial(jit, static_argnums=(0,))
    def Hel(self, R):
        Vij = jnp.zeros((self.NStates, self.NStates)) + 0.j
        t_vec = jnp.zeros(self.NSites)
        t_vec = t_vec.at[1].set(self.tc)
        hopping = jsp.linalg.toeplitz(t_vec)
        eph = self.c * R * jnp.eye(self.NSites)
        Vij = Vij.at[:self.NSites, :self.NSites].set(hopping + eph)
        Vij = Vij.at[self.NSites:, self.NSites:].set(hopping + eph)
        
        # carry = Vij
        # x: site index
        def scanned_fun(carry, x):
            x_u = x
            x_d = x + self.NSites
            # uu
            carry = carry.at[x_u + 1, x_u].add(-1.j * self.lambda_soc * self.p * self.kappa)
            carry = carry.at[x_u, x_u + 1].add(1.j * self.lambda_soc * self.p * self.kappa)
            # dd
            carry = carry.at[x_d + 1, x_d].add(1.j * self.lambda_soc * self.p * self.kappa)
            carry = carry.at[x_d, x_d + 1].add(-1.j * self.lambda_soc * self.p * self.kappa)            
            # ud
            carry = carry.at[x_u + 1, x_d].add(-1.j * self.lambda_soc * (self.p * self.tau * jnp.sin((x + 0.5) * self.dphi) - 1.j * self.tau * jnp.cos((x + 0.5) * self.dphi)))
            carry = carry.at[x_d, x_u + 1].add(1.j * self.lambda_soc * (self.p * self.tau * jnp.sin((x + 0.5) * self.dphi) + 1.j * self.tau * jnp.cos((x + 0.5) * self.dphi)))
            # du
            carry = carry.at[x_d + 1, x_u].add(-1.j * self.lambda_soc * (self.p * self.tau * jnp.sin((x + 0.5) * self.dphi) + 1.j * self.tau * jnp.cos((x + 0.5) * self.dphi)))
            carry = carry.at[x_u, x_d + 1].add(1.j * self.lambda_soc * (self.p * self.tau * jnp.sin((x + 0.5) * self.dphi) - 1.j * self.tau * jnp.cos((x + 0.5) * self.dphi)))
            return carry, x

        Vij, _ = lax.scan(scanned_fun, Vij, jnp.arange(self.NSites-1))                                        
        return Vij

    #@partial(jit, static_argnums=(0,))
    @jit
    def dHel0(self, R):
        omega = self.omega
        dH0 = omega**2 * R
        return dH0
    
    @partial(jit, static_argnums=(0,))
    def dHel(self, R):
        c = self.c
        dHij = jnp.zeros((self.NStates, self.NStates, R.size))
        
        # carry = dHij
        # x: site index
        def scanned_fun(carry, x):
            carry = carry.at[x, x, x].set(c)
            carry = carry.at[x + self.NSites, x + self.NSites, x].set(c)
            return carry, x
        
        dHij, _ = lax.scan(scanned_fun, dHij, jnp.arange(self.NSites))
        return dHij
    
    @partial(jit, static_argnums=(0,))
    def Force(self, ci, R):
        #dH = self.dHel(R)
        dH0  = self.dHel0(R)
        prob = (ci * ci.conjugate()).real
        F = -dH0 - self.c * (prob[:self.NSites] + prob[self.NSites:]).real
        #F = -dH0 - jnp.einsum('iij,i->j', dH, (ci * ci.conjugate()).real)
        return F

    #@partial(jit, static_argnums=(0,))
    def initR(self):
        R0 = 0.0
        P0 = 0.0
        beta = self.beta
        omega = self.omega
        ndof = self.NSites

        sigP = jnp.sqrt(omega / (2 * jnp.tanh(0.5*beta*omega)))
        sigR = sigP/omega

        R = jnp.zeros((ndof))
        P = jnp.zeros((ndof))
        for d in range(ndof):
            R = R.at[d].set(np.random.normal()*sigR)
            P = P.at[d].set(np.random.normal()*sigP)
        return R, P
    
    def __hash__(self):
        return hash((self.dtN, self.time, self.ps, self.NTraj, self.NStates, self.M, self.initState, self.nskip, self.beta, self.theta, self.NSteps, self.dtE, self.omega, self.c, self.tc, self.NUnit, self.dphi, self.p, self.kappa, self.tau, self.lambda_soc, self.NSites))

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
    np.random.seed(0)
    R, P = par.initR()
    #print(R.shape)
    #print(P.shape)
    #print(par.dHel(R).shape)
    #Vij = par.Hel(R)
    np.set_printoptions(precision=6, suppress=True)
    #print(Vij)
    #print(np.allclose(Vij, Vij.T.conj()))
    ci = jnp.array(np.random.rand(par.NStates) + 1.j * np.random.rand(par.NStates))
    ci /= jnp.sqrt(jnp.sum(ci.conjugate() * ci))
    F = par.Force(ci, R)
    print(F)