import numpy as np
from numpy import array as A


class parameters():
   dtN = 100.0
   time = 1000
   ps = 41342
   NSteps = time * ps // int(dtN)  # int(2*10**6)
   NTraj = 5
   dtE = dtN/20
   NStates = 2
   M = 1
   initState = 0
   nskip = 10
   beta = 1 / 0.00095
   theta = np.pi / 16
   gamma_1 = 0.00001837465
   gamma_2 = gamma_1 / 2
   j_exc = 4.2629194e-6
   ev = 27.211385
   bath = np.loadtxt('bath.txt')
   omega, c_1 = bath[:, 0], bath[:, 1]
   lambda_1 = 0.1
   lambda_2 = 0.2
   e_ct1 = -0.1 / ev + lambda_1 / ev
   e_ct2 = -0.35 / ev + (np.sqrt(lambda_1 / ev) + np.sqrt(lambda_2 / ev))**2
   c_2 = (np.sqrt(lambda_2) + np.sqrt(lambda_1)) * c_1 / np.sqrt(lambda_1)
   ndof = len(c_1)

def Hel(R):
    par = parameters()
    Vij = np.zeros((2, 2))
    #Vij = np.zeros((2, 2), dtype=complex)

    Vij[1, 1] = par.e_ct1 + np.sum(par.c_1 * R)

    Vij[0, 1], Vij[1, 0] = par.gamma_1, par.gamma_1
    return Vij


def dHel0(R):
    omega = parameters.omega
    dH0 = omega**2 * R
    return dH0


def dHel(R):
    c_1 = parameters.c_1
    c_2 = parameters.c_2

    dHij = np.zeros((2, 2, len(R)))
    dHij[1, 1, :] = c_1
    return dHij


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
