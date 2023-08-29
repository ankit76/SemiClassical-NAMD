import os
import random
import numpy as np
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false                            intra_op_parallelism_threads=1'
#os.environ['JAX_ENABLE_X64'] = 'True'
from jax import jit, vmap, lax, numpy as jnp
from functools import partial
from numpy.random import random

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print(f"# Number of cores: {size}\n")

# Initialization of the mapping Variables
def initMapping(Nstates, initState = 0, stype = "trinagle", rot=None):
    qF  = np.zeros((Nstates))
    pF  = np.zeros((Nstates))
    gamma_0  = np.zeros((Nstates)) # Adjusted ZPE

    if (stype == "square" or stype == "□"):
        gamma = (np.sqrt(3.0) -1)/2
        eta = 2 * gamma * random(Nstates)  
        theta = 2 * np.pi * random(Nstates)

    if (stype == "triangle" or stype == "Δ"):
        eta = np.zeros(Nstates)
        theta = 2 * np.pi * random(Nstates)
        # For initial State
        while (True):
            eta[initState] = random()
            if ( 1 - eta[initState] >= random() ):
                break
        
        # For other States
        for i in range(Nstates):
            if (i != initState):
                eta[i] = random() * ( 1 - eta[initState] )
    
    eta[initState] += 1.0
    qF =  np.sqrt( 2 * eta ) * np.cos(theta)
    pF = -np.sqrt( 2 * eta ) * np.sin(theta)

    for i in range(Nstates):
        gamma_0[i] = eta[i] - 1 * (i == initState)
    
    if rot is not None:
        qF = rot @ qF
        pF = rot @ pF

    return jnp.array(qF.real), jnp.array(pF.real), gamma_0

@jit
def Umap(qF, pF, VMat, dt):
    qFin, pFin = qF * 1.0, pF * 1.0  # Store input position and momentum for verlet propogation
    # Store initial array containing sums to use at second derivative step

    VMatxqF =  VMat @ qFin #np.array([np.sum(VMat[k,:] * qFin[:]) for k in range(NStates)])

    # Update momenta using input positions (first-order in dt)
    pF -= 0.5 * dt * VMatxqF  # VMat @ qFin  
    # Now update positions with input momenta (first-order in dt)
    qF += dt * VMat @ pFin  
    # Update positions to second order in dt
    qF -=  (dt**2/2.0) * VMat @ VMatxqF
       #-----------------------------------------------------------------------------
    # Update momenta using output positions (first-order in dt)
    pF -= 0.5 * dt * VMat @ qF  

    return qF.real, pF.real 

@jit
def Force(dat, par):
    gamma_0 = dat['gamma_0']
    dH = dat['dHij'] #dHel(R) Nxnxn Matrix, N = Nuclear DOF, n = NStates 
    dH0 = dat['dH0']
    qF, pF = par.u_rot.conj().T @ dat['qF'], par.u_rot.conj().T @ dat['pF'] 
    # F = np.zeros((len(dat.R)))
    #F = -dH0
    #for i in range(len(qF)):
    #    F -= 0.5 * dH[i,i,:] * ( qF[i] ** 2 + pF[i] ** 2 - 2 * gamma_0[i])
    #    for j in range(i+1, len(qF)):
    #        F -= dH[i,j,:] * ( qF[i] * qF[j] + pF[i] * pF[j])
    F = -dH0 - 0.5 * jnp.einsum('ijk,i,j->k', dH, qF, qF) - 0.5 * jnp.einsum('ijk,i,j->k', dH, pF, pF) + jnp.einsum('iik,i->k', dH, gamma_0)
    return F

@partial(jit, static_argnums=(2,))
def VelVer(dat, par, NESteps) : # R, P, qF, qB, pF, pB, dtI, dtE, F1, Hij,M=1): # Ionic position, ionic velocity, etc.
 
    # data 
    v = dat['P']/par.M
    #EStep = int(par.dtN/par.dtE)
    dtE = par.dtE

    # half electronic evolution
    def half_elec_evolution(dat, x):
        dat['qF'], dat['pF'] = Umap(dat['qF'], dat['pF'], dat['Hij'], dtE)
        return dat, x
    
    dat, _ = lax.scan(half_elec_evolution, dat, jnp.arange(NESteps))

    # ======= Nuclear Block ==================================
    F1    =  Force(dat, par) # force with {qF(t+dt/2)} * dH(R(t))
    dat['R'] += v * par.dtN + 0.5 * F1 * par.dtN ** 2 / par.M
    
    #------ Do QM ----------------
    dat['Hij']  = par.Hel(dat['R'])
    dat['dHij'] = par.dHel_o(dat['R'])
    dat['dH0']  = par.dHel0(dat['R'])
    #-----------------------------
    F2 = Force(dat, par) # force with {qF(t+dt/2)} * dH(R(t+ dt))
    v += 0.5 * (F1 + F2) * par.dtN / par.M

    dat['P'] = v * par.M
    # =======================================================
    
    

    # half-step mapping
    dat['Hij'] = par.Hel(dat['R']) # do QM
    dat, _ = lax.scan(half_elec_evolution, dat, jnp.arange(NESteps))    
    return dat

@jit
def popSquare(dat, par):
    qF, pF = par.u_rot.conj().T @ dat['qF'], par.u_rot.conj().T @ dat['pF']
    N = qF.size
    eta = 0.5 * ( qF**2 + pF**2 )
    gamma = dat['gamma']
    rho_ij = jnp.outer(qF + 1j * pF, qF - 1j * pF) * 0 # have to recheck coherences
    rho_ij = rho_ij.at[jnp.diag_indices(N)].set(jnp.ones(N))

    # Inspired from Braden's (Braden Weight) Implementation
    # Check his Package : https://github.com/bradenmweight/QuantumDynamicsMethodsSuite
    #for i in range(N):
    #    for j in range(N):
    #        if ( eta[j] - (i == j) < 0.0 or eta[j] - (i == j) > 2 * gamma ):
    #            rho_ij[i,i] = 0
    
    mask_1 = jnp.tile(eta, (N,1)) - jnp.eye(N)
    mask = np.prod(mask_1 > 0., axis=1) * jnp.prod(mask_1 < 2 * gamma, axis=1) 
    rho_ij = rho_ij.at[jnp.diag_indices(N)].set(mask * jnp.diag(rho_ij))
    return rho_ij

@jit
def popTriangle(dat, par):
    qF, pF = par.u_rot.conj().T @ dat['qF'], par.u_rot.conj().T @ dat['pF']
    N = qF.size
    eta = 0.5 * ( qF**2 + pF**2 )
    rho_ij = jnp.outer(qF + 1j * pF, qF - 1j * pF) * 0 # have to recheck coherences
    rho_ij = rho_ij.at[jnp.diag_indices(N)].set(jnp.ones(N))
    # Inspired from Braden's (Braden Weight) Implementation
    # Check his Package : https://github.com/bradenmweight/QuantumDynamicsMethodsSuite
    #for i in range(N):
    #    for j in range(N):
    #            if ( (i == j and eta[j] < 1.0) or (i != j and eta[j] >= 1.0) ):
    #                rho_ij[i,i] = 0
    
    mask_1 = eta >= 1. 
    mask_2 = jnp.tile(eta, (N,1)) - 1.
    mask = mask_1 * jnp.prod(mask_2.at[jnp.diag_indices(N)].set(-jnp.ones(N)) < 0., axis=1)
    rho_ij = rho_ij.at[jnp.diag_indices(N)].set(mask * jnp.diag(rho_ij))
    return rho_ij

    return rho_ij

def runTraj(parameters, stype="triangle"):
    #------- Seed --------------------
    np.random.seed(rank)
    #------------------------------------
    ## Parameters -------------
    NSteps = parameters.NSteps
    NTraj = parameters.NTraj
    NStates = parameters.NStates
    initState = parameters.initState # intial state
    #stype = parameters.stype
    nskip = parameters.nskip
    #---------------------------
    if NSteps%nskip == 0:
        pl = 0
    else :
        pl = 1
    rho_ensemble = np.zeros((NStates,NStates,NSteps//nskip + pl), dtype=complex)
    # Ensemble
    for _ in range(NTraj): 
        # Trajectory data
        dat = {}
        dat['R'], dat['P'] = parameters.initR()

        # set propagator
        vv  = VelVer

        # Call function to initialize mapping variables
        dat['qF'], dat['pF'], dat['gamma_0'] = initMapping(NStates, initState, stype, parameters.u_rot) 
        if stype == "square" or stype == "□":
            dat['gamma'] = (np.sqrt(3.0) - 1.0)/2.0
            pop = popSquare
        if stype == "triangle" or stype == "Δ":
            dat['gamma'] = 1/3.0 
            pop = popTriangle
        #----- Initial QM --------
        dat['Hij']  = parameters.Hel(dat['R'])
        dat['dHij'] = parameters.dHel_o(dat['R'])
        dat['dH0']  = parameters.dHel0(dat['R'])
        #----------------------------
        iskip = 0 # please modify
        for i in range(NSteps): # One trajectory
            #------- ESTIMATORS-------------------------------------
            if (i % nskip == 0):
                rho_ensemble[:,:,iskip] += pop(dat, parameters)
                iskip += 1
            #-------------------------------------------------------
            dat = vv(dat, parameters, parameters.NESteps)

    # mpi averaging
    global_rho = None
    if rank == 0:
        global_rho = np.zeros((NStates, NStates, NSteps//nskip + pl)) + 0.j
    comm.Reduce([rho_ensemble, MPI.DOUBLE], [global_rho, MPI.DOUBLE], op=MPI.SUM, root=0)
    if rank == 0:
        global_rho /= size

    return global_rho

if __name__ == "__main__": 
    from Model import ciss_rot
    par = ciss_rot.parameters(NTraj=4)
    rho_ensemble = runTraj(par)
    NSteps = par.NSteps
    NTraj = par.NTraj
    NStates = par.NStates
    
    if rank == 0:
        PiiFile = open("Pii.txt", "w")
        for t in range(rho_ensemble.shape[-1]):
            PiiFile.write(f"{t * par.nskip  * par.dtN} \t")
            norm = 0
            for i in range(NStates):
                norm += rho_ensemble[i,i,t].real    
            for i in range(NStates):
                PiiFile.write(str(rho_ensemble[i, i, t].real / norm) + "\t")
            PiiFile.write("\n")
        PiiFile.close()