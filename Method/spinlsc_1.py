import os
import random
import numpy as np
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
#os.environ['JAX_ENABLE_X64'] = 'True'
from jax import jit, vmap, lax, numpy as jnp
from functools import partial

"""
This is a Focused-spinPLDM code 
Here sampled mean all combinations 
of forward-backward initialization 
Here focused mean only forward backward 
focused on initial state
"""

# Initialization of the mapping Variables
def initMapping(NStates, F = 0):
    """
    Returns np.array zF and zB (complex)
    Only Focused-PLDM implemented
    (Originally by Braden Weight)
    """

    gw = (2/NStates) * (np.sqrt(NStates + 1) - 1)

    # Initialize mapping radii
    rF = np.ones(( NStates )) * np.sqrt(gw)

    rF[F] = np.sqrt( 2 + gw ) # Choose initial mapping state randomly

    zF = np.zeros(( NStates ),dtype=complex)

    for i in range(NStates):
        phiF = random.random() * 2 * np.pi # Azimuthal Angle -- Always Random
        zF[i] = rF[i] * ( np.cos( phiF ) + 1j * np.sin( phiF ) )         

    return zF 

@jit
def Umap(z, VMat, dt):
    """
    Updates mapping variables
    """
        
    Zreal = jnp.real(z) 
    Zimag = jnp.imag(z) 

    # Propagate Imaginary first by dt/2
    Zimag -= 0.5 * VMat @ Zreal * dt

    # Propagate Real by full dt
    Zreal += VMat @ Zimag * dt
    
    # Propagate Imaginary final by dt/2
    Zimag -= 0.5 * VMat @ Zreal * dt

    return  Zreal + 1j*Zimag


def Force(dat):
    R  = dat['R']
    dH = dat['dHij']  
    dH0 = dat['dH0']
    gw  = dat['gw'] 

    zF = dat['zF']
    NStates = zF.size
    η = jnp.real( ( jnp.outer(jnp.conjugate(zF),zF) - gw * jnp.identity(NStates) ) )

    #η = 0.5 * np.real( ( np.outer( zF.conjugate(), zF ) + np.outer( zB.conjugate(), zB ) - 2 * gw * np.identity(NStates) ) )
    
    #F = np.zeros((len(R)))
    #F -= dH0
    #for i in range(NStates):
    #    F -= 0.5 * dH[i,i,:] * η[i,i]
    #    for j in range(i+1,NStates): # Double counting off-diagonal to save time
    #        F -= 2 * 0.5 * dH[i,j,:] * η[i,j]
    
    F = - dH0 - 0.5 * jnp.einsum('ijk,ij->k', dH, η)
    
    return F

@partial(jit, static_argnums=(2,))
def VelVer(dat, par, NESteps) :
    # data 
    v = dat['P'] / par.M
    #EStep = int(par.dtN/par.dtE)
    dtE = par.dtE

    # half electronic evolution
    def half_elec_evolution(dat, x):
        dat['zF'] = Umap(dat['zF'], dat['Hij'], dtE)
        return dat, x

    dat, _ = lax.scan(half_elec_evolution, dat, jnp.arange(NESteps))
    
    #F = dat['F']
    # ======= Nuclear Block ==================================
    #F1 = F - dat['dH0']  # force with {qF(t+dt/2)} * dH(R(t))
    F1 = Force(dat)
    dat['R'] += v * par.dtN + 0.5 * F1 * par.dtN ** 2 / par.M
    
    #------ Do QM ----------------
    dat['Hij'] = par.Hel(dat['R'])
    dat['dHij'] = par.dHel(dat['R'])
    dat['dH0'] = par.dHel0(dat['R'])
    #-----------------------------
    #F2 = F - dat['dH0'] # force with {qF(t+dt/2)} * dH(R(t+ dt))
    F2 = Force(dat)
    v += 0.5 * (F1 + F2) * par.dtN / par.M

    dat['P'] = v * par.M
    # =======================================================
    
    # half-step mapping
    dat['Hij'] = par.Hel(dat['R']) # do QM
    dat, _ = lax.scan(half_elec_evolution, dat, jnp.arange(NESteps))

    return dat

@jit
def pop(dat):
    NStates = dat['zF'].size
    rho = 0.5 * (jnp.outer(dat['zF'][:].conjugate(), dat['zF']) - dat['gw'] * np.identity(NStates))
    return rho

def runTraj(parameters):
    #------- Seed --------------------
    #try:
    #    np.random.seed(parameters.SEED)
    #except:
    #    np.random.seed(0)
    #------------------------------------
    ## Parameters -------------
    NSteps = parameters.NSteps
    NTraj = parameters.NTraj
    NStates = parameters.NStates
    initState = parameters.initState # intial state
    nskip = parameters.nskip
    #---------------------------
    if NSteps%nskip == 0:
        pl = 0
    else :
        pl = 1
    rho_ensemble = np.zeros((NStates,NStates,NSteps//nskip + pl)) + 0.j

    #---------------------------
    # Ensemble
    for _ in range(NTraj): 
        gw = (2/NStates) * (np.sqrt(NStates + 1) - 1)
        # Trajectory data
        dat = {}
        dat['gw'] = gw

        # initialize R, P
        dat['R'], dat['P'] = parameters.initR()
        
        # set propagator
        vv  = VelVer
 
        # Call function to initialize mapping variables
 
        # various 
        dat['zF']  = initMapping(NStates, initState) 

        #----- Initial QM --------
        dat['Hij']  = parameters.Hel(dat['R'])
        dat['dHij'] = parameters.dHel(dat['R'])
        dat['dH0']  = parameters.dHel0(dat['R'])
        dat['F1'] = Force(dat) # Initial Force
        #----------------------------
        iskip = 0  
        for i in range(NSteps): # One trajectory
            #------- ESTIMATORS-------------------------------------
            if (i % nskip == 0):
                rho_ensemble[:,:,iskip] += pop(dat)  
                iskip += 1
            #-------------------------------------------------------
            dat = vv(dat, parameters, parameters.NESteps)

    return rho_ensemble

if __name__ == "__main__": 
    from Model import marcus_1
    par = marcus_1.parameters(NTraj=10)
    rho_ensemble = runTraj(par)
    NSteps = par.NSteps
    NTraj = par.NTraj
    NStates = par.NStates

    PiiFile = open("Pii.txt", "w")
    for t in range(rho_ensemble.shape[-1]):
        PiiFile.write(f"{t * par.nskip  * par.dtN} \t")
        for i in range(NStates):
            PiiFile.write(str(rho_ensemble[i, i, t].real / NTraj) + "\t")
        PiiFile.write("\n")
    PiiFile.close()
