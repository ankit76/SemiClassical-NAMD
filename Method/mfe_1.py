import os
import numpy as np
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'
from jax import jit, vmap, lax, numpy as jnp
from functools import partial

# Initialization of the electronic part
def initElectronic(Nstates, initState = 0):
    #global qF, qB, pF, pB, qF0, qB0, pF0, pB0
    c = jnp.zeros((Nstates)) + 0.j
    c = c.at[initState].set(1.0)
    return c

@jit
def propagateCi(ci, Vij, dt):
    c = ci * 1.0
    # https://thomasbronzwaer.wordpress.com/2016/10/15/numerical-quantum-mechanics-the-time-dependent-schrodinger-equation-ii/
    ck1 = (-1j) * (Vij @ c)
    ck2 = (-1j) * Vij @ (c + (dt/2.0) * ck1 )
    ck3 = (-1j) * Vij @ (c + (dt/2.0) * ck2 )
    ck4 = (-1j) * Vij @ (c + (dt) * ck3 )
    c = c + (dt/6.0) * (ck1 + 2.0 * ck2 + 2.0 * ck3 + ck4)
    return c

@jit
def Force(dat):
    dH = dat['dHij'] #dHel(R) # Nxnxn Matrix, N = Nuclear DOF, n = NStates
    dH0  = dat['dH0']
    ci = dat['ci']
    F = -dH0 - jnp.einsum('iij,i->j', dH, (ci * ci.conjugate()).real)
    return F

@partial(jit, static_argnums=(2,))
def VelVer(dat, par, NESteps):
    v = dat['P'] / par.M
    F1 = dat['F1']
    # electronic wavefunction

    #NESteps = par.NESteps
    dtE = par.dtE

    # half electronic evolution
    def half_elec_evolution(dat, x):
        dat['ci'] = propagateCi(dat['ci'], dat['Hij'], dtE)
        return dat, x
    
    dat, _ = lax.scan(half_elec_evolution, dat, jnp.arange(NESteps))
    dat['ci'] /= jnp.sum(dat['ci'].conjugate()*dat['ci'])

    # ======= Nuclear Block ==================================
    dat['R'] += v * par.dtN + 0.5 * F1 * par.dtN ** 2 / par.M

    #------ Do QM ----------------
    dat['Hij']  = par.Hel(dat['R'])
    dat['dHij'] = par.dHel(dat['R'])
    dat['dH0']  = par.dHel0(dat['R'])
    #-----------------------------
    F2 = Force(dat) # force at t2
    v += 0.5 * (F1 + F2) * par.dtN / par.M
    dat['F1'] = F2
    dat['P'] = v * par.M
    # ======================================================
    # half electronic evolution
    dat, _ = lax.scan(half_elec_evolution, dat, jnp.arange(NESteps))
    dat['ci'] /= jnp.sum(dat['ci'].conjugate()*dat['ci'])
    #dat['ci'] = ci * 1.0
    return dat

@jit
def pop(dat):
    ci =  dat['ci']
    return jnp.outer(ci.conjugate(),ci)

def runTraj(parameters):
    #------- Seed --------------------
    try:
        np.random.seed(parameters.SEED)
    except:
        np.random.seed(0)
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
    # Ensemble
    for _ in range(NTraj):
        # Trajectory data
        #dat = Bunch(param =  parameters )
        dat = {}
        dat['R'], dat['P'] = parameters.initR()

        # set propagator
        vv  = VelVer

        # Call function to initialize mapping variables
        dat['ci'] = initElectronic(NStates, initState) # np.array([0,1])

        #----- Initial QM --------
        dat['Hij']  = parameters.Hel(dat['R'])
        dat['dHij'] = parameters.dHel(dat['R'])
        dat['dH0']  = parameters.dHel0(dat['R'])
        dat['F1'] = Force(dat) # Initial Force

        #----------------------------
        iskip = 0 # please modify
        for i in range(NSteps): # One trajectory
            #------- ESTIMATORS-------------------------------------
            if (i % nskip == 0):
                rho_ensemble[:,:,iskip] += pop(dat)
                iskip += 1
            #-------------------------------------------------------
            dat = vv(dat, parameters, parameters.NESteps)

    return rho_ensemble

if __name__ == "__main__":
    #from Model import spinBoson as model
    #par =  model.parameters
    from Model import marcus_1
    par = marcus_1.parameters(NTraj=1)
    rho_ensemble = runTraj(par)
    NSteps = par.NSteps
    NTraj = par.NTraj
    NStates = par.NStates

    PiiFile = open("Pii.txt","w")
    for t in range(rho_ensemble.shape[-1]):
        PiiFile.write(f"{t * par.nskip  * par.dtN} \t")
        for i in range(NStates):
            PiiFile.write(str(rho_ensemble[i,i,t].real / NTraj) + "\t")
        PiiFile.write("\n")
    PiiFile.close()

