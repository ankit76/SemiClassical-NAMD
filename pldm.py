import numpy as np
# Initialization of the mapping Variables
def initMapping(states, initState = 0,stype= "focused"):
    qF = np.zeros((states))
    qB = np.zeros((states))
    pF = np.zeros((states))
    pB = np.zeros((states))
    if stype = "focused":
        qF[initState] = 1.0
        qB[initState] = 1.0
        pF[initState] = 1.0
        pB[initState] = -1.0
    else:
       qF = np.array([ np.random.normal() for i in range(states)]) 
       qB = np.array([ np.random.normal() for i in range(states)]) 
       pF = np.array([ np.random.normal() for i in range(states)]) 
       pB = np.array([ np.random.normal() for i in range(states)]) 
    return qF, qB, pF, pB 

def propagateMap():
