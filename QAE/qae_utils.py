####################################################################
####################################################################

## imports ~
from tabnanny import verbose
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import math
from typing import Union

## qiskit imports ~
# from qiskit import *
from qiskit.algorithms import *
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, AncillaRegister
from qiskit import quantum_info, IBMQ, Aer
from qiskit import BasicAer, transpile
from qiskit.utils import QuantumInstance
from qiskit.quantum_info import Statevector, partial_trace , DensityMatrix
from qiskit.algorithms import optimizers, AmplitudeEstimation, EstimationProblem, AmplificationProblem, Grover, GroverResult, AmplitudeAmplifier
from qiskit.circuit.library import LinearAmplitudeFunction, LinearPauliRotations, PiecewiseLinearPauliRotations, WeightedAdder, GroverOperator
from qiskit_finance.circuit.library import LogNormalDistribution, NormalDistribution
from qiskit.visualization import plot_histogram, plot_state_qsphere, plot_bloch_multivector, plot_bloch_vector
from qiskit import execute

qsm = Aer.get_backend('qasm_simulator')
stv = Aer.get_backend('statevector_simulator')
aer = Aer.get_backend('aer_simulator')

####################################################################
####################################################################

## grover operator  fucntions ~

def str_to_oracle(pattern: str, name= 'oracle', return_type = "QuantumCircuit" ) -> Union[QuantumCircuit,  Statevector] :
    """ Convert a given string to an oracle circuit
        ARGS:
             pattern: a numpy vector with binarry entries 
        RETURNS: 
                QuantumCircuit   """

    l = len(pattern)
    qr = QuantumRegister(l, name='reg')
    a = AncillaRegister(1, name='ancilla')
    oracle_circuit = QuantumCircuit(qr, a, name= name+str(pattern))
    for q in range(l):
        if(pattern[q]=='0'): oracle_circuit.x(qr[q])
    oracle_circuit.x(a)
    oracle_circuit.h(a)
    oracle_circuit.mcx(qr, a)
    oracle_circuit.h(a)
    oracle_circuit.x(a)
    for q in range(l):
        if(pattern[q]=='0'): oracle_circuit.x(qr[q])
    
    #oracle_circuit.barrier()
    if return_type == "QuantumCircuit":
        return oracle_circuit


def generate_oracles(good_states: list) -> QuantumCircuit :
    """ Return a QuantumCircuit that implements the oracles given the good_states
        ARGS:
            good_states: list of good staes, states must be binary strings, Eg. ['00', '11']
        RETURNS:
            QuantumCircuit """

    oracles = [ str_to_oracle(good_state) for good_state in good_states ]
    oracle_circuit = oracles[0]
    for oracle in oracles[1:] :
        oracle_circuit.compose(oracle,  inplace= True)
    
    return oracle_circuit


def to_oracle(pattern, name= 'oracle'):
    """ Convert a given pattern to an oracle
        input: pattern= a numpy vector with binarry entries 
        output: oracle.Gate    """

    l = len(pattern)
    qr = QuantumRegister(l, name='reg')
    a = AncillaRegister(1, name='ancilla')
    qc = QuantumCircuit(qr, a, name= name+str(pattern))
    for q in range(l):
        if(pattern[q]==0): qc.x(qr[q])
    qc.x(a)
    qc.h(a)
    qc.mcx(qr, a)
    qc.h(a)
    qc.x(a)
    for q in range(l):
        if(pattern[q]==0): qc.x(qr[q])
    #qc.barrier()
    return qc.to_gate()


def diffuser(l:int)-> QuantumCircuit :
    """ Generate the Diffuser operator for the case where the initial state  is 
        the equal superposition state of all basis vectors 
        input: l= no. of qubits
        output: diffuser.Gate    """

    qr = QuantumRegister(l, name='reg')
    a = AncillaRegister(1, name='ancilla')
    circuit = QuantumCircuit(qr, a, name= 'Diff.')
    
    circuit.h(qr)
    circuit.x(qr)
    
    circuit.x(a)
    circuit.h(a)
    circuit.mcx(qr ,a)
    circuit.h(a)
    circuit.x(a)

    circuit.x(qr)
    circuit.h(qr)
          
    return circuit.to_gate()


def grover_iterate(qc, oracles, diffuser, qreg_u, ancilla, steps):
    """ Run full Grover iteration for given number of steps.
        input:
        qc: QuantumCiruit to append to 
        oracles: a list of oracles generated from 'to_oracle()' function 
        diffuser: a diffuser from 'diffuser()' function 
        steps: no. of grover iterates"""
    for step in range(steps):
        for oracle in oracles:
            qc.append(oracle, list(range(qc.num_qubits)) )
        qc.append(diffuser, list([q for q in qreg_u])+ list(ancilla) )
        # qc.barrier()
    return qc


def grover(patterns, grover_steps): ## modified sub-routine for grover based on the 'grover_iterate()' 

    dim = len(patterns[0])
    
    # create oracles ~\
    oracles = []
    for pattern in patterns : oracles.append( to_oracle(pattern)) 
    
    # create diffuser ~\
    diff = diffuser(dim)

    # create circuit ~\
    qreg = QuantumRegister(dim, name= 'init')
    ancilla = AncillaRegister(1, name='ancilla')
    qc = QuantumCircuit(qreg, ancilla, name='grover'+'^'+str(grover_steps))
    qc = grover_iterate(qc, oracles, diff, qreg, ancilla,grover_steps)
  
    return qc


####################################################################################
####################################################################################

## single qubit handler functions  ~

def s_psi0(p:float)-> QuantumCircuit :  ## initial state preparation for a single qubit 
    """ Prepare a QuantumCircuit that intiates a single qubit state
        input:
            p= amplitude of |1> in intial state
        output:
            s_psi0                                             """

    if verbose: print("inside-> 's_psi0'")

    qc = QuantumCircuit(1, name= " S_psi0 ")
    theta = 2*np.arcsin(np.sqrt(p))
    qc.ry(theta, 0)

    return qc


def Q(p: float, power:int= 1)-> QuantumCircuit:
    """ Prepare an 'QuantumCircuit' to implement grover operator 'Q'
        input:
            p= amplitude of |1> in intial state
            power= no.of times 'Q' is imposed
        output:
            Q^k                                                 """
    
    if verbose: print("inside-> 'Q'")

    theta = 2*np.arcsin(np.sqrt(p))
    qc = QuantumCircuit(1, name= ' Q'+ '^'+ str(power) )
    qc.ry(2*theta*power, 0)

    return qc


#####################################################################################
#####################################################################################

## sub-routines for QPE ~

def crot(qc, l):
    """ Function to generate Controlled Rotation Ooeration  """

    if l == 0:
        return qc
    l = l-1
    qc.h(l)
    for q in range(l):
        qc.cp(pi/2**(l-q), q, l)
    #qc.barrier()
    # qc.draw()
    
def QFT(qc):
   """function to generate QFT circuit """

   dim = qc.num_qubits
   for q in range(dim):
      crot(qc, dim-q)
   for q in range(int(dim/2)):
      qc.swap(q, dim-q-1)
   
   #qc.draw()
   return qc

def qpe(): ## TODO 
    pass  

#####################################################################################
#####################################################################################

## helper function for Quantum Phase Estimation ~

def qpe(p: float, trainable:bool= False, p_param:Union[float, None]= None , precision:int= 4, no_estimates:int= 5 ):
    
    ## define the grover operator ~
    # p = 0.25

    # precision= 4
    ## generate the Quantum Circuit ~
    preg = QuantumRegister(precision, name= 'precision_q')
    qreg = QuantumRegister(1, name='qreg')
    creg = ClassicalRegister(precision, name='precision_c')
    qc = QuantumCircuit(preg,qreg, creg)

    qc.h(preg)
    qc.append(s_psi0(p), [precision] )
    qc.barrier()

    if trainable== True:
        for q in range(precision):
            qc.append(Q( p_param  , 2**q).to_gate().control(1), [q]+list(range(precision,precision+1)) )
    
    else:
        for q in range(precision):
            qc.append(Q(p,2**q).to_gate().control(1), [q]+list(range(precision,precision+1)) )

    qftgate_inv = QFT(QuantumCircuit(precision, name='QFT')).to_gate().inverse()
    qc.barrier()
    qc.append(qftgate_inv, list(range(precision)))

    qc.measure(preg, creg)
    shots = 2000
    job = execute(qc, backend= aer, shots= shots)
    counts = job.result().get_counts()

    # no_estimates= 5
    estimate = sorted(zip(counts.values(), counts.keys()), reverse= True)
    p_est = 0
    for p in range(no_estimates): 
        p_est += np.sin(int(estimate[p][1], 2)*pi/(2**precision))**2 * (estimate[p][0]/ shots)

    return p_est


def to_minimize(p_params, p= 0.2 , precision:int= 4, no_estimates:int= 5 ):
    print("inside-> to_minimize") ## 
    p_est = qpe(p, trainable= True, p_param= p_params[0])
    print("ourside-> qpe, p_est", p_est)
    cost = (p_params - p_est)**2
    print("cost ", cost)
    return cost






#####################################################################################
#####################################################################################


## helper functions for maximum likelihood estimation ~

def likelihood(n,m, shots, thetas= np.linspace(0, 2*pi, 100), log= False):
    """ Generate likelihood function over the given range of 'thetas'
        Input:
            n = count of 'good' state i.e '1'
            m = no. of grover steps
            shots = no. of shots
            thetas = range of angles 
            log = (Bool) return the log value of likelihood   
        Output:
            dictionary containing {theta: liklihood(theta)} pairs"""
    dic = {}
    for theta in thetas: 
        dic[theta] = (((np.sin((2*m+1)*theta))**2)**n)*(((np.cos((2*m+1)*theta))**2)**(shots-n)) 
        if log== True: dic[theta] = np.log((((np.sin((2*m+1)*theta))**2)**n)*(((np.cos((2*m+1)*theta))**2)**(shots-n)) )
    return dic

def combine_likelihood(lkhs):
    """ Combine a list of liklehoods """
    dic = lkhs[0]
    for lkh in lkhs[1:]: 
        for theta in lkh.keys():
            dic[theta] = dic[theta]*lkh[theta]
            
    return dic

def generate_le(p,q= 1, shots= 100, thetas= np.linspace(0, 2*pi, 50, endpoint= False)):
    """ Generate likelihood for various values of 'p' and 'q' 
        Input:
            p= amplitude
            q= no. of grover steps
        Output:
            dictionary of likelihood
            """
        
    qreg = QuantumRegister(1, name= 'qreg')
    creg = ClassicalRegister(1, name= 'creg')
    qc = QuantumCircuit(qreg, creg )

    qc.append(s_psi0(p),[0])
    qc.barrier()
    qc.append(Q(p, q), [0])

    qc.measure(qreg, creg)
    counts =  execute(qc, backend= aer, shots = shots).result().get_counts()
    le = likelihood(counts['1'],q,shots,thetas= thetas)

    return le
    
    
