####################################################################
####################################################################

## imports ~
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import math
from typing import Union

## qiskit imports ~
from qiskit import *
from qiskit.algorithms import *
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import quantum_info, IBMQ, Aer
from qiskit import BasicAer, transpile
from qiskit.utils import QuantumInstance
from qiskit.quantum_info import Statevector, partial_trace , DensityMatrixs
from qiskit.algorithms import optimizers, AmplitudeEstimation, EstimationProblem, AmplificationProblem, Grover, GroverResult, AmplitudeAmplifier
from qiskit.circuit.library import LinearAmplitudeFunction, LinearPauliRotations, PiecewiseLinearPauliRotations, WeightedAdder, GroverOperator
from qiskit_finance.circuit.library import LogNormalDistribution, NormalDistribution
from qiskit.visualization import plot_histogram, plot_state_qsphere, plot_bloch_multivector, plot_bloch_vector

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


def diffuser(l):
    """ Gnerate the Diffuser operator for the case where the initial state  is 
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

def s_psi0(p):  ## initial state preparation for a single qubit 
    """ Prepare a QuantumCircuit that intiates a single qubit state
        input:
            p= amplitude 
        output:
            s_psi0 gate                                            """

    qc = QuantumCircuit(1, name= " S_psi0 ")
    theta = 2*np.arcsin(np.sqrt(p))
    qc.ry(theta, 0)

    return qc.to_gate()


#####################################################################################
#####################################################################################

## helper functions for likelihood estimation

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
    
    
