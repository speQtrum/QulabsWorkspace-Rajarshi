####################################################################
                            ## imports ##
####################################################################


from tabnanny import verbose
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import math
from typing import Iterable, Union, Optional

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

#############################################################################
                    ## grover operator  fucntions ##
#############################################################################



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
                        ## single qubit handler functions  ##
####################################################################################



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
                            ## sub-routines for QPE ##
#####################################################################################



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
                ## Helper function for Quantum Phase Estimation ##
#####################################################################################



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
                ## Helper functions for maximum likelihood estimation ##
#####################################################################################


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
    
    
#############################################################################################
                       ### State Preparation Ansatz ###
#############################################################################################

class state_preparation_ansatz(QuantumCircuit):
    """ 
        A class to initiate the state preparation ansatz for the Variational QAE algorithm

        ARGS
        ----
            num_ancillas:[int]
                        no. of ancillas to be used for the ansatz prepration
            name:[str]
                    name of the ansatz circuit

        RETURNS
        -------
                [QuantumCircuit] implementing the state preparation ansatz

    """
    
    def __init__(self,
        num_ancillas: int,
        name: str= "Q",
        # params: Optional[Union[Iterable, str, None]]= "random",
        insert_barrier:bool= False)-> None:

        super().__init__(name= name)

        ## set inputs ~
        self._num_ancillas= num_ancillas
        self._name= name
        self._insert_barrier= insert_barrier
        # self._params= params
        self._num_params = self._num_ancillas * 4 + 2

        ## build parameters ~
        qreg = QuantumRegister(1)
        ancillas = AncillaRegister(self._num_ancillas)
        qc = QuantumCircuit(qreg, ancillas)
        self.add_register(*qc.qregs)
                

    @property
    def num_ancillas(self):
        """ No, of ancillas used in the ansatz """
        return self._num_ancillas
    
    @property
    def num_params(self):
        """ No. of parmaters passed into the ansatz """
        return self._num_params

    def load_params(self, params_to_load: Iterable):
        """ 
            ARGS
            ----
                params_to_load:[Iterable]
                                new parametes to be reassigned into the ansatz

            RETURNS
            -------
                 [QuantumCircuit] with reassigned paramters

        """
        
        ## get list of tunable paramters ~
        param_data = [ data[0] for data in self.data if data[0].name== 'ry' ]
        
        ## reset the paramters with new 'params_to_load' ~
        for index, instrc in enumerate(param_data): 
            instrc.params = [ params_to_load[index]]

        return self

    def show_params(self):
        """ Print the current paramter values in the ansatz """
        
        param_data = [ data[0] for data in self.data if data[0].name== 'ry' ]
        for index, intrc in enumerate(param_data):
            print( 'param' +str(index)+': ' ,intrc.params)


    def initialize(self, params):
        """ Function to build the state preparation circuit 

            ARGS
            ----
                params:[Iterable, str]
                        paramters to initialize the circuit.
                        if params = 'random' the circuit paramters are initialised randomly 
            RETURNS
            -------
                 [QuantumCircuit] with initialised paramters


        """

        if isinstance(params, str) and params== "random":

            print(" 'random' ")
            self._params_state = np.random.uniform(low=0, high= 2*pi, size= 2)
            self._params_ancilla_1= np.random.uniform(low=0, high= 2*pi, size= self._num_ancillas )
            self._params_ancilla_2= np.random.uniform(low=0, high= 2*pi, size= self._num_ancillas )
            self._params_ancilla_init= np.random.uniform(low=0, high= 2*pi, size= self._num_ancillas )
            self._params_ancilla_end= np.random.uniform(low=0, high= 2*pi, size= self._num_ancillas )
        
            
        elif isinstance(params, Iterable):

            if len(params) != self._num_params: raise ValueError(" no. of elements in 'params' must be ", self._num_params )
            self._params_state= np.append(params[0],params[-1])
            self._params_ancilla_init= params[1: 1 + self._num_ancillas]
            self._params_ancilla_1= params[1 + self._num_ancillas: 1 + 2 * self._num_ancillas]
            self._params_ancilla_2= params[1+ 2 * self._num_ancillas: 1 + 3 * self._num_ancillas]
            self._params_ancilla_end = params[1 + 3 * self._num_ancillas: -1]


        params_state = self._params_state
        params_ancilla_1= self._params_ancilla_1
        params_ancilla_2 = self._params_ancilla_2
        params_ancilla_init = self._params_ancilla_init
        params_ancilla_end = self._params_ancilla_end
        

        ancillas = self.ancillas
        qreg = self.qubits[0]

        ## set_ansatz ~
        self.h(ancillas)
        self.ry(params_state[0], qreg)

        for index, ancilla in list(enumerate(self.ancillas))  : self.ry(params_ancilla_init[index], ancilla)
        if self._insert_barrier == True: self.barrier()

        self.cx(qreg, ancillas[0])

        for index, ancilla in list(enumerate(ancillas))  :
            self.ry(params_ancilla_1[index], ancilla)
            self.cx(ancillas[index], ancillas[(index+1)%(self._num_ancillas)])
        
        if self._insert_barrier == True: self.barrier()
        
        for index, ancilla in list(reversed(list(enumerate(ancillas))))  :
            self.ry(params_ancilla_2[index], ancilla)
            self.cx(ancillas[index], ancillas[(index-1)%(self._num_ancillas)])

        self.cx( ancillas[0], qreg)
        
        if self._insert_barrier == True: self.barrier()     
        for index, ancilla in list(enumerate(ancillas))  : self.ry(params_ancilla_end[index], ancilla)   
     
        self.ry(params_state[1], qreg)
        self.h(ancillas)
        
    
        return self

           