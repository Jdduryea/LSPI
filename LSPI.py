'''
Functions needed for training agents using LSPI
'''

### Public methods ###
import numpy as np
from scipy.optimize import minimize

def LSPI(basis_functions, gamma, epsilon, w, env, method = "discrete", n_trial_samples = 1000, n_timestep_samples=20):
    '''
    Compute the parameters of the policy, w, using the LSPI algorithm.
    
    Inputs:
    sample: list of tuples of the form (s,a,r,s')
    basis_functions: list of basis functions
    gamma: float, discount factor
    epsilon: float, convergence threshold
    w: intial policy parameter vector
    
    Outputs:
    w: the converged policy paramters
    '''
    w0 = []

    # for mountain car, use 1000 trials with 20 timesteps
    
    samples = _generate_samples(env, n_trial_samples, n_timestep_samples)

    while True:
        w_prev = w
        # try recollecting samples, doesn't seem to have a big affect
        #samples = _generate_samples(env, n_trial_samples, n_timestep_samples)

        w = _LSTDQ_OPT(samples, basis_functions, gamma, w, env, method=method)
        
        
        if _converged(w, w_prev, epsilon):
            break 
        else:
            w_prev = w
        w0.append(w[0])
        print (w[0])
    return w, w0


def get_policy_action(s,w, basis_functions, env, method="discrete"):
    '''
    Computes the best action for the current parameterized policy 
    according to the provided method (discrete, continuous, or continuous-discretized)
    
    Inputs:
    s: state
    w: parameters for policy
    basis_functions: list of basis functions that operate on states and actions
    env: gym environment
    method: string, 
    '''
    if method == "discrete":
        return _get_policy_action_discrete(s,w,basis_functions,env)
    if method == "continuous":
        return _get_policy_actions_continuous(s, w, basis_functions, env)
    if method == "continuous-discretized":
        return _get_policy_actions_continuous_discretized(s, w, basis_functions, env, n_discretizations=10)


### Private methods

def _converged(w, w_prev, epsilon):
    '''
    Determines if the policy parameters have converged based
    on whether or not the norm of the difference of w
    is less than the threshold epsilon.
    
    Inputs:
    w: a policy parameter vector
    w_prev: the policy parameter vetor from a previous iteration.
    epsilon: float, convergence threshold
    '''
    return np.linalg.norm(w - w_prev) < epsilon

def _LSTDQ(samples, basis_functions, gamma, w, env, method="discrete"):
    '''
    Simple version of LSTDQ
    '''
    k = len(basis_functions)
    #A = np.zeros((k,k)), this might not have an inverse, use the next line instead
    A = np.identity(k) * 0.01
    b = np.zeros(k)
    
    #samples[np.random.choice(len(samples), 100, replace=False)]
    
    for s, a, r, sp in samples:
        phi = _compute_phi(s,a, basis_functions)
        phi_p = _compute_phi(sp, get_policy_action(sp, w, basis_functions, env, method), basis_functions)

        A += np.outer(phi, (phi - gamma*phi_p))
        b = b + phi*r
    
    
    w = np.dot(np.linalg.inv(A),b)
    return w
    
    

    
    
def _LSTDQ_OPT(samples, basis_functions, gamma, w, env, sigma=0.1, method = "discrete" ):
    '''
    A faster version of LSTDQ. Computes an approximation of the policy parameters based
    on the LSTDQ-OPT algorithm presented in the paper.
    
    Inputs:
    sample: list of tuples of the form (s,a,r,s')
    basis_functions: list of basis functions
    gamma: float, discount factor
    epsilon: float, convergence threshold
    w: intial policy parameter vector
    
    sigma: small positive float.
    '''
    k = len(basis_functions)
    B = np.identity(k) * float(1/sigma)
    b = np.zeros(k)
    
    for s, a, r, sp in samples:
        phi = _compute_phi(s, a, basis_functions)
        phi_p = _compute_phi(sp, get_policy_action(sp, w, basis_functions, env, method), basis_functions)

        # Some computations that can be reused in the computation
        Bphi = np.dot(B, phi)
        phi_t = (phi - gamma*phi_p).T
        

        top = np.dot(np.outer(Bphi, phi_t), B)
        bottom = 1 + np.dot(phi_t, Bphi)
        B = B - top/bottom
        
        b = b + phi*r
    
    w = np.dot(B, b)

    return w
       

def _compute_phi(s, a, basis_functions):
    """
    Computes the vector ϕ(s,a) according to the basis function ϕ_1...ϕ_k
    
    Inputs:
    s: state
    a: action
    basis_functions: list of basis functions that operate on s and a
    
    Outputs:
    ϕ(s,a), a vector where each entry is the result of one of the basis functions.
    """

    phi = np.array([bf(s, a) for bf in basis_functions])
    return phi
    
    

    
def _get_policy_action_discrete(s, w, basis_functions, env):
    '''
    For discrete action spaces. Given a parameterization for the policy,
    reconstruct the policy and querery it to get 
    the optimal action for state s. That is,
    the argmax over actions of ϕ(s,a).w
    
    Inputs:
    s: state
    w: policy parameters
    basis_functions: the basis functions that are used in the model
    
    Outputs:
    action a that the policy says is best
    '''
    a_max = None
    max_score = float("-inf")
    
    #action_space = [0,1,2] # for acrobat
    #action_space = [0,1,2,3] # for
    action_space = [0, 1] # for cart pole
    # Search action space for most valuable action
 
    #TODO:  use sympy for grad desc
    for a in action_space:
        score = np.dot(_compute_phi(s, a, basis_functions), w)
        # update if we found something better
        if score > max_score:
            max_score = score
            a_max = a

    return a_max    
    
    
def _get_policy_actions_continuous_discretized(s, w, basis_functions, env, n_discretizations=10):
    '''
    For continuous action spaces, discretize the space and
    given a parameterization for the policy, reconstruct the policy and querery it to get 
    the optimal action for state s. That is,
    the argmax over actions of ϕ(s,a).w
    
    Inputs:
    s: stateget_policy_action
    w: policy parameters
    basis_functions: the basis functions that are used in the model
    n_discretizations: the number of chunks to split the continuous space into
       
    
    Outputs:
    action a that the policy says is best
    '''
    
    a_max = None
    max_score = float("-inf")
    
    # Discretize the continuous space into n_discretizations chunks
    action_space = np.linspace(env.action_space.low[0], env.action_space.high[0], n_discretizations)

    for a in action_space:
        score = np.dot(_compute_phi(s, a, basis_functions), w)
       
        # update if we found something better
        if score > max_score:
            max_score = score
            a_max = a
            
    return a_max



def _get_policy_actions_continuous(s,w,basis_functions, env):
    '''
    For continuous action spaces. Given a parameterization for the policy,
    reconstruct the policy and querery it to get 
    the optimal action for state s. That is,
    the argmax over actions of (s,a).w
    
    Inputs:
    s: stateget_policy_action
    w: policy parameters
    basis_functions: the basis functions that are used in the model
       
    
    Outputs:
    action a that the policy says is best
    '''
    f = lambda a: np.dot(_compute_phi(s, a, basis_functions), w)
    x0 = 0
    result = minimize(f, x0, method='L-BFGS-B', options={'xtol': 1e-8, 'disp': True}, bounds = [(-1,1)])
    return result.x

    
def _generate_samples(env, n_samples, n_steps=100):
    samples = []
    print (env.reset())
    for i in range(n_samples):
        env.reset()
        for j in range(n_steps):
            #s = list(env.env.state)
            s= env.env.state
            a = env.action_space.sample()
            
            sp,r, _,_ = env.step(a)
            
            sample = (s, a, r, sp)
            samples.append(sample)

    return np.array(samples)