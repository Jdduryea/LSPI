
""" 
File containing an LSPI RL agent.
"""

import numpy as np
from scipy.optimize import minimize
import abc

import gym


class RLAgent(object):
    """ Abstract base class for reinforcement learning agents. """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def reset(self):
        """
        Reset the agent and the contained environment.

        :return: An initial observation.
        """
        pass

    @abc.abstractmethod
    def step(self, episode_num, step_num):
        """
        Take a step in the previously initialized environment from the input state.

        :return: The previous state, action taken, reward received, and next (aka current) state, as well
        as whether the environment reported that the current episode is done.
        """
        pass


class RandomAgent(RLAgent):
    """ Reinforcement learning agent which takes random actions. """

    def __init__(self, env, discount_factor=1.0):
        self.env_ = env
        self.discount_factor_ = discount_factor
        self.current_state_ = None

    def reset(self):
        """
        Reset by simply resetting the environment.

        :return: An initial observation from the environment.
        """
        self.current_state_ = self.env_.reset()
        return self.current_state_

    def step(self, episode_num, step_num):
        """
        Step by taking a random action in the action space of the stored environment.

        :return: (state, action, reward, next state, done)
        """
        action = self.env_.action_space.sample()
        prev_state = self.current_state_
        self.current_state_, reward, done, _ = self.env_.step(action)


        return prev_state, action, reward, self.current_state_, done

# Good basis functions for the cartpole problem
def get_best_cartpole_basis_functions():
    '''
    This one does really well! USE THIS ONE!!
    Just some simple quadratics
    '''

    Q1 = np.identity(5)
    Q2 = np.ones((5, 5))
    Q3 = np.array([[1, 1, 1, 1, -1], [1, 1, 1, -1, 1], [1, 1, -1, 1, 1], [1, -1, 1, 1, 1], [-1, 1, 1, 1, 1]])

    v = lambda s, a: np.append(s, a)
    bf1 = lambda s, a: 1
    bf2 = lambda s, a: np.dot(np.dot(v(s, a), Q1), v(s, a))
    bf3 = lambda s, a: np.dot(np.dot(v(s, a), Q2), v(s, a))
    bf4 = lambda s, a: np.dot(np.dot(v(s, a), Q3), v(s, a))

    bfs = [bf1, bf2, bf3, bf4]
    return bfs





# RL Agent that uses LSPI
class LSPIAgent(RLAgent):
    '''
    An agent that uses the LSPI RL algorithm.
    '''
    def __init__(self, env, policy_param, basis_functions, discount_factor=1.0):
        self.env_ = env
        self.policy_param = policy_param
        self.basis_functions = basis_functions
        self.discount_factor_ = discount_factor
        self.current_state_ = None
        #self.action_bounds = [(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi)]
        self.action_bounds = [(0, 1)]
        self.reset()

    def reset(self):
        """
        Reset by simply resetting the environment.

        :return: An initial observation from the environment.
        """
        self.current_state_ = self.env_.reset()
        return self.current_state_

    def step(self, episode_num, step_num):
        """
        Step by taking a random action in the action space of the stored environment.

        :return: (state, action, reward, next state, done)
        """
        prev_state = self.current_state_
        action = self._get_policy_action(prev_state)

        self.current_state_, reward, done, _ = self.env_.step(int(action))
        return prev_state, action, reward, self.current_state_, done

    def _compute_phi(self, s, a):
        """
        Computes the vector ϕ(s,a) according to the basis function ϕ_1...ϕ_k

        Inputs:
        s: state
        a: action
        basis_functions: list of basis functions that operate on s and a

        Outputs:
        ϕ(s,a), a vector where each entry is the result of one of the basis functions.
        """

        phi = np.array([bf(s, a) for bf in self.basis_functions])
        return phi

    def _get_policy_action(self, state):
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
        f = lambda action: -1*np.dot(self._compute_phi(state, action), self.policy_param)
        x0 = 1
        result = minimize(f, x0, method='L-BFGS-B', options={'xtol': 1e-8, 'disp': False}, bounds=self.action_bounds)
        return result.x
        # amax = None
        # max_score = 1*float("inf")
        # for a in [0,1]:
        #     if f(a) < max_score:
        #         amax = a
        #         max_score = f(a)
        #
        # #print int((np.round(action[0])))
        # print ("amax:" ,amax)
        # print("result x0:", result.x)
        # print ("-----------------")
        # return amax



    def _generate_samples(self, n_samples, n_steps=100):
        """
        Gather some samples to use in training
        :param n_samples:
        :param n_steps:
        :return:
        """
        samples = []
        print(self.env_.reset())
        for i in range(n_samples):
            self.env_.reset()
            for j in range(n_steps):
                # s = list(env.env.state)
                s = self.env_.env.state
                a = self.env_.action_space.sample()

                sp, r, _, _ = self.env_.step(a)

                sample = (s, a, r, sp)
                samples.append(sample)

        return np.array(samples)

    def train(self):
        gamma = 0.95
        epsilon = 0.01
        self.LSPI(gamma, epsilon)
        pass

    def LSPI(self, gamma, epsilon, n_trial_samples=10000, n_timestep_samples=6):
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

        samples = self._generate_samples(n_trial_samples, n_timestep_samples)

        for i in range(5):
        #while True:
            w_prev = self.policy_param

            self.policy_param = self._LSTDQ_OPT(samples, gamma)
            print(self.policy_param)
            if self._converged(self.policy_param, w_prev, epsilon):
                break
            #else:
             #   w_prev = w

            # sanity check
            print(self.policy_param[0])

    def _converged(self, w, w_prev, epsilon):
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

    def _generate_samples(self, n_samples, n_steps=100):
        samples = []

        for i in range(n_samples):
            self.env_.reset()
            for j in range(n_steps):
                # s = list(env.env.state)
                s = self.env_.env.state
                a = self.env_.action_space.sample()

                sp, r, _, _ = self.env_.step(a)

                sample = (s, a, r, sp)
                samples.append(sample)

        return np.array(samples)



    def _LSTDQ_OPT(self, samples, gamma, sigma=0.1):
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
        k = len(self.basis_functions)


        #B = np.identity(k) * (1.0/0.1)
        B = np.identity(k) * float(1/sigma)
        b = np.zeros(k)

        for s, a, r, sp in samples:
            phi = self._compute_phi(s, a)
            phi_p = self._compute_phi(sp, self._get_policy_action(sp))

            # Some computations that can be reused in the computation
            Bphi = np.dot(B, phi)

            phi_t = (phi - gamma * phi_p).T

            top = np.dot(np.outer(Bphi, phi_t), B)
            bottom = 1 + np.dot(phi_t, Bphi)
            B = B - top / bottom

            b = b + phi * r

        w = np.dot(B, b)

        return w
#
env = gym.make("CartPole-v0")
env.reset()
bfs = get_best_cartpole_basis_functions()
policy_param = np.zeros(len(bfs))
agent = LSPIAgent(env, policy_param, bfs)

agent.train()
print("best params:", agent.policy_param)
#agent.policy_param = [ 1.99899848e+01 ,-1.65499892e-03  ,1.36777801e-03 , 5.32496602e-04]

env._max_episode_steps = 10000
method = "discrete"
#method = "continuous"
num_steps = []
for i_episode in range(100):
    observation = env.reset()
    agent.reset()
    print ("--------")
    t = 0
    actions = []
    while True:
        t+=1
        #env.render()
        #action = LSPI.get_policy_action(env.env.state, w_est, bfs, env, method = method)
        # action = env.action_space.sample()
        # if method == "continuous":
        #     action = [action[0]]

        #prev_state, action, reward, new_state, done = agent.step(0, 0)
        action = agent._get_policy_action(env.env.state)
        action = int((np.round(action[0])))
        observation, reward, done, info = env.step(action)

        #print(action)
        #print observation
        if done:
            print ("reward:", reward)
            num_steps.append(t)
            print("Episode finished after {} timesteps".format(t+1))
            break
