import numpy as np
import pandas as pd
import os
from copy import deepcopy

from algcol.shared.save import save_pickle
from algcol.shared.load import load_pickle

####
class _AgentConstructor:
    '''
    Class for all generic agent code
    Note: Additional policy functions: https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py
    '''
    def __init__(self, alpha, gamma, policy, policy_parameter, environment_object, n_runs, learningrate_decay,
                 num_opponents, egreedy_decay):

        self.demand = environment_object
        self.m = self.demand.m

        # Instantiation of policy and update rules
        #self._policy_selection = {'egreedy': self._egreedy,
                                #'boltzmann': self._boltzmann}[policy]

        #self._update_policyparameter = {'egreedy': self._egreedy_update,
                                  #'boltzmann': self._boltzmann_update}[policy]

        #self._set_parameters = {'egreedy': self._egreedy_setparameters,
                                  #'boltzmann': self._boltzmann_setparameters}[policy]

        self.policy = policy
        self.p = [policy_parameter, policy_parameter]
        self.epsilondecay = egreedy_decay

        self.a = [alpha, alpha]
        self.learning = learningrate_decay

        self.n_runs = n_runs
        self.g = np.array(gamma, dtype=np.float64)
        self.t = 0
        self.td_error = []
        self.method = None
        self.actions = None
        self.qmatrix = None
        self.id_agent = None
        self.agents = num_opponents + 1

        self.original_values = {}

    def _qmatrix(self, initialization):
        # Initialization Q-matrix, throw error if unknown option is chosen
        options = ["zeros", "random", "myopic", "optimistic", "opp_rand_uniform"]
        if not isinstance(initialization, (float, int)) and initialization not in options:
            raise Exception(f'You need to specify how the q-matrix should be initialized, options are: {options}')

        if initialization in ["zeros", "myopic", "optimistic"] or isinstance(initialization, (float, int)):
            vals = {'zeros': 0,
                    'myopic': self.demand.bertrandprofit / (1 - self.g),
                    'numeric': initialization,
                    'optimistic': self.demand.monopolyprofit / (1 - self.g)}

            if not isinstance(initialization, str):
                val = vals['numeric']
            else:
                val = vals[initialization]

            l = [val for _ in range(self.m)]
            qmatrix = [deepcopy(l) for _ in range(self.m ** self.agents)]

        elif initialization is 'opp_rand_uniform':
            # Regardless of the state, you want to know the value that would accrue if one agent plays a_i and the other
            # player randomizes uniformly. So you take all the profits for the case of a_i and all actions A of player j,
            # and then use the value function to get the perpetuity (approach p. 11 Calvano)
            value = lambda p: np.sum(np.divide(p, (1-self.g))) / self.m
            profs = [[self.demand.env[a][x][0] for x in range(self.m)] for a in range(self.m)]
            p = [value(profs[x]) for x in range(len(profs))]
            qmatrix = [deepcopy(p) for _ in range(self.m ** self.agents)]

        else:
            # Get bound of Q-values (perpetuity of Bertrand and Monopoly profit)
            lower_bound = self.demand.bertrandprofit / (1 - self.g)
            upper_bound = self.demand.monopolyprofit / (1 - self.g)
            qmatrix = [np.random.uniform(lower_bound, upper_bound + 0.0001, self.m).tolist() for _ in range(self.m**self.agents)]

        listdivider = lambda l, n: [l[i:i + n] for i in range(0, len(l), n)]

        x = 0
        while x < self.agents:
            qmatrix = listdivider(qmatrix, self.m)
            x += 1

        self.qmatrix = qmatrix[0]
        self.original_values['qmatrix'] = self.qmatrix

    def _set_learning_rate(self, decay_length):
        options = ['linear', 'constant']
        if self.learning not in options:
            Exception(f'The option for parameter updating must be of {options}')

        if self.learning == 'linear':
            tot = self.n_runs * decay_length
            one_period = self.a[0] / tot
            self.a = [self.a[0], one_period]

    def _learning_rate_update(self):
        if self.learning == 'linear':
            self.a[0] -= self.a[1]
            if self.a[0] < 0: self.a[0] = 0

    # def _egreedy_setparameters(self, decay_length):
    def _set_parameters(self, decay_length):
        options = ['linear', 'constant', 'exponential']
        if self.epsilondecay not in options:
            Exception(f'The option for parameter updating must be of {options}')

        if self.epsilondecay == 'linear':
            tot = self.n_runs * decay_length
            one_period = self.p[0] / tot
            self.p = [self.p[0], one_period]
        if self.epsilondecay == 'exponential':
            self.p[0] = 0.9999999999

    #def _egreedy_update(self):
    def _update_policyparameter(self):
        '''
        Note: For each decay schedule, param[0] is the current version and param[1] is the helper
        '''
        self.t += 1
        if self.epsilondecay == 'linear':
            self.p[0] -= self.p[1]
            if self.p[0] < 0: self.p[0] = 0
        elif self.epsilondecay == 'exponential':
            self.p[0] = np.exp(-self.p[1]*self.t)

    #def _egreedy(self, agent_state):
    def _policy_selection(self, agent_state):
        _, _, _, state = agent_state

        # Determine whether to explore or the exploit
        nature = np.random.choice(2, 1, p=[self.p[0], 1-self.p[0]])[0]

        # Based on the 'draw of nature' follow the policy (1: exploit, 0: explore)
        if nature == 1:
            s = self.qmatrix[state[0]][state[1]]
            a = s.index(max(s))

            # If multiple estimates with same value, pick one at random
            value_opt_action = s[a]

            if s.count(value_opt_action) > 1:

                opt_values = [x for x in range(self.m) if s[x] == value_opt_action]
                a = np.random.choice(opt_values)

            policy = 1
        else:
            a = np.random.randint(self.m)
            policy = 0

        return a, policy

    def _reset_agent(self):
        for x in list(self.original_values.keys()):
            setattr(self, x, self.original_values[x])

    def save_agent(self, location, filename):
        os.chdir(location)

        # Save relevant attributes of each agent
        objs = ['td_error', 'qmatrix', 'qmatrix2', 'traces']

        for o in objs:

            if hasattr(self, o):
                obj = vars(self)[o]
                save_pickle(obj, f'{filename}_{o}.pkl')

    def _initialize_agent(self, initialization="myopic", decay_learning=1, decay_exploration=0.8, location=None,
                          experiment_identifier=None, id=None):
        # Initialize action space
        self.actions = self.demand.actions

        # Initialize the type of parameter updates (default is linear)
        self._set_learning_rate(decay_length=decay_learning)
        self._set_parameters(decay_length=decay_exploration)

        # Initialize either the type of Q-Matrix initialization, or load pretrained values
        if initialization is not 'pretrained':
            self._qmatrix(initialization)
        else:
            home = os.getcwd()
            os.chdir(location)

            # Import the Q-matrix
            values = ['qmatrix', 'traces', 'qmatrix2']

            # Index of the agent (in case of random agent chosen)
            if id is None:
                id = self.id_agent

            for x in values:
                if hasattr(self, x):
                    obj = load_pickle(f'{experiment_identifier}_agent{id}_{x}.pkl')
                    if isinstance(obj, pd.DataFrame):
                        obj = obj.values.tolist()
                    setattr(self, x, obj)
                    self.original_values[x] = obj

            # Return to home directory
            os.chdir(home)
