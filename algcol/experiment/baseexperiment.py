import numpy as np
import os
from copy import deepcopy

from algcol.shared.load import load_pickle
from algcol.experiment.experimentconstructor import _ExperimentConstructor

####
class BaseExperiment(_ExperimentConstructor):
    '''
    Note: check if you can make check bm part of initialization, makes more sense
    '''
    def __init__(self, identifier, setting, number_of_agents, alpha, gamma, policy, policy_parameter, lambda_trace,
                 learningrate_decay, egreedy_decay, n_runs, environment=None, price_intervals=None, utility=None,
                 h_diff=None, firmcost=None, demand=None, strategy=None, BM_increment=False):
        super().__init__(identifier, setting, number_of_agents, alpha, gamma, policy, policy_parameter, lambda_trace,
                         learningrate_decay, egreedy_decay, n_runs, environment, price_intervals, utility, h_diff,
                         firmcost, demand, strategy, BM_increment)

    def _random_first_state(self):
        # Initialize the first state by randomly selecting actions
        actions = [np.random.randint(0, self.m) for _ in range(self.agents)]

        random_state = [[0], actions, [0]*self.agents, actions, self.env.env[actions[0]][actions[1]]]

        self.history.append(random_state)

    def _predefined_first_state(self, first_state=None, location=None, same_opponent=True, random_idx=None):
        if first_state is not None and isinstance(first_state, list):
            fs = first_state
        else:
            home = os.getcwd()

            # If homogeneous training setting true, loc is normal place
            if same_opponent is True:
                loc = os.path.join(location, f"{''.join(self.setting)}/short_environment")
                if not os.path.exists(loc):
                    loc = os.path.join(location, f"{''.join(self.setting)}/full_environment")
            else:  # Else save a list with agent locations
                loc = []
                for x in range(self.agents):
                    l = os.path.join(location, f"{self.setting[x]*self.agents}/short_environment")
                    if not os.path.exists(l):
                        l = os.path.join(location, f"{self.setting[x]*self.agents}/full_environment")
                    loc.append(l)

            if first_state is 'only_converged_series':
                os.chdir(loc)

                pols = []
                acts = []
                for x in range(self.agents):
                    acts.append(load_pickle(f'{self.identifier}_agent{x}_action.pkl'))
                    pols.append(load_pickle(f'{self.identifier}_agent{x}_policy.pkl'))

                x = 1
                p0, p1 = [0, 0]
                while p0 is 0 or p1 is 0:
                    p0 = pols[0][-x]
                    p1 = pols[1][-x]
                    x += 1

                a0 = [acts[0][-(x + 1)], acts[1][-(x + 1)]]
                a1 = [acts[0][-(x)], acts[1][-(x)]]

                fs = [[[0], a0, [1] * self.agents, a1, self.env.env[a1[0]][a1[1]]]]
            else:
                # Get a list with the actions
                actions = []
                for x in range(self.agents):
                    if isinstance(loc, list):
                        print(loc[x])
                        os.chdir(loc[x])
                    else:
                        print(loc[x])
                        os.chdir(loc)

                    if random_idx is not None:
                        name = f'{self.identifier}_agent{random_idx[x]}_action.pkl'
                    else:
                        name = f'{self.identifier}_agent{x}_action.pkl'

                    action = load_pickle(name)[-2:]
                    actions.append(action)

                # State
                a0 = [actions[x][0] for x in range(self.agents)]
                a1 = [actions[x][1] for x in range(self.agents)]
                fs = [[[0], a0, [1]*self.agents, a1, self.env.env[a1[0]][a1[1]]]]

            os.chdir(home)

        self.history += fs

    def initialize_experiment(self, initialization='myopic', location=None, first_state=None,
                              random=True, homogeneous_training_setting=True, decay_learning=1, decay_exploration=0.8):
        '''
        Note: to set the agent initialization add the key-word
        Add the type of agent updating here (so if you want linear, constant, exponential decay)
        Also add this later on here with respect to the pre-trained values
        Note: For the homogeneous pretrained setting, the agent in pretrained version is randomized
        '''
        # Initialize environment
        np.random.seed(self.identifier)

        if initialization is not 'pretrained':
            self._random_first_state()

            # Initialize the agents
            for x in range(self.agents):
                vars(self)[self.agent_names[x]]._initialize_agent(initialization=initialization,
                                                                  decay_learning=decay_learning,
                                                                  decay_exploration=decay_exploration)
                vars(self)[self.agent_names[x]].id_agent = x
        else:
            loc = location+f'data_experiment/'
            rand_idx = [np.random.randint(0, self.agents) for _ in range(self.agents)]
            for x in range(self.agents):
                # Set pretrained values of the exact same setting specification
                if homogeneous_training_setting is False:
                    vars(self)[self.agent_names[x]]._initialize_agent(location=os.path.join(loc, f'{"".join(self.setting)}/agent{x}'),
                                                                      initialization='pretrained',
                                                                      decay_learning=decay_learning,
                                                                      decay_exploration=decay_exploration,
                                                                      experiment_identifier=self.identifier,
                                                                      id=x)
                else: # Set pretrained values based on homogeneous training setting
                    if random:
                        id = rand_idx[x]
                    else:
                        id = x
                    vars(self)[self.agent_names[x]]._initialize_agent(location=os.path.join(loc, f'{self.setting[x]*self.agents}/agent{id}'),
                                                                      initialization='pretrained',
                                                                      decay_learning=decay_learning,
                                                                      decay_exploration=decay_exploration,
                                                                      id=id,
                                                                      experiment_identifier=self.identifier)
                vars(self)[self.agent_names[x]].id_agent = x

            self._predefined_first_state(first_state=first_state, location=loc,
                                         same_opponent=not homogeneous_training_setting,
                                         random_idx=rand_idx)

    def _market_interaction(self, actions, policies):
        '''
        History:    0:Time
                    1: Previous actions (S_t)
                    2: Policies
                    3: New actions (S_{t+1})
                    4: Return(R_t)
        '''
        # Get indices of the states
        St = self.history[self.t][3]
        Rt = self.env.env[actions[0]][actions[1]]

        # Create the generic state (PREV A, POLICY, A, RETURN)
        state = [St, policies, actions, Rt]

        self.t += 1 # Increase env time
        self.history.append([[self.t]] + state)

        s0 = (St, actions[0], Rt[0], actions)
        s1 = (St, actions[1], Rt[1], actions)

        self.agent0._update_step(s0) # Increase agent time
        self.agent1._update_step(s1)

        a0, p0 = self.agent0._policy_selection(s0)
        a1, p1 = self.agent1._policy_selection(s1)

        return [a0, a1], [p0, p1]

    def _first_run(self):
        '''
        Based on the first state, choose actions but do not do any updates
        '''
        # History: 0:time, 1: previous actions, 2: policy, 3: new actions, 4:return
        actions = self.history[self.t][3]
        Rt = self.env.env[actions[0]][actions[1]]

        a0, p0 = self.agent0._policy_selection((actions, actions[0], Rt[0], actions))
        a1, p1 = self.agent1._policy_selection((actions, actions[1], Rt[1], actions))

        return [a0, a1], [p0, p1]

    def training(self, convergence='fixed_time_period', number_of_runs=None, max_runs=1000000, convergence_runs=100000):
        # First run
        actions, policies = self._first_run()

        if convergence is 'fixed_time_period':

            if number_of_runs is None:
                number_of_runs = self.runs

            # Train until number of n_runs is reached
            stop_criteria = self.t + number_of_runs
            while self.t <= stop_criteria:
                actions, policies = self._market_interaction(actions, policies)

        elif convergence is 'stabilize_all_actions':
            l = [0] * self.m
            agent0 = [deepcopy(l) for _ in range(self.m)]
            agent1 = [deepcopy(l) for _ in range(self.m)]

            check0 = 0
            check1 = 0

            while check0 < convergence_runs or check1 < convergence_runs:
                if self.t > max_runs:
                    break

                a, p = self._market_interaction(actions, policies)

                if agent0[actions[0]][actions[1]] == a[0]:
                    check0 += 1
                else:
                    check0 = 0
                    agent0[actions[0]][actions[1]] = a[0]

                if agent1[actions[0]][actions[1]] == a[1]:
                    check1 += 1
                else:
                    check1 = 0
                    agent1[actions[0]][actions[1]] = a[1]

                del actions
                actions = a

        elif convergence is 'stabilize_optimal_actions':
            l = [0] * self.m
            agent0 = [deepcopy(l) for _ in range(self.m)]
            agent1 = [deepcopy(l) for _ in range(self.m)]

            check0 = 0
            check1 = 0

            while check0 < convergence_runs or check1 < convergence_runs:
                if self.t > max_runs:
                    break

                a, policies = self._market_interaction(actions, policies)

                s0 = self.agent0.qmatrix[actions[0]][actions[1]]
                if s0.index(max(s0)) == agent0[actions[0]][actions[1]]:
                    check0 += 1
                else:
                    check0 = 0
                    agent0[actions[0]][actions[1]] = a[0]

                s1 = self.agent1.qmatrix[actions[0]][actions[1]]
                if s1.index(max(s1)) == agent1[actions[0]][actions[1]]:
                    check1 += 1
                else:
                    check1 = 0
                    agent1[actions[0]][actions[1]] = a[1]

                del actions
                actions = a
        else:
            raise NotImplementedError('The chosen option has not been implemented.')

    def obtain_convergence_data(self, location=None, runs=10000):
        '''
        Go to the state where both agents were playing greedy and run from there with zero exploration etc to obtain
        cleaned data
        '''
        p0 = 0
        p1 = 0

        # Get the data in case you have not saved the session yet
        if location is None and hasattr(self, 'history'):
            t = 1
            while p0 is 0 or p1 is 0:
                p0, p1 = self.history[-(t-1)][2]
                t += 1

            self.history = self.history[-t]

            for x in self.agent_names:
                vars(self)[x].p = [0, 0]
                vars(self)[x].epsilondecay = 'constant'
        else:
            self.initialize_experiment(initialization='pretrained', location=location, first_state='only_converged_series',
                                       random=False, homogeneous_training_setting=False, decay_learning=1, decay_exploration=0)

        self.training(convergence='fixed_time_period', number_of_runs=runs)




