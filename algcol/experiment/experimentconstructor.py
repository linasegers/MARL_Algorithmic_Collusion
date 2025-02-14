import pandas as pd
import os

from algcol.shared.save import save_pickle

from algcol.experiment.qagent import Q_Agent
#from algcol.experiment.tblagent import TBL_Agent
from algcol.experiment.esagent import ES_Agent
#from algcol.experiment.sagent import S_Agent
#from algcol.experiment.dqagent import DQ_Agent
#from algcol.experiment.qlagent import QL_Agent
#from algcol.experiment.fixedagent import Fixed_Agent
from algcol.experiment.environment import Environment

####
class _ExperimentConstructor:
    '''
    Watch out that the type of original values etc, depends on the agent type, so adjust to constructor to account for
    that

    Note: Need to adjust the pre-trained value settings before using them (set_pretrained values and reset) part of agent
    plus the update function still needs to be updated
    '''
    #agent_types = {'Sarsa': S_Agent,
                   #'ExpectedSarsa': ES_Agent,
                   #'QLearning': Q_Agent,
                   #'DoubleQLearning': DQ_Agent,
                   #'WatsonsQ': QL_Agent,
                   #'TBL': TBL_Agent,
                   #'FixedStrategy': Fixed_Agent}

    agent_types = {'ExpectedSarsa': ES_Agent, 'QLearning': Q_Agent}

    def __init__(self, identifier, setting, number_of_agents, alpha, gamma, policy, policy_parameter, lambda_trace,
                         learningrate_decay, egreedy_decay, n_runs, environment, price_intervals, utility, h_diff,
                         firmcost, demand, strategy, BM_increment):

        # Experiment: (PREV A, POLICY, A, RETURN)
        self.history = []
        self.t = 0

        if isinstance(environment, Environment):
            self.env = environment
        else:
            self.env = Environment(price_intervals, utility, h_diff, firmcost, demand, number_of_agents)
            self.env.environment(BM_increment=BM_increment)

        # Environment
        self.m = self.env.m
        self.utility = self.env.utility
        self.h_diff = self.env.h_diff
        self.c = self.env.c

        # Agents
        self.setting = setting
        self.agents = number_of_agents
        if not isinstance(self.setting, list):
            self.setting = [self.setting]*self.agents
        self.identifier = identifier
        self.agent_names = []
        self.runs = n_runs

        set1 = [alpha, gamma, policy, policy_parameter, self.env, n_runs, learningrate_decay,
                         self.agents-1, egreedy_decay]
        set2 = [alpha, gamma, policy, policy_parameter, self.env, n_runs, learningrate_decay,
                         self.agents-1, lambda_trace, egreedy_decay]

        params = {'QLearning': set1,
                  'ExpectedSarsa': set1,
                  'TBL': set2,
                  'Sarsa': set1,
                  'DoubleQLearning': set1,
                  'WatsonsQ': set2,
                  'FixedStrategy': [alpha, gamma, policy, policy_parameter, self.env, n_runs, strategy]}

        # Create the agent objects according to the setting specification
        for x in range(self.agents):
            name_obj = f'agent{x}'
            p = params[self.setting[x]]

            # This ensures if a list of values is provided, different values are set
            parameters = []
            for i in p:

                if i is None:
                    continue
                if isinstance(i, (int, float, str, Environment)):
                    parameters.append(i)
                else:
                    if len(i) != self.agents:
                        parameters.append(i[0])
                    else:
                        parameters.append(i[x])

            setattr(self, name_obj, self.agent_types[self.setting[x]](*parameters))
            self.agent_names.append(name_obj)

    def reset(self, keep_runs_hist=False, reset_agent=True):
        # Reset the agents
        if reset_agent is True:
            for x in range(self.agents):
                vars(self)[self.agent_names[x]]._reset_agent()

        # Remove part of the history that was created after initialization
        if keep_runs_hist is False:
            keep = 1
        else:
            keep = keep_runs_hist
        self.history = self.history[:-keep]

        # Reset the time indices
        self.t = keep
        for x in range(self.agents):
            vars(self)[self.agent_names[x]].t = keep

    def _grab_history(self, part, agent):
        hist_idx = {'time': 0, 'prev_action': 1, 'policy': 2, 'action': 3, 'return': 4}

        if isinstance(part, str):
            part = hist_idx[part]

        hist = [sublist[part][agent] for sublist in self.history]

        return hist

    def save_session(self, location, foldername=None, save_fullhistory=True, save_shorthistory=True, finalvalues=10000):
        '''
        Function to save the session output
        '''
        # Auxiliary functions
        def saveasseries(history, location=None):
            if location is not None:
                os.chdir(location)

            cols = ['policy', 'action', 'return']

            for x in range(len(cols)):

                for y in range(self.agents):
                    obj = [hist[x + 2][y] for hist in history]
                    save_pickle(obj, f'{self.identifier}_agent{y}_{cols[x]}.pkl')

        home = os.getcwd()

        # Create the location
        if foldername is None:
            path = f'{location}/data_experiment/{"".join(self.setting)}'
        else:
            path = f'{location}/data_experiment/{foldername}'

        locations = {'full_environment' : os.path.join(path, 'full_environment'),
                     'short_environment': os.path.join(path, 'short_environment')}

        for x in range(self.agents):
            locations[f'loc_{self.agent_names[x]}'] = os.path.join(path, self.agent_names[x])

        for value in locations.values():
            if not os.path.isdir(value):
                os.makedirs(value)

        os.chdir(path)

        # Store the objects
        for y in range(self.agents):
            vars(self)[self.agent_names[y]].save_agent(location=locations[f'loc_{self.agent_names[y]}'],
                                                       filename=f'{self.identifier}_{self.agent_names[y]}')

        if save_fullhistory:
            saveasseries(history=self.history, location=locations['full_environment'])

        if save_shorthistory:
            saveasseries(history=self.history[-finalvalues:], location=locations['short_environment'])

        os.chdir(home)


