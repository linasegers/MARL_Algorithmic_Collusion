import numpy as np

from algcol.experiment.agent import Agent

class QL_Agent(Agent):
    '''
    Implementation of Watkin's Q: https://www.cs.utexas.edu/~pstone/Courses/394Rfall16/resources/week6-sutton.pdf
    '''
    def __init__(self, alpha, gamma, policy, policy_parameter, demand_object, n_runs, learningrate_decay,
                 num_opponents, lambda_trace, egreedy_decay=None):
        super().__init__(alpha, gamma, policy, policy_parameter, demand_object, n_runs, learningrate_decay,
                         num_opponents, egreedy_decay)

        self.traces = np.array([[0] * self.m]*(self.m**self.agents))
        self.zeros = np.array([[0] * self.m]*(self.m**self.agents))
        self.l = lambda_trace

        self.original_values['traces'] = self.traces

    def _initialize_agent(self, initialization="myopic", decay_learning=1, decay_exploration=0.8,
                          pretrained=False, location=None, experiment_identifier=None, experiment_name=None,
                          random=True, random_idx=None):
        super(QL_Agent, self)._initialize_agent(initialization, decay_learning, decay_exploration,
                                                pretrained, location, experiment_identifier, experiment_name,
                                                random, random_idx)
        self.qmatrix = np.array(self.qmatrix, dtype='float64')
        if isinstance(self.traces, list):
            self.traces = np.array(self.traces)

    def _update_step(self, agent_state):
        '''
        '''
        # Unpack agent state
        s, a, r, sn, _ = agent_state

        # If the action used was explorative, remove the traces
        if self.qmatrix[s][a] != max(self.qmatrix[s]):
            self.traces = self.zeros
        else:
            # Update the traces
            self.traces = np.multiply(self.traces, (self.g * self.l))

        # Calculate TD error of Q-learning (target policy is greedy)
        td_error = r + self.g * self.qmatrix[sn][np.where(self.qmatrix[sn] == max(self.qmatrix[sn]))[0][0]] - self.qmatrix[s][a]
        self.td_error.append(td_error)

        # Update estimated Q-values
        self.traces[s][a] = np.float(1)
        self.qmatrix += ((self.a[0] * td_error) * self.traces)

        self._update_policyparameter()
        self._update_decayschedule(method=self.learning, parameter='a')