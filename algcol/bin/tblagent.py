import numpy as np

from algcol.experiment.agent import Agent

class TBL_Agent(Agent):
    '''
    Implementation of Tree-Backup lambda: https://web.eecs.umich.edu/~baveja/Papers/OffPolicy.pdf.

    Note: Currently only e-greedy strategy implementation
    '''
    def __init__(self, alpha, gamma, policy, policy_parameter, demand_object, n_runs, learningrate_decay,
                 num_opponents, lambda_trace, egreedy_decay=None):
        super().__init__(alpha, gamma, policy, policy_parameter, demand_object, n_runs, learningrate_decay,
                         num_opponents, egreedy_decay)
        self.traces = np.array([[0] * self.m]*(self.m**self.agents))
        self.p = np.array([[1] * self.m]*(self.m**self.agents))
        self.qmatrix = np.array(self.qmatrix)
        self.l = lambda_trace

    def _update_step(self, agent_state):
        '''
        Note: This is the update for the e-greedy strategy, where the target policy means maximizing
        Note: Only trace implementation is replacing traces, for inspiration look at the slides below:
        https://www.cs.utexas.edu/~pstone/Courses/394Rfall16/resources/week6-sutton.pdf

        Note: E-greedy target means p=1 for max, p=0 for rest. Ties in max values are now still given p=1
        '''
        # Unpack agent state
        s, a, r, sn, _ = agent_state

        # Update the traces
        self.traces = np.multiply(self.traces, self.p)*self.g*self.l
        self.traces[s][a] = 1

        # Calculate TD error of Q-learning (target policy is greedy)
        td_error = r + self.g*max(self.qmatrix[sn]) - self.qmatrix[s][a]
        self.td_error.append(td_error)

        # Update estimated Q-values
        self.qmatrix += (self.a * td_error * self.traces)

        # Consecutively, update the probabilities of playing under the target value (max = 1, 0 otherwise)
        idx = np.where(np.array(self.qmatrix[s]) == max(self.qmatrix[s]))[0]
        self.p[s].fill(0)
        for x in range(len(idx)):
            self.p[s][idx[x]] = 1

        self._update_policyparameter()
        self._update_decayschedule(method=self.learning, parameter='a')