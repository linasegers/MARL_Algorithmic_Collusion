import numpy as np

from algcol.experiment.agent import Agent

class DQ_Agent(Agent):
    '''
    Implementation of double Q-Learning

    Note: Boltzmann implementation is not supported yet. To support this, rewrite the policy selection function as it is
    done for the e-greedy policy.
    '''
    def __init__(self, alpha, gamma, policy, policy_parameter, demand_object, n_runs, learningrate_decay,
                 num_opponents, egreedy_decay=None):
        super().__init__(alpha, gamma, policy, policy_parameter, demand_object, n_runs, learningrate_decay,
                         num_opponents, egreedy_decay)

    def _initialize_agent(self, initialization="myopic", decay_learning=1, decay_exploration=0.8,
                          pretrained=False, location=None, experiment_identifier=None, experiment_name=None,
                          random=True, random_idx=None):
        super(DQ_Agent, self)._initialize_agent(initialization, decay_learning, decay_exploration,
                          pretrained, location, experiment_identifier, experiment_name, random, random_idx)
        self.qmatrix = np.array(self.qmatrix)

        if pretrained is False:
            setattr(self, 'qmatrix2', self.qmatrix)
        else:
            self.qmatrix2 = np.array(self.qmatrix2)

        self.original_values['qmatrix'] = self.qmatrix
        self.original_values['qmatrix2'] = self.qmatrix

    def _egreedy(self, agent_state):
        _, _, _, state, _ = agent_state

        # Determine whether to explore or the exploit
        nature = np.random.choice(2, 1, p=[self.p[0], 1-self.p[0]])[0]

        # Based on the 'draw of nature' follow the policy (1: exploit, 0: explore)
        if nature == 1:
            s1 = self.qmatrix[state]
            s2 = self.qmatrix2[state]
            s = s1 + s2

            a = np.where(s == max(s))[0]

            if len(a) > 1:
                a = np.random.choice(a)
            else:
                a = a[0]
            policy = 1
        else:
            a = np.random.randint(self.m)
            policy = 0

        return a, policy

    def _update_step(self, agent_state):
        '''
        Note: If the Q-matrix ties, now you always take the first one
        '''
        # Unpack agent state
        s, a, r, sn, _ = agent_state

        # Calculate TD error of Q-learning
        nature = np.random.choice(2, 1, p=[0.5, 0.5])

        if nature == 0:
            # Update the first qmatrix
            td_error = r + self.g*self.qmatrix2[sn][np.where(self.qmatrix[sn] == max(self.qmatrix[sn]))[0][0]] - self.qmatrix[s][a]
            self.qmatrix[s][a] += self.a[0] * np.array(td_error, dtype=np.float64)
        else:
            # or update the second qmatrix
            td_error = r + self.g * self.qmatrix[sn][np.where(self.qmatrix2[sn] == max(self.qmatrix2[sn]))[0][0]] - self.qmatrix2[s][a]
            self.qmatrix2[s][a] += self.a[0] * np.array(td_error, dtype=np.float64)

        self.td_error.append(td_error)

        self._update_policyparameter()
        self._update_decayschedule(method=self.learning, parameter='a')