import numpy as np

from algcol.experiment.agent import Agent

class S_Agent(Agent):
    '''
    Note: Check next action
    '''
    def __init__(self, alpha, gamma, policy, policy_parameter, demand_object, n_runs, learningrate_decay,
                 num_opponents, egreedy_decay=None):
        super().__init__(alpha, gamma, policy, policy_parameter, demand_object, n_runs, learningrate_decay,
                         num_opponents, egreedy_decay)

    def _egreedy(self, agent_state):
        # Unpack agent state
        s_old, a_old, r, sn, _ = agent_state

        # Determine whether to explore or the exploit
        nature = np.random.choice(2, 1, p=[self.p[0], 1 - self.p[0]])[0]

        # Based on the 'draw of nature' follow the policy (1: exploit, 0: explore)
        if nature == 1:
            s = list(self.qmatrix[sn])
            a = s.index(max(s))

            # If multiple estimates with same value, pick one at random
            value_opt_action = s[a]

            if s.count(value_opt_action) > 1:

                opt_values = []
                for x in range(self.m):
                    if s[x] == value_opt_action:
                        opt_values.append(x)

                a = np.random.choice(opt_values)
            policy = 1
        else:
            a = np.random.randint(self.m)
            policy = 0

        # Use the actual action to perform the update step
        td_error = r + self.g*self.qmatrix[sn][a] - self.qmatrix[s_old][a_old]
        self.td_error.append(td_error)

        # Update estimated Q-value
        self.qmatrix[s_old][a_old] += self.a[0] * np.array(td_error, dtype=np.float64)

        return a, policy

    def _update_step(self, agent_state):
        self._update_policyparameter()
        self._update_decayschedule(method=self.learning, parameter='a')


