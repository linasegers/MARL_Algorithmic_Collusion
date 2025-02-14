import numpy as np

from algcol.experiment.agentcontstructor import _AgentConstructor

class ES_Agent(_AgentConstructor):
    '''
    Note: Update parameters e and a should be loose
    '''
    def __init__(self, alpha, gamma, policy, policy_parameter, demand_object, n_runs, learningrate_decay,
                 num_opponents, egreedy_decay=None):
        super().__init__(alpha, gamma, policy, policy_parameter, demand_object, n_runs, learningrate_decay,
                         num_opponents, egreedy_decay)

        self._EV = {'egreedy': self._EV_egreedypolicy,
                    'boltzmann': self._EV_boltzmann}[policy]

    def _EV_egreedypolicy(self, state_idx):
        state = self.qmatrix[state_idx[0]][state_idx[1]]

        # identify actions that yield optimal value
        value_opt_action = max(state)
        opt_values = state.count(value_opt_action)

        # Probability of playing maximum value and probability of the rest
        p_random = self.p[0] / self.m
        p_rest = p_random * (self.m - opt_values)
        p_max = (1 - self.p[0]) + (opt_values * p_random)

        # Calculate expected value based on the probabilities
        rest = list(filter((value_opt_action).__ne__, state))
        if len(rest) != 0:
            rest = np.mean(rest) * p_rest
        else:
            rest = 0
        maximum = value_opt_action * p_max

        return rest + maximum

    def _EV_boltzmann(self, state_idx):
        return np.sum(np.multiply(self.qmatrix[state_idx], self.p_helper[state_idx]))

    def _update_step(self, agent_state):
        self._update_policyparameter()

        # State -> EV -> TD-error -> Update step
        s, a, r, sn = agent_state

        expected_value = self._EV(sn)

        td_error = r + self.g * expected_value - self.qmatrix[s[0]][s[1]][a]
        self.td_error.append(td_error)

        self.qmatrix[s[0]][s[1]][a] += self.a[0] * td_error

        self._learning_rate_update()