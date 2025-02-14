from algcol.experiment.agentcontstructor import _AgentConstructor

class Q_Agent(_AgentConstructor):
    '''
    Implementation of Q-Learning agent. 
    '''
    def __init__(self, alpha, gamma, policy, policy_parameter, demand_object, n_runs, learningrate_decay,
                 num_opponents, egreedy_decay):
        super().__init__(alpha, gamma, policy, policy_parameter, demand_object, n_runs, learningrate_decay,
                         num_opponents, egreedy_decay)

    def _update_step(self, agent_state):
        # Unpack agent state
        s, a, r, sn = agent_state

        # Calculate TD error of Q-learning
        td_error = r + self.g*max(self.qmatrix[sn[0]][sn[1]]) - self.qmatrix[s[0]][s[1]][a]
        self.td_error.append(td_error)

        # Update estimated Q-value
        self.qmatrix[s[0]][s[1]][a] += self.a[0] * td_error

        self._update_policyparameter()
        self._learning_rate_update()

