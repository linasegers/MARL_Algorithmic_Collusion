import numpy as np
import warnings, itertools

from algcol.experiment.interaction import BaseExperiment
from algcol.experiment.environment import Environment

####
class _GameSettings(BaseExperiment):
    '''
    '''
    def __init__(self, identifier, setting, number_of_agents, alpha, gamma, policy, policy_parameter, lambda_trace,
                 learningrate_decay, egreedy_decay, n_runs, environment=None, price_intervals=None, utility=None,
                 h_diff=None, firmcost=None, demand=None, strategy=None, BM_increment=False):
        super().__init__(identifier, setting, number_of_agents, alpha, gamma, policy, policy_parameter, lambda_trace,
                         learningrate_decay, egreedy_decay, n_runs, environment, price_intervals, utility, h_diff,
                         firmcost, demand, strategy, BM_increment)

    def deviation(self, periods, agent=None, undercut=None):
        '''
        player:             The player that deviates, should be 0 for agent 1 and 1 for agent 2
        :param kwargs: agent -> Number to specify which agent will deviate, 0 is player 1, 1 is player 2. Default is
                                random selection
        '''
        # Randomly let one of the two agents deviate
        if agent is None:
            player = np.random.choice(self.agents)
            warnings.warn(f'No choice for deviating agent provided, resorting to {player}')
        else:
            player = agent

        setattr(self, 'deviating', player)

        # All previous actions opponent to build naive expectation on (most used action is expected action)
        opponent_actions = []
        for x in range(len(self.history)):
            hist = self.history[x][3]
            opponent_actions.append([hist[x] for x in range(self.agents) if x != player])

        temp_list = [[opponent_actions[y][x] for y in range(len(opponent_actions))] for x in range(self.agents-1)]

        exp_action = []
        for y in range(self.agents-1):
            expected_action = max(set(temp_list[y]), key=temp_list[y].count)
            exp_action.append(expected_action)

        # Given expected action of opponents, determine best response action for one-shot game deviating agent
        all = [[exp_action[x]] * vars(self)[self.agent_names[player]].m for x in range(self.agents-1)]
        all.append(list(range(vars(self)[self.agent_names[player]].m)))

        acts = list(itertools.product(*all))
        action_index = ["".join([str(x) for x in t]) for t in acts]

        returns = []
        for i in action_index:
            ret = self.env.env[self.state[i]][1][player]
            returns.append(ret)

        # Profit maximizing action for deviating agent
        exogenous_deviation_idx = action_index[returns.index(max(returns))]
        exogenous_deviation = int(exogenous_deviation_idx[player])

        starting_state = "".join([str(x) for x in self.history[-1][3]])
        true_actions = []
        for x in range(self.agents):
            a, _ = vars(self)[self.agent_names[x]]._policy_selection(self.state[starting_state])
            true_actions.append(a)

        # If optional undercut argument is set to true, all agents are forced to undercut
        if undercut is not None and true_actions[player] < exogenous_deviation:
            exogenous_deviation = true_actions[player] - 1

        true_actions[player] = exogenous_deviation
        # Note here reflects not true policy used, but is for plotting purposes
        policies = [1, 1, 1]
        policies[player] = 2

        # Now run the experiment with the true actions and the deviation
        self._market_interaction(actions=true_actions, policies=policies)

        # Given these actions, continue the game
        self.training(periods)

    def demand_shock(self, agent, shock, periods, type='all', demand='logit'):
        # Renew the values of the environment
        if not isinstance(self.utility, list):
            self.utility = [self.utility] * self.agents

        self.utility[agent] = self.utility[agent] + shock

        # Option in case the full environment, including for agents changes
        if type == 'all':
            delattr(self, 'env')
            setattr(self, 'env', Environment(5, self.utility, self.h_diff, demand=demand, agents=self.agents))
            self.env.environment(output=False, check_bm=None)
            self.state = self.env.idx

            for x in range(self.agents):
                delattr(vars(self)[self.agent_names[x]], 'demand')
                setattr(vars(self)[self.agent_names[x]], 'demand', self.env)
                vars(self)[self.agent_names[x]]._initialize_agent(initialization=None)
        # Option in case only the returns change
        else:
            delattr(self.env, 'env')
            self.env.environment(output=False, check_bm=None, demand_shock=shock, ag=agent)

        # Now simply run the game for the prespecified periods
        self.training(periods=periods)
