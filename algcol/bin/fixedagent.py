import warnings

from algcol.experiment.agent import Agent

####
class Fixed_Agent(Agent):
    '''
    Note: Tit for tat and direct price follower is not implemented
    Note: Initialization of agent still needs to be specified correctly when using this class (in experiment class)
    '''
    def __init__(self, alpha, gamma, policy, policy_parameter, demand_object, n_runs, strategy):
        super().__init__(alpha, gamma, policy, policy_parameter, demand_object, n_runs)

        self.strategy = strategy
        self.default = 5

        # Set the fixed policy selection
        self._policy_selection = self._fixed_policy_selection
        self._update_agentparameters = self._update_agent_fixed
        self._update_step = self._update_step_fixed

        # Thow an error if a strategy is chosen that does not exist
        options = ['fixed']
        if self.strategy not in options:
            if self.strategy in ['tit_for_tat', 'price_follower']:
                NotImplementedError(f'These strategies are not implemented yet')
            Exception(f'The strategy of the fixed strategy agent needs to be of {options}')

    def _update_agent_fixed(self):
        self.t += 1

    def _initialize_agent(self, initialization="myopic", check_BM=False, **kwargs):
        super(Fixed_Agent, self)._initialize_agent(initialization=initialization, check_BM=check_BM, **kwargs)
        if 'price' in kwargs:
            if kwargs['price'] not in list(range(self.m)):
                warnings.warn(f'This is not a possible price, so resorting to the default value '
                              f' {self.actions[self.default]}')
            else:
                self.default = kwargs['price']

    def _fixed_policy_selection(self, state):
        self._update_agentparameters()
        a = self.default
        return a, 1

    def _update_step_fixed(self, agent_state):
        pass

