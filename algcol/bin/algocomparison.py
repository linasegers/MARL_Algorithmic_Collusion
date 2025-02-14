from algcol.experiment.interaction import BaseExperiment

####
class _AlgorithmicComparison(BaseExperiment):
    '''
    '''
    def __init__(self, identifier, setting, number_of_agents, alpha, gamma, policy, policy_parameter, lambda_trace,
                 learningrate_decay, egreedy_decay, n_runs, environment=None, price_intervals=None, utility=None,
                 h_diff=None, firmcost=None, demand=None, strategy=None, BM_increment=False):
        super().__init__(identifier, setting, number_of_agents, alpha, gamma, policy, policy_parameter, lambda_trace,
                         learningrate_decay, egreedy_decay, n_runs, environment, price_intervals, utility, h_diff,
                         firmcost, demand, strategy, BM_increment)

    def rms_efficiency_test(self):
        print('todo')

    def bandit_game(self):
        print('todo')