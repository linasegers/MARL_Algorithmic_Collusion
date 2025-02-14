from algcol.experiment.gamesettings import _GameSettings
from algcol.experiment.algocomparison import _AlgorithmicComparison

####
class FullExperiment(_GameSettings, _AlgorithmicComparison):
    '''
    Note: Fix all init files
    '''
    def __init__(self, identifier, price_intervals, utility, h_diff, firmcost, demand, setting, number_of_agents, alpha,
                 gamma, policy, policy_parameter, lambda_trace, learningrate_decay, egreedy_decay, n_runs,
                 strategy=None, BM_increment=False):
        super().__init__(identifier, price_intervals, utility, h_diff, firmcost, demand, setting, number_of_agents, alpha,
                         gamma, policy, policy_parameter, lambda_trace, learningrate_decay, egreedy_decay, n_runs,
                         strategy, BM_increment)


