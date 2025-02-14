import numpy as np
from math import exp

####
class _EnvironmentConstructor:
    '''
    Backend of the class defining the specification of demand
    '''
    def __init__(self, price_intervals, utility, h_diff, firmcost, demand, agents):
        self.m = price_intervals
        self.h_diff = h_diff
        self.c = firmcost
        self.utility = utility
        self.agents = agents

        self.pb = None
        self.pm = None
        self.monopolyprofit = None
        self.bertrandprofit = None

        self.actions = None
        self.env = None
        self.full_environment = None

        if self.agents is None:
            try:
                self.agents = len(self.utility)
            except TypeError:
                self.agents = 1

        if isinstance(self.utility, int):
            self.utility = [self.utility] * self.agents

        self._demand = {'logit': self._logit,
                        'linear': self._linear,
                        'testlogit': self._testlogit}[demand]

    def _logit(self, prices):
        '''
        Calculate the quantity (outside good is 1)
        '''
        opps = np.exp(np.divide(np.subtract(self.utility, prices), self.h_diff))
        quantities = np.divide(opps, (1+sum(opps)))

        return quantities

    def _testlogit(self, prices):
        '''
        Calculate the quantity (outside good is exp(1))
        '''
        opps = np.exp(np.divide(np.subtract(self.utility, prices), self.h_diff))
        quantities = np.divide(opps, (exp(1)+sum(opps)))

        return quantities

    def _linear(self):
        raise NotImplementedError('This option is not available.')