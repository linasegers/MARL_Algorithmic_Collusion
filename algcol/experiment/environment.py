import numpy as np
from scipy.optimize import minimize
import warnings

from algcol.experiment.environmentconstructor import _EnvironmentConstructor

####
class Environment(_EnvironmentConstructor):
    '''
    Current implementation: All agents use an identical economic environment.
    '''
    def __init__(self, price_intervals, utility, h_diff, firmcost, demand, agents):
        super().__init__(price_intervals, utility, h_diff, firmcost, demand, agents)

    def _monopoly_price(self):
        '''
        The monopolist sets a price to optimize combined profit in the game
        '''
        def _optimization(p, c):
            quantities = self._demand(p)
            profit = np.multiply(quantities, (p-c))
            return -1 * np.sum(profit)

        opt = minimize(_optimization, self.utility, args=self.c)

        self.pm = opt.x[0]

    def _bertrand_price(self, tolerance):
        '''
        Check: https://github.com/microsoft/BertrandNashEquilibriumComputation
        '''
        from random import sample

        # Optimize following function:
        def _optimization(p, prices, agent, c):
            prices[agent] = p[0]  # Just to unlist
            quantities = self._demand(prices)
            profit = -1 * quantities[agent] * (p - c)
            return profit

        def _bestresponsedynamics(prices, agent, c):
            opt = minimize(_optimization, 0, args=(prices, agent, c))
            return opt.x[0]

        # Initialize setting
        stability = [100]*self.agents
        previous_prices = [[float(0)] for _ in range(self.agents)]
        ag = list(range(self.agents))

        # Loop over options until equilibrium: Each agent stabilizes behaviour
        x = 1
        while np.sum(stability) > self.agents:

            agents = sample(ag, self.agents)
            for agent in agents:
                new_price = _bestresponsedynamics([item[-1] for item in previous_prices], agent, c=self.c)
                previous_prices[agent].append(new_price)

            stability = [len(np.unique(np.round(previous_prices[x][-50:], tolerance))) for x in range(self.agents)]

            x += 1
            if x > 10000:
                warnings.warn(f'Optimization procedure interrupted, converging takes too many periods')
                break

        if len(np.unique([item[-1] for item in previous_prices])) > 1:
            warnings.warn(f'Seems like the Bertrand prices incorporate a rounding discrepancy, price are: {[item[-1] for item in previous_prices]}')

        self.pb = [item[-1] for item in previous_prices][0]

    def actionspace(self, BM_increment, output, tolerance):
        '''
        :param BM_increment: In case you want an additional increment above and below monopoly and bertrand price
        '''
        self._monopoly_price()
        self._bertrand_price(tolerance=tolerance)

        pb = self.pb
        pm = self.pm

        # Create action space: Either starting from B to M prices, or minus an increment
        if BM_increment is False:
            # Regular actionspace
            spaces = self.m - 1
            actionspace = [pb] + [(pb + ((pm - pb) * (x / spaces))) for x in range(1, spaces)] + [pm]
        elif BM_increment is 'including_pbpm':
            # Create action space with one increment below and above action space
            spaces = self.m - 3
            increment = (pm - pb) * (1 / spaces)
            actionspace = [pb - increment] + [pb] + [(pb + ((pm - pb) * (x / spaces))) for x in range(1, spaces)] + [pm] + [pm + increment]
        else:
            param = BM_increment
            if not isinstance(param, float):
                warnings.warn(f'No parameter value given for BM_increment, resorting to 0.1')
                param = 0.1
            first = pb - param*(pm-pb)
            last = pm + param * (pm - pb)
            spaces = self.m - 1
            actionspace = [first] + [(first + ((last - first) * (x / spaces))) for x in range(1, spaces)] + [last]

        self.actions = actionspace

        if output is True:
            return actionspace, pb, pm

    def environment(self, BM_increment, tolerance=5):

        from itertools import product

        self.actionspace(tolerance=tolerance, BM_increment=BM_increment, output=False)

        self.monopolyprofit = self._demand(self.pm)[0] * (self.pm - self.c)
        self.bertrandprofit = self._demand(self.pb)[0] * (self.pb - self.c)

        # Get all action combinations
        p = [self.actions] * self.agents
        prices = [list(x) for x in list(product(*p))]

        # For each combination of prices, get corresponding quantity, profit, normalized profit, normalized price
        q = [list(self._demand(x)) for x in prices]
        prof = [np.multiply(np.array(prices[x])-self.c, q[x]).tolist() for x in range(len(prices))]

        standardize = lambda x, b, m: (x - b) / (m - b)

        prof_s = [[standardize(y, self.bertrandprofit, self.monopolyprofit) for y in x] for x in prof]
        price_s = [[standardize(y, self.pb, self.pm) for y in x] for x in prices]

        full_environment = [[prices[x], q[x], prof[x], prof_s[x], price_s[x]] for x in range(len(prices))]
        env = prof.copy()

        listdivider = lambda l, n: [l[i:i + n] for i in range(0, len(l), n)]

        x = 0
        while x < self.agents:
            full_environment = listdivider(full_environment, self.m)
            env = listdivider(env, self.m)
            x += 1

        # Full environment: [prices, quantity, profit, standardized_profit, standardized price]
        self.env = env[0]
        self.full_environment = full_environment[0]

    def plot_environment(self, column, agent_number='both', title=''):
        if self.agents != 2:
            raise NotImplementedError(f'This option is currently only available for two agents')
        from algcol.shared.plot_environment import plot_environment
        plot_environment(column, agent_number=agent_number, title=title, environment=self)
