import numpy as np
from scipy.stats import wilcoxon
import timeit, psutil, warnings, os

from algcol.shared.save import save_pickle

from algcol.experiment.environment import Environment
from algcol.analysis.load_series import load_series
from algcol.analysis.load_attributes import load_attributes

####
class _HelperAgent:
    def __init__(self, id, environment_object, location):
        self.id = id
        self.d = environment_object
        self.location = location

    def _attach_series(self, series, sample, type_history, name, subset):
        for y in series:
            if y not in ['stand_profit', 'price', 'stand_price']:
                setattr(self, y+name, np.array(load_series(serie=y, location=self.location, type_history=type_history,
                                                      agent=self.id, sample=sample, subset=subset)))
            elif y is 'stand_profit':
                if y is 'stand_profit' and not hasattr(self, 'profit'+name):
                    warnings.warn('Not able to calculate this information without the returns loaded')
                    break
                temp = np.divide((vars(self)['profit'+name] - self.d.bertrandprofit),
                                              (self.d.monopolyprofit-self.d.bertrandprofit))
                setattr(self, y + name, temp)
                del temp
            else:
                if y in ['stand_price', 'price'] and not hasattr(self, 'action'+name):
                    warnings.warn('Not able to calculate this information without the actions loaded')
                    break

                temp = vars(self)['action'+name].copy().astype('float16')

                prices = self.d.actions
                actions = list(range(len(prices)))

                for x in range(len(actions)):
                    temp[temp == actions[x]] = prices[x]

                if y is 'stand_price':
                    temp = np.divide((temp - self.d.pb), (self.d.pm - self.d.pb))

                setattr(self, y+name, temp)
                del temp

            print(f'Attached: {y+name} for agent {self.id}')

    def _optimal_policy(self, shortened_history=False, sample=False, convergence_period=100000):
        name=""
        hist = "full_environment"
        if shortened_history:
            name = "_short"
            hist = "short_environment"

        if not hasattr(self, 'policy'+name):
            setattr(self, 'policy'+name, np.array(load_series(serie='policy'+name, location=self.location, type_history=hist,
                                                         agent=self.id, sample=sample)))

        pols = np.array([vars(self)['policy'+name][y][-(convergence_period*3):] for y in range(len(vars(self)['policy'+name]))])
        pols = np.where(pols == 1, True, False)

        attributes = [x+name for x in ['action', 'stand_profit', 'price', 'stand_price', 'profit']]
        for x in list(vars(self).keys()):

            if x in attributes:
                vars(self)[x] = np.array([vars(self)[x][y][-(convergence_period*3):] for y in range(len(vars(self)[x]))])
                vars(self)[x] = np.array([np.array(vars(self)[x][y])[pols[y]][-convergence_period:] for y in range(len(pols))])

    def attach_series(self, series_short, series_full, sample, subset):
        if series_short is not None:
            name = '_short'
            self._attach_series(series=series_short, sample=sample, type_history='short_environment', name=name, subset=subset)
        if series_full is not None:
            name = ''
            self._attach_series(series=series_full, sample=sample, type_history='full_environment', name=name, subset=subset)

    def attach_attributes(self, attributes, sample):
        if attributes is not None:
            for y in attributes:
                setattr(self, y, load_attributes(serie=y, location=self.location, agent=self.id, sample=sample))
                print(f'Attached: {y} for agent {self.id}')

####
class _AnalysisBody:
    '''
    By means of this class, the backend of the analysis is constructed
    '''
    def __init__(self, location, data_series_to_load_full, data_series_to_load_short, attributes_to_load,
                 agent_identifiers, environment, sample, BM_increment, reduce_full_environment):

        start = timeit.default_timer()
        print(f'Creating the object...')

        if not isinstance(environment, Environment):
            self.demand = Environment(*environment)
            self.demand.environment(BM_increment=BM_increment, tolerance=5)
        else:
            self.demand = environment

        self.agents = []
        self.agent_id = agent_identifiers
        self.location = location

        options = ['action', 'policy', 'stand_price', 'stand_profit', 'price', 'profit']
        if data_series_to_load_full is not None and data_series_to_load_short is not None:
            for x in data_series_to_load_full + data_series_to_load_short:
                if x not in options:
                    raise KeyError(f'Series must be of {options}')
        self.series = data_series_to_load_full
        self.series_short = data_series_to_load_short

        options = ['qmatrix', 'qmatrix2', 'traces', 'td_error']
        if attributes_to_load is not None:
            for x in attributes_to_load:
                if x not in options:
                    raise KeyError(f'Attributes must be of {options}')
        self.attributes = attributes_to_load

        # Create the agents and populate with data
        for x in self.agent_id:
            name = f'agent{x}'
            setattr(self, name, _HelperAgent(id=x, environment_object=self.demand, location=self.location))
            self.agents.append(name)

            vars(self)[name].attach_series(series_full=self.series, series_short=self.series_short, sample=sample,
                                           subset=reduce_full_environment)
            vars(self)[name].attach_attributes(attributes=self.attributes, sample=sample)

        stop = timeit.default_timer()
        time = stop - start
        print(f'Object creation succesful. Elapsed time: {time} seconds.\n'
              f'The percentage of working memory that is currently occupied is: {psutil.virtual_memory().percent} %.')

    def clean_agents(self, attributes=None):
        '''
        Function to clean agents in the middle of the analysis if too much memory is used
        '''
        for x in attributes:
            for y in self.agents:
                try:
                    delattr(vars(self)[y], x)
                except AttributeError:
                    print(f'misspecification {x}, {y}?')
                    continue

    # Functions used in analysis classes ------------------------------------------------------------
    @staticmethod
    def _mean_sd(obj):
        '''
        Mean of the mean over all sessions and the SD of that mean
        '''
        m = [np.mean(y) for y in obj]
        return round(np.mean(m), 4), round(np.std(m), 4), m

    @staticmethod
    def _sd_mean(obj):
        '''
        Mean SD over all sessions, and the SD of the SD's
        '''
        s = [np.std(y) for y in obj]
        return round(np.mean(s), 4), round(np.std(s), 4), s

    def _symmetric(self, variable='action', shortened_series=True):
        '''
        Checking which agents are symmetric
        '''
        name = '_short'
        if shortened_series is False:
            name = ''

        series = len(vars(self.agent0)[variable + name])

        check = [
            np.sum(abs(vars(self.agent0)[variable + name][x] -
                       vars(self.agent1)[variable + name][x]))
            for x in range(series)
        ]
        return check.count(0), check.count(0) / len(check), np.array(check)

    def _stability_series(self, variable='action', shortened_series=True):
        '''
        Stability of the agent, in the sense that it chooses the same action in the convergence period
        '''
        name = '_short'
        if shortened_series is False:
            name = ''

        series = len(vars(self.agent0)[variable+name])

        stab = [
            x for x in range(series)
            if
            len(np.unique(vars(self.agent0)[variable+name][x])) == 1
            and len(np.unique(vars(self.agent1)[variable+name][x])) == 1
        ]

        return len(stab), len(stab)/series, np.array(stab)

    @staticmethod
    def _wsr(obj1, obj2):
        '''
        # Wilcoxon signed rank test, for two lists of input values (f.e. mean profit)
        '''
        try:
            stat, p = wilcoxon(obj1, obj2)
            p = round(p, 4)
        except ValueError:
            stat = 'na'
            p = 'na'
        return [p, stat]

    @staticmethod
    def _runningmean(agent, serie, N, short=False):
        if short in [False, None, 0]:
            s = serie
        else:
            s = serie + "_short"

        obj = vars(agent)[s]
        cumsum = [np.cumsum(np.insert(obj[x], 0, 0)) for x in range(len(obj))]

        return [(cumsum[x][N:] - cumsum[x][:-N]) / float(N) for x in range(len(obj))]

    def _get_means(self, serie, agent, window, reduce_datapoints, shortened_history):
        '''
        Function to get the mean of the series
        '''
        obj = self._runningmean(agent=vars(self)[self.agents[agent]], serie=serie, N=window, short=shortened_history)

        if reduce_datapoints not in [None, False, 0]:
            obj = [obj[x][::reduce_datapoints] for x in range(len(obj))]

        return obj

    def save_setting_comparison_data(self, name_session, window, location, reduce_datapoints=False,
                                     serie=['td_error', 'stand_profit'], shortened_history=False):
        for x in range(len(self.agents)):

            loc = location+'/setting_comparison/'+self.agents[x]
            if not os.path.isdir(loc):
                os.makedirs(loc)
            os.chdir(loc)

            for y in serie:
                try:
                    if y is 'td_error':
                        shortened_history = False

                    obj = self._get_means(serie=y, agent=x, window=window, reduce_datapoints=reduce_datapoints,
                                          shortened_history=shortened_history)
                    obj = np.mean(obj, axis=0)
                    save_pickle(obj, f'{name_session}_agent{x}_{y}.pkl')
                except KeyError:
                    print(f'The indicated serie: {y} for agent {x} is not attached to the agent object')




