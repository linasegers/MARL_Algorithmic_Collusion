import pandas as pd
import os

from algcol.analysis.analysisbody import _AnalysisBody

####
class _Table(_AnalysisBody):
    '''
    By means of this class, each table in the experiment can be constructed dynamically
    '''
    def __init__(self, location, data_series_to_load_full, data_series_to_load_short, attributes_to_load,
                 agent_identifiers, environment, sample, BM_increment, reduce_full_environment):
        super().__init__(location, data_series_to_load_full, data_series_to_load_short, attributes_to_load,
                 agent_identifiers, environment, sample, BM_increment, reduce_full_environment)

    def _stats_per_agent(self, agent_id, obj, shortened_history=True):
        '''
        Returns a list with the mean and sd based on the order of the objects list
        '''
        if shortened_history:
            obj = obj + "_short"
            m, s, a = self._mean_sd(vars(vars(self)[self.agents[agent_id]])[obj])
        elif isinstance(shortened_history, (int, float)):
            m, s, a = self._mean_sd(vars(vars(self)[self.agents[agent_id]])[obj][-shortened_history:])
        else:
            m, s, a = self._mean_sd(vars(vars(self)[self.agents[agent_id]])[obj])

        return m, s, a

    def standardtable(self, parts=['stand_price', 'stand_profit'], test_statistic=True, rownames=None):
        '''
        Get the data of the standard table that is used.
        '''
        options = ['stand_price', 'stand_profit']
        for x in parts:
            if x not in options:
                raise Exception(f'The expected input for the arguments "parts", is a list containing the relevant \n'
                                f'metrics of the table that need to be produced. Options for \n'
                                f'parts are: \n'
                                f'{options}')

        s = False
        if hasattr(vars(self)['agent0'], 'action_short'):
            s = True
            n = len(self.agent0.action_short)
        else:
            n = len(self.agent0.action)

        _, stability, _ = self._stability_series(variable='action', shortened_series=s)
        _, symmetry, _ = self._symmetric(variable='action', shortened_series=s)

        results = [n, stability, symmetry]
        colnames = ['n', 'stability', 'symmetry']

        for y in parts:

            m0, s0, a0 = self._stats_per_agent(agent_id=0, obj=y, shortened_history=s)
            results += [m0, s0]
            colnames += [f'm_{y}_A0', f'sd_{y}_A0']

            m1, s1, a1 = self._stats_per_agent(agent_id=1, obj=y, shortened_history=s)
            results += [m1, s1]
            colnames += [f'm_{y}_A1', f'sd_{y}_A1']

            if test_statistic:
                results += self._wsr(a0, a1)
                colnames += [f'p_{y}', f'tstat_{y}']

        if rownames is not None:
            if rownames is 'setting':
                rownames = os.path.basename(self.location)
            results = [rownames] + results
            colnames = ['Setting'] + colnames

        results = pd.DataFrame(results).transpose()
        results.columns = colnames

        return results

    def show_table(self, table='standardtable', *args, **kwargs):
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        if table is 'standardtable':
            results = self.standardtable(*args, **kwargs)
            print(results)
        else:
            raise NotImplementedError('Other tables are still under construction')
        pd.set_option("display.max_rows", 5, "display.max_columns", 6)


