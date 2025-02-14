from algcol.analysis.tables import _Table
from algcol.analysis.figures import _Figure

####
class Analysis(_Table, _Figure):
    '''
    By means of this class, each table in the experiment can be constructed dynamically
    '''
    def __init__(self, location, data_series_to_load_full, data_series_to_load_short, attributes_to_load,
                 agent_identifiers, environment, sample, BM_increment=None, reduce_full_environment=False):
        super().__init__(location, data_series_to_load_full, data_series_to_load_short, attributes_to_load,
                 agent_identifiers, environment, sample, BM_increment, reduce_full_environment)



