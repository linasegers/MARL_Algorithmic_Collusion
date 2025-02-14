import os, random

from algcol.shared.load import load_pickle

####
def load_series(serie, location, agent, sample=False, type_history='full_environment', subset=False):
    series = ['return', 'action', 'policy', 'profit']

    if serie is 'profit':
        serie = 'return'

    if serie not in series:
        raise NotImplementedError(f'The requested series is not available, please make sure that "serie" is one of the following: \n'
                                  f'{series} \n')

    home = os.getcwd()
    location = location + f"/{type_history}"
    os.chdir(location)
    files = os.listdir('.')

    filenames = [file for file in files if f'agent{agent}' in file and file.endswith(f'{serie}.pkl')]

    # object
    if sample is not False:
        filenames = random.sample(filenames, sample)

    if subset:
        allfiles = [load_pickle(file)[-subset:] for file in filenames]
    else:
        allfiles = [load_pickle(file) for file in filenames]

    os.chdir(home)

    return allfiles
