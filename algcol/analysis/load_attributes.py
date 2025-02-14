import os, random

from algcol.shared.load import load_pickle

####
def load_attributes(serie, location, agent, sample=False):
    series = ['td_error', 'qmatrix', 'qmatrix2', 'traces']

    if serie not in series:
        raise KeyError(f'The requested series is not available, please make sure that "serie" is one of the following: \n'
                                  f'{series} \n')

    home = os.getcwd()
    loc = location + f"/agent{agent}"
    os.chdir(loc)
    files = os.listdir('.')

    filenames = [file for file in files if f"_agent{agent}_" in file and file.endswith(f'{serie}.pkl')]

    if sample is not False:
        filenames = random.sample(filenames, sample)

    allfiles = [load_pickle(file) for file in filenames]

    os.chdir(home)

    return allfiles


