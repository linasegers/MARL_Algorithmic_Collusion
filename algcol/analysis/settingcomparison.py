import os

from algcol.shared.load import load_pickle
import matplotlib.pyplot as plt
import numpy as np

###
def plot_setting_comparison(location, serie, save=False):
    os.chdir(location)
    files = os.listdir('.')
    files = [x for x in files if x.endswith(f'{serie}.pkl')]
    name = [x.split('_')[0] for x in files]

    f = [load_pickle(x) for x in files]

    fig = plt.figure(figsize=(6, 5))
    color = iter(plt.cm.RdGy(np.linspace(0, 1, len(f))))
    for x in range(len(f)):
        c = next(color)
        plt.plot(f[x], color=c, label=name[x])
    plt.xlabel(f'Period')
    plt.ylabel('Level')
    plt.title(f'Comparison of the different settings')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center")
    plt.tight_layout()

    if save is False:
        plt.show()
    else:
        fig.savefig(save)
        plt.close()