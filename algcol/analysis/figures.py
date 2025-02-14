import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from algcol.analysis.analysisbody import _AnalysisBody

####
class _Figure(_AnalysisBody):
    '''
    By means of this class, each table in the experiment can be constructed dynamically

    ADD THE PLOT FOR DEVIATION/PUNISHMENT STRATEGIES
    ADD RMS PLOT
    '''
    def __init__(self, location, data_series_to_load_full, data_series_to_load_short, attributes_to_load,
                 agent_identifiers, environment, sample, BM_increment, reduce_full_environment):
        super().__init__(location, data_series_to_load_full, data_series_to_load_short, attributes_to_load,
                 agent_identifiers, environment, sample, BM_increment, reduce_full_environment)

    def _plot_data(self, serie, agent, window, datapointreduction, shortened_history):
        obj = self._get_means(serie=serie, agent=agent, window=window,
                              reduce_datapoints=datapointreduction, shortened_history=shortened_history)

        # Get the mean of the object
        average = np.mean(obj, axis=0)
        # Time:
        t = list(range(len(average)))

        # Get min of the object
        minimum = np.min(obj, axis=0)
        # Get the maximum of the object
        maximum = np.max(obj, axis=0)

        # Make an area plot:
        long_t = t * len(obj)
        flat = np.concatenate(obj)

        return average, t, minimum, maximum, long_t, flat

    def contour_plot(self, serie, agent, window=100, title='Object',
                        save=False, datapoint_reduction=None, figsize=(6, 5), bins=(1000, 50), lims=None, shortened_history=False):

        average, t, minimum, maximum, long_t, flat = self._plot_data(serie=serie, agent=agent,
                                                                     window=window, datapointreduction=datapoint_reduction,
                                                                     shortened_history=shortened_history)
        heatmap, xedges, yedges = np.histogram2d(long_t, flat, bins=bins)

        # Plot the figure
        fig, ax = plt.subplots(figsize=figsize)
        cf = ax.contour(xedges[1:], yedges[1:], heatmap.T, 20, norm = colors.PowerNorm(gamma=0.5), cmap='Reds')
        ax.plot(t, average, color='black')

        # Set limits dynamically
        if lims is None:
            ax.set_ylim(bottom=-0.4, top=1.4)
        else:
            ax.set_ylim(bottom=lims[0], top=lims[1])
        ax.set_title(f'Moving average of {title}')
        ax.set_xlabel(f'Period, t x {datapoint_reduction}')
        ax.set_ylabel(f'Level of {title}')

        if save is False:
            plt.show()
        else:
            fig.savefig(save)
            plt.close()

    def compare_agent_learning(self, serie, window=1000, datapointreduction=10, save=False, short=False):
        # Convert to numpy array for the rest of the operations
        means = []
        for x in range(len(self.agents)):
            if serie is 'td_error': short = False
            m = self._get_means(serie=serie, agent=x, window=window, reduce_datapoints=datapointreduction,
                                shortened_history=short)
            m = np.mean(np.array(m), axis=0)
            means.append(m)

        fig = plt.figure(figsize=(6, 5))
        color = iter(plt.cm.viridis(np.linspace(0, 1, len(means))))
        for x in range(len(self.agents)):
            c = next(color)
            plt.plot(means[x], color=c, label=f'agent {[x]}')
        plt.xlabel(f'Period (x {window})')
        plt.ylabel('Level')
        plt.title(f'Moving average of the {serie}')
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center")
        plt.tight_layout()
        if save is False:
            plt.show()
        else:
            fig.savefig(save)
            plt.close()

    def development_of_exploration(self, agent=[0, 1], shortened_series=True, bins=100):
        if agent not in [0, 1, [0, 1]]:
            raise ValueError(f'This option is not available, choose: 0, 1 or [0,1] to choose both')

        name = 'policy'
        if shortened_series:
            name = 'policy_short'

        if not isinstance(agent, list):
            agent = [agent]

        hists = []
        for x in agent:
            n = len(vars(vars(self)[f'agent{x}'])[name])
            pol = np.where(vars(vars(self)[f'agent{x}'])[name] == 1, np.nan, 1)
            pol = np.nansum(pol, axis=0)
            hists.append([np.sum(pol[x:x+bins]) for x in list(range(len(pol)))[::bins]])

        if len(agent) > 1:
            hist = np.array(hists[0]) + np.array(hists[1])
        else:
            hist = hists[0]

        period = list(range(len(hist)))

        z = np.polyfit(period, hist, 1)
        p = np.poly1d(z)

        fig, ax = plt.subplots()
        ax.bar(period, height=hist, color='gray')
        ax.set_ylabel(f'Count of exploring moves per bin for {n} experiments', color='gray')
        ax.set_xlabel(f'Total period: {len(pol)}. Divided in {len(hist)} bins')
        ax.plot(period, p(period), "r--")
        ax2 = ax.twinx()
        ax2.scatter(period, np.array(hist)/250/100, color='blue', s=4)
        ax2.set_ylabel('Occurrence of exploration per experiment for a single period', color='blue')
        fig.suptitle(f'Evolution of exploration for agent {agent}')

        plt.tight_layout()
        plt.show()

    def plot_environment(self, column, agent_number='both', title=''):
        from algcol.shared.plot_environment import plot_environment
        plot_environment(column, agent_number=agent_number, title=title, environment=self.demand)

    def plot_series(self, agent, serie, window=100, datapointreduction=10, save=False, short=False, alpha=0.05):
        # Convert to numpy array for the rest of the operations
        if serie is 'td_error': short = False
        m = self._get_means(serie=serie, agent=agent, window=window, reduce_datapoints=datapointreduction,
                            shortened_history=short)
        overall = np.mean(np.array(m), axis=0)

        fig = plt.figure(figsize=(6, 5))
        for x in range(len(m)):
            plt.plot(m[x], color='gray', alpha=alpha)
        plt.plot(overall, color='black')
        plt.xlabel(f'Period (x {window})')
        plt.ylabel('Level')
        plt.title(f'Moving average of the {serie}')
        plt.tight_layout()
        if save is False:
            plt.show()
        else:
            fig.savefig(save)
            plt.close()





