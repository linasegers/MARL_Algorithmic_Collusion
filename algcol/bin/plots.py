def plot(self, method=None, interval=None, plot_type=None):
    '''
    :param method: Actions, profit, quantities or standardized profit
    :param kwargs: If 'interval' is specified calculates a rolling window of the specified method
    '''
    # Throw error when the wrong type of data is asked
    options = ["profit", "standardized_profit", "standardized_price", "quantities", "prices"]
    if method not in options:
        raise Exception(f'Method needs to be of {options}')

    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    # Get data
    idx = [self.history[x][0] for x in range(len(self.history))]
    lines = self.env._translator(idx, method, list(range(self.agents)))

    # Plot a rolling window variant
    if 'interval' is not None:
        lines = [running_mean(x, interval) for x in lines]

    # Plot the figure
    plt.figure()
    plt.title("Plot progress")
    plt.xlabel("Period")

    if interval is not None:
        # Change plot label based on the method of choice
        plt.ylabel(f'Rolling window {method} (interval: {interval}')
        w = 0.01
    else:
        plt.ylabel(f'{method}')
        w = 0.5

    color = iter(plt.cm.RdGy(np.linspace(0, 1, self.agents)))

    for y in range(self.agents):
        c = next(color)
        if plot_type is 'scatter':
            plt.scatter(list(range(len(lines[y]))), lines[y], color=c, alpha=0.1, s=1, label=f"Agent {y}")
        else:
            plt.plot(lines[y], marker='', color=c, linewidth=w, label=f"Agent {y}")

    plt.show()


def plot_performance(self, method, reduced=False):
    '''
    :param method:                  Whether you want to display the td_error or the qmatrix
    :param reduced (in kwargs):     If the opponent is a fixed strategy agent and you want to display the reduced (relevant)
                                    Q-matrix, use the argument 'reduced='yes'
    '''
    methods = ['qmatrix', 'td_error']

    # Raise exception in the case of a wrong method
    if method not in methods:
        raise Exception(f'This is not a valid option, choose method from {methods}')

    # To plot the Q-matrix
    if method == "qmatrix":
        t = pd.DataFrame(self.qmatrix)
        if reduced is True:
            t = t.loc[(t != 0).any(axis=1)]
        sns.heatmap(t, cmap="PuRd", annot=False)
        plt.title(f'Q-matrix')
        plt.xlabel('Action')
        plt.ylabel('State')
        plt.show()
    # To plot the TD-error
    else:
        pd.DataFrame(self.td_error).plot.line(linewidth=0.1, color='gray', legend=None)
        plt.title('TD-error over time')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.show()


def plot_deviation(self, title='', save=False, startpoint=-2, endpoint=15):
    # Point in history where the agent deviated
    hist = [self.history[x][2] for x in range(len(self.history))]
    search = self.agents + 1
    deviation = 0
    for sublist in hist:
        if np.sum(sublist) >= search:
            player = sublist.index(2)
            break
        else:
            deviation += 1

    actions = []
    for i in range(self.agents):
        a = [self.history[x][3][i] for x in range(len(self.history))]
        actions.append(a)

    # Convert all actions to prices
    s = []
    for y in range(self.agents):
        prices = vars(self)[self.agent_names[y]].actions
        a = actions[y][(deviation + startpoint):(deviation + endpoint)]
        series = [prices[a[x]] for x in range(len(a))]
        s.append(series)

    fig = plt.figure(figsize=(7, 3))
    plt.title(f"{title}")
    plt.xlabel("Period")
    plt.ylabel('Price')

    plt.hlines(self.env.pm, xmin=0, xmax=len(s[0]), linestyles='dotted', label='Monopoly price')
    plt.hlines(self.env.pb, xmin=0, xmax=len(s[0]), linestyles='dashdot', label='Bertrand price')
    plt.vlines(-startpoint, ymin=self.env.pb, ymax=self.env.pm, linewidth=0.15, linestyles='dotted')
    plt.vlines(-startpoint + 1, ymin=self.env.pb, ymax=self.env.pm, linewidth=0.15, linestyles='dotted')

    color = plt.cm.rainbow(np.linspace(0, 1, self.agents))
    for i, c in zip(range(self.agents), color):
        if i == player:
            plt.plot(s[i], c=c, label=f'Deviating {self.agent_names[i]}')
        else:
            plt.plot(s[i], c=c, label=f'{self.agent_names[i]}')

    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="lower center")
    plt.title(f'{title}')
    plt.tight_layout()

    # Save or display
    if save is False:
        plt.show()
    else:
        fig.savefig(save)