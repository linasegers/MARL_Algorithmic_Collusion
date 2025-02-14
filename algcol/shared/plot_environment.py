import numpy as np
import matplotlib.pyplot as plt
import itertools

####
def plot_environment(column, show=True, agent_number='both', title='', environment=None, environment_parameters=None,
                     check_bm=False):
    # Throw error is wrong option is chosen
    options = ["profit", "standardized_profit", "demand"]
    if column not in options:
        raise Exception(f'Column needs to be of {options}')

    # quantities, prices, profit, standardized_profit, standardized price
    locations = {'profit': 2, 'standardized_profit': 3, 'demand': 1}

    # Get the full environment
    if environment is None:
        from algcol.experiment.environment import Environment
        environment = Environment(*environment_parameters)

    if 'full_environment' not in vars(environment):
        environment.environment(BM_increment=check_bm)

    # Function to plot the confusion matrix given a list of class names
    def plot_confusion(conf_mat, class_names, title, num):
        # Turn list into matrix
        if type(conf_mat) is not np.ndarray:
            conf_mat = np.array(conf_mat)

        # Flip the matrix over
        conf_mat = np.flip(conf_mat, axis=0)

        plt.clf()

        # place labels at the top
        plt.gca().xaxis.tick_bottom()
        plt.gca().xaxis.set_label_position('bottom')

        # plot the matrix per se
        plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.viridis_r)

        # write the number of predictions in each bucket
        for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
            # if background is dark, use a white number, and vice-versa
            plt.text(j, i, f'{round(conf_mat[i, j], 2)}', fontsize=8,
                     horizontalalignment="center",
                     color="white")

        tick_marks = np.arange(len(conf_mat))
        plt.xticks(tick_marks, class_names, rotation=45, size=8)
        class_names.reverse()
        plt.yticks(tick_marks, class_names, rotation=45, size=8)
        class_names.reverse()
        plt.tight_layout()
        if num == 2:
            plt.ylabel('Action opponent', size=10)
            plt.xlabel('Action', size=10)
        elif num == 1:
            plt.ylabel('Action', size=10)
            plt.xlabel('Action opponent', size=10)
        else:
            plt.ylabel('Action agent 1', size=10)
            plt.xlabel('Action agent 2', size=10)
        plt.title(f'{title}')

    # Get the actions to create the list with labels
    act = environment.actions
    act = [f'A{x+1}: {round(act[x],2)}' for x in range(len(act))]

    # Now get the part of the environment that you want to plot
    df = environment.full_environment
    cols = [df[x][y][locations[column]] for x in range(len(df)) for y in range(len(df[0]))]

    col1 = np.array_split([cols[x][0] for x in range(len(cols))], np.sqrt(len(cols)))
    col2 = np.array_split([cols[x][1] for x in range(len(cols))], np.sqrt(len(cols)))

    # Determine which of the matrices you want to plot
    if agent_number == 1:
        col = col1
    elif agent_number == 2:
        col = col2
    else:
        col = np.array(col1) + np.array(col2)

    # Plot the object:
    plot_confusion(col, act, title=title, num=agent_number)
    if show is True:
        plt.show()
    else:
        plt.savefig(show, bbox_inches='tight')