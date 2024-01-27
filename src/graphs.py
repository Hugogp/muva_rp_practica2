from matplotlib import pyplot as plt


def display_and_save_losses(losses: [float], title: str, save_path: str | None = None):
    # matplotlib.use('Agg') # In case of error (was giving errors on Python 3.11)

    x = [None] * len(losses)
    y = [None] * len(losses)

    for i, loss in enumerate(losses):
        x[i] = i
        y[i] = loss

    # Plot the graph
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2.0)
    plt.title(title)

    # Save the graph in case the save_path is provided
    if save_path:
        plt.savefig(save_path)

    plt.show()
