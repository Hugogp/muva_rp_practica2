import numpy as np
from matplotlib import pyplot as plt


def display_and_save_losses(losses: [float], save_path=None):
    # plt.style.use('_mpl-gallery')

    x = [None] * len(losses)
    y = [None] * len(losses)

    for i, loss in enumerate(losses):
        x[i] = i
        y[i] = loss

    # plot
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2.0)

    if save_path:
        plt.savefig(save_path)

    plt.show()
