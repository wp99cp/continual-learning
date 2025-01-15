import numpy as np
from matplotlib import pyplot as plt, colors as mcolors, cm as cm


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Truncate a colormap to only use a specific part of it.

    :param cmap: The original colormap.
    :param minval: Minimum value of the original colormap to use (0.0 to 1.0).
    :param maxval: Maximum value of the original colormap to use (0.0 to 1.0).
    :param n: Number of discrete colors to include in the truncated colormap.
    :return: Truncated colormap.
    """
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def get_epoch_evolution_graph(
    fig_title: str,
    x_label: str,
    y_label: str,
    z_label: str,
    data: dict,
    with_std: bool = False,
):
    """
    Prints a 2D graph with a third, discrete dimension represented by the color of the lines.

    :param fig_title: the title of the figure
    :param x_label: the label of the x-axis
    :param y_label: the label of the y-axis
    :param z_label: the label of the z-axis (color bar)
    :param data: a dictionary containing the data
    :param with_std: if True, the standard deviation of the data is plotted as well

    Example data dictionary:
    {
        "1": [[1, 2, 3], [4, 5, 6]],
        "2": [[7, 8, 9], [10, 11, 12]],
    }
    """
    fig, ax = plt.subplots()
    ax.set_title(fig_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Convert keys to numeric z-values
    z_values = list(map(float, data.keys()))
    z_labels = list(data.keys())

    norm = mcolors.Normalize(vmin=min(z_values), vmax=max(z_values))
    cmap = truncate_colormap(cm.GnBu, minval=0.5, maxval=1.0)

    # Plot each line with corresponding color
    tasks = None
    for z_key, _tasks in data.items():
        tasks = _tasks

        color = cmap(norm(float(z_key)))

        if not with_std:
            for task_values in tasks:
                epochs = np.arange(
                    len(task_values)
                )  # Create discrete x-values for epochs
                ax.plot(epochs, task_values, color=color)

        else:

            # Calculate mean and standard deviation
            task_values = np.array(tasks)
            mean_values = np.mean(task_values, axis=0)
            std_values = np.std(task_values, axis=0)

            epochs = np.arange(len(mean_values))  # Create discrete x-values for epochs
            ax.plot(
                epochs, mean_values, color=color, label=f"Task {z_key}"
            )  # Plot mean
            ax.fill_between(
                epochs,
                mean_values - std_values,
                mean_values + std_values,
                color=color,
                alpha=0.3,
            )  # Fill std area

    # Set discrete ticks for epochs
    max_epoch = max(len(task_values) for task_values in tasks)
    ax.set_xticks(np.arange(max_epoch))
    ax.set_xticklabels(np.arange(max_epoch))

    # Create a color bar with discrete ticks
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=z_values)
    cbar.set_label(z_label)
    cbar.set_ticks(z_values)
    cbar.set_ticklabels(z_labels)

    ax.set_ylim(0.0, 1.0)
    return fig
