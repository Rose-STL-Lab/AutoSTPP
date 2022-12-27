import matplotlib.pyplot as plt
import numpy as np


def plot_temporal(lambs, events, colors=None, width=10, dpi=120):
    """
    Plot λ(t) vs. time at a specified location

    :param lambs: a name->data dictionary with two mandatory entries,
                  lamb['x'] and lamb['GT'] with shape (num_sample,).
                  other entries should also have shape (num_sample,) and are considered as predictions.
    :param events: (seq_len, ) specifying the events corresponding to the intensities
    :param width: width of the figure
    :param colors: dictionary specifying lambs colors
    :param dpi: dpi of the figure
    :return: figure
    """
    t_start, t_end = lambs['x'][0], lambs['x'][-1]
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(width, width / 4),
                                   gridspec_kw={'height_ratios': [10, 1]}, dpi=dpi)

    # Plot the intensity
    ax1.plot(lambs['x'], lambs['GT'], 'grey', label=f'λ(t) ground truth', linewidth=5, alpha=0.5)
    for key in lambs:
        if key == 'x' or key == 'GT':
            continue
        else:
            if key in colors:
                ax1.plot(lambs['x'], lambs[key], label=key, color=colors[key])
            else:
                ax1.plot(lambs['x'], lambs[key], label=key)
    ax1.set_xlim([t_start, t_end])
    ax1.get_xaxis().set_visible(False)

    plt.subplots_adjust(wspace=0., hspace=0.)

    # Plot the events
    x = np.zeros(len(events) + 2)
    x[1:-1] = events
    x[0] = t_start - 1
    x[-1] = t_end + 1
    y = np.ones_like(x)
    y[0] = 0
    y[-1] = 1.8

    marker_line, stem_line, baseline = ax1.stem(x, np.ones_like(y) * np.mean(lambs['GT']), label=f'Events',
                                                linefmt='k1-', basefmt=' ')
    plt.setp(stem_line, linewidth=1.25)
    plt.setp(marker_line, markersize=7, color='yellow', markeredgecolor='black')
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
    marker_line.set_visible(False)
    stem_line.set_visible(False)

    marker_line, stem_line, baseline = ax2.stem(x, y, label=f'Events', linefmt='k1-', basefmt=' ')
    plt.setp(stem_line, linewidth=1.25)
    plt.setp(marker_line, markersize=7, color='yellow', markeredgecolor='black')
    ax2.set_xlim([t_start, t_end])
    ax2.invert_yaxis()
    ax2.get_yaxis().set_visible(False)

    return fig
