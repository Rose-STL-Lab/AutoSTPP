from typing import Union, List

from loguru import logger
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
from tqdm.auto import tqdm
from utils import relpath_under
from scipy.stats import multivariate_normal
from plotly.subplots import make_subplots
from rose_colormap.plotly import rose_vivid
from rose_colormap import rose_vivid as rose_vivid_matplotlib


def visualize_diff(outputs, targets, portion=1., fn=None):
    """
    Plot and compare the event trajectories in an index-aligned fashion
    assuming outputs is a time series with 3 features, (lat, lon, delta_time)
    or 1-step sliding windows of such a series with (lookahead) windows

    :param outputs: [batch, lookahead, 3] or [batch, 3]
    :param targets: [batch, lookahead, 3] or [batch, 3]
    :param portion: portion of outputs to be visualized, 0. ~ 1.
    :param fn: the saving filename
    :return: fig and axis
    """
    if len(targets.shape) == 2:
        outputs = np.expand_dims(outputs, 1)
        targets = np.expand_dims(targets, 1)

    outputs = outputs[:int(len(outputs) * portion)]
    targets = targets[:int(len(targets) * portion)]

    plt.figure(figsize=(14, 10), dpi=180)
    plt.subplot(2, 2, 1)

    n = outputs.shape[0]
    lookahead = outputs.shape[1]

    for i in range(lookahead):
        plt.plot(range(i, n), outputs[:n - i, i, 0], "-o", label=f"Predicted {i} step")
    plt.plot(targets[:, 0, 0], "-o", color="b", label="Actual")
    plt.ylabel('Latitude')
    plt.legend()

    plt.subplot(2, 2, 2)
    for i in range(lookahead):
        plt.plot(range(i, n), outputs[:n - i, i, 1], "-o", label=f"Predicted {i} step")
    plt.plot(targets[:, 0, 1], "-o", color="b", label="Actual")
    plt.ylabel('Longitude')
    plt.legend()

    plt.subplot(2, 2, 3)
    for i in range(lookahead):
        plt.plot(range(i, n), outputs[:n - i, i, 2], "-o", label=f"Predicted {i} step")
    plt.plot(targets[:, 0, 2], "-o", color="b", label="Actual")
    plt.ylabel('delta_t (hours)')
    plt.legend()

    if fn is not None:
        plt.savefig(fn)

    return plt.gcf(), plt.gca()


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration},
    }


def inverse_transform(x_range, y_range, t_range, scaler):
    """
    x_range, y_range, t_range: 1D array of any length
    """
    # Inverse transform the data
    temp = np.zeros((len(x_range), 3))
    temp[:, 0] = x_range
    x_range = scaler.inverse_transform(temp)[:, 0]

    temp = np.zeros((len(y_range), 3))
    temp[:, 1] = y_range
    y_range = scaler.inverse_transform(temp)[:, 1]

    temp = np.zeros((len(t_range), 3))
    temp[:, 2] = t_range
    t_range = scaler.inverse_transform(temp)[:, 2]

    return x_range, y_range, t_range


def plot_lambst_static(lambs, x_range, y_range, t_range, fps, scaler=None, cmin=None, cmax=None,
                       history=None, decay=0.3, base_size=300, cmap=rose_vivid_matplotlib, fn='result.mp4'):
    """
    The result could be saved as file with command `ani.save('file_name.mp4', writer='ffmpeg', fps=fps)`
                                        or command `ani.save('file_name.gif', writer='imagemagick', fps=fps)`

    :param lambs: list, len(lambs) = len(t_range), element: [len(x_range), len(y_range)]
    :param fps: # frame per sec
    :param fn: file_name
    """
    if type(t_range) is torch.Tensor:
        t_range = t_range.numpy()
    
    # Inverse transform the range to the actual scale
    if scaler is not None:
        x_range, y_range, t_range = inverse_transform(x_range, y_range, t_range, scaler)

    if cmin is None:
        cmin = 0
    if cmax == "outlier":
        cmax = np.max([np.max(lamb_st) for lamb_st in lambs])
    if cmax is None:
        cmax = np.max(lambs)
    logger.debug(f'Inferred cmax: {cmax}')
    cmid = cmin + (cmax - cmin) * 0.9

    grid_x, grid_y = np.meshgrid(x_range, y_range)

    frn = len(t_range)  # frame number of the animation

    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = fig.add_subplot(111, projection='3d', xlabel='x', ylabel='y', zlabel='λ', zlim=(cmin, cmax),
                         title='Spatio-temporal Conditional Intensity')
    ax.set_box_aspect([1, 1, 1])
    ax.title.set_position([.5, .95])
    text = ax.text(min(x_range), min(y_range), cmax, "t={:.2f}".format(t_range[0]), fontsize=10)
    plot = [ax.plot_surface(grid_x, grid_y, lambs[0], rstride=1, cstride=1, cmap=cmap)]

    if history is not None:
        his_s, his_t = history
        zs = np.ones_like(lambs[0]) * cmid  # add a platform for locations
        plat = ax.plot_surface(grid_x, grid_y, zs, rstride=1, cstride=1, color='white', alpha=0.2)
        points = ax.scatter3D([], [], [], color='black')  # add locations
        plot.append(plat)
        plot.append(points)

    pbar = tqdm(total=frn + 2)

    def update_plot(frame_number):
        t = t_range[frame_number]
        plot[0].remove()
        plot[0] = ax.plot_surface(grid_x, grid_y, lambs[frame_number], rstride=1, cstride=1, cmap=cmap)
        text.set_text('t={:.2f}'.format(t))

        if history is not None:
            mask = np.logical_and(his_t <= t, his_t >= t_range[0])
            locs = his_s[mask]
            times = his_t[mask]
            sizes = np.exp((times - t) * decay) * base_size
            zs = np.ones_like(sizes) * cmid
            plot[2].remove()
            plot[2] = ax.scatter3D(locs[:, 0], locs[:, 1], zs, c='black', s=sizes, marker='x')

        pbar.update()

    ani = animation.FuncAnimation(fig, update_plot, frn, interval=1000 / fps)
    ani.save(fn, writer='ffmpeg', fps=fps)
    return ani


def plot_lambst_interactive(lambs: Union[List, np.array], x_range, y_range, t_range, cmin=None, cmax=None,
                            scaler=None, heatmap=False, colorscale=rose_vivid, show=True, cauto=False,
                            master_title='Spatio-temporal Conditional Intensity', subplot_titles=None):
    """
    :param lambs:   3D Array-like sampled intensities of shape (t_range, x_range, y_range)
                    or 4D Array-like of shape (N, ...) to compare plots side-by-side
    :param x_range: 1D Array-like, specifying sampling x's locations
    :param y_range: 1D Array-like, specifying sampling y's locations
    :param t_range: 1D Array_like, specifying sampling t's locations
    :param cmin: lower bound of lambs axis, 0 if unspecified
    :param cmax: upper bound of lambs axis, max(lambs) if unspecified
    :param scaler: scipy.MinMaxScaler, used for scaling the intensities
    :param heatmap: whether draw the intensities as a heatmap instead of 3D surface plot
    :param colorscale: Color scales used for surface
    :param show: whether to show the figure
    :param master_title: the one title above all
    :param subplot_titles: 1D Array of N str, title of each side-by-side comparison plot
    """
    assert type(colorscale) == list and type(colorscale[0][1]) == str, "Unrecognized colorscale"

    if scaler is not None:  # Inverse transform the range to the actual scale
        x_range, y_range, t_range = inverse_transform(x_range, y_range, t_range, scaler)

    n_subplot = 1
    if type(lambs) == list:  # Convert lists to numpy array
        lambs = np.array(lambs)

    # Shape checks
    if len(lambs.shape) == 4:
        n_subplot = len(lambs)
        lambs_shape = lambs.shape[1:]
        lambs = lambs.transpose([1, 0, 2, 3])  # Put time before N
        assert subplot_titles is None or len(subplot_titles) == n_subplot
    else:
        assert len(lambs.shape) == 3
        lambs_shape = lambs.shape
    assert lambs_shape == (len(t_range), len(x_range), len(y_range))

    if cmin is None:
        cmin = min(0, np.min(lambs))
    if cmax is None:
        cmax = np.max(lambs)
    frames = []

    for i, lamb_st in enumerate(lambs):
        if n_subplot != 1:
            data = []
            for j, lamb_st_i in enumerate(lamb_st):
                if heatmap:
                    data.append(go.Heatmap(z=lamb_st_i, x=x_range, y=y_range, zmin=cmin, zmax=cmax,
                                           colorscale=colorscale, cauto=cauto))
                else:
                    data.append(go.Surface(z=lamb_st_i, x=x_range, y=y_range, cmin=cmin, cmax=cmax,
                                           colorscale=colorscale, cauto=cauto))
            frames.append(go.Frame(data=data, name="{:.2f}".format(t_range[i])))
        else:
            if heatmap:
                data = go.Heatmap(z=lamb_st, x=x_range, y=y_range, zmin=cmin, zmax=cmax, 
                                  colorscale=colorscale, cauto=cauto)
            else:
                data = go.Surface(z=lamb_st, x=x_range, y=y_range, cmin=cmin, cmax=cmax, 
                                  colorscale=colorscale, cauto=cauto)
            frames.append(go.Frame(data=data, name="{:.2f}".format(t_range[i])))

    if n_subplot != 1:
        if heatmap:
            specs = [[{"type": "xy"}] * n_subplot]
        else:
            specs = [[{"type": "scene"}] * n_subplot]
        fig = make_subplots(rows=1, cols=n_subplot, horizontal_spacing=0.05,
                            specs=specs, subplot_titles=subplot_titles)
        fig.frames = frames
    else:
        fig = go.Figure(frames=frames)

    # Add data to be displayed before animation starts
    if n_subplot != 1:
        for j, lamb_st_i in enumerate(lambs[0]):
            if heatmap:
                fig.add_trace(go.Heatmap(z=lamb_st_i, x=x_range, y=y_range, zmin=cmin, zmax=cmax,
                                         colorscale=colorscale, cauto=cauto), row=1, col=j + 1)
            else:
                fig.add_trace(go.Surface(z=lamb_st_i, x=x_range, y=y_range, cmin=cmin, cmax=cmax,
                                         colorscale=colorscale, cauto=cauto), row=1, col=j + 1)
    else:
        if heatmap:
            fig.add_trace(go.Heatmap(z=lambs[0], x=x_range, y=y_range, zmin=cmin, zmax=cmax, 
                                     colorscale=colorscale, cauto=cauto))
        else:
            fig.add_trace(go.Surface(z=lambs[0], x=x_range, y=y_range, cmin=cmin, cmax=cmax, 
                                     colorscale=colorscale, cauto=cauto))

    # Slider
    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": f.name,
                    "method": "animate",
                }
                for f in fig.frames
            ],
        }
    ]

    fig.update_scenes(  # Control zaxis for all subplots
        aspectmode='cube',
        zaxis_title='λ',
        zaxis=dict(range=[cmin, cmax], autorange=False),
        # camera=dict(
        #     eye=dict(x=1, y=-1.73205, z=1.15470)
        # )
    )

    # Layout
    fig.update_layout(
        title=master_title,
        width=500 * n_subplot + 180,
        height=700,
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(1)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders
    )
    if show:
        fig.show()
    return fig


class TrajectoryPlotter:
    """
    Interactive plot of the spatiotemporal trajectory in 3D
    """
    def __init__(self):
        self.data = []
        self.layout = go.Layout(
            width=1200,
            height=600,
            scene=dict(
                camera=dict(
                    up=dict(x=1, y=0., z=0),
                    eye=dict(x=0., y=2.5, z=0.)
                ),
                xaxis=dict(title="latitude"),
                yaxis=dict(title="longitude"),
                zaxis=dict(title="time"),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=3)
            ),
            showlegend=True,
        )

    def compare(self, outputs, targets, portion=1.):
        """
        Plot and compare the event trajectories over the absolute time axis and spatial domain,
        assuming outputs is a time series with 3 features, (lat, lon, delta_time)
        or 1-step sliding windows of such a series with (lookahead) windows

        :param outputs: [batch, lookahead, 3] or [batch, 3], (lat, lon, delta_t)
        :param targets: [batch, lookahead, 3] or [batch, 3], (lat, lon, delta_t)
        :param portion: portion of outputs to be visualized, 0. ~ 1.
        """
        if len(targets.shape) == 2:
            outputs = np.expand_dims(outputs, 1)
            targets = np.expand_dims(targets, 1)

        outputs = outputs[:int(len(outputs) * portion)]
        targets = targets[:int(len(targets) * portion)]

        target_t = np.cumsum(targets[:, 0, 2])
        self.add_trace(targets[:, 0, 0], targets[:, 0, 1], target_t, "actual")

        n = outputs.shape[0]
        lookahead = outputs.shape[1]
        for i in range(lookahead):
            # cumsum: time before prediction starts
            output_t = np.sum(targets[:i, 0, 2]) + np.cumsum(outputs[:n - i, i, 2])
            self.add_trace(outputs[:n - i, i, 0], outputs[:n - i, i, 1], output_t, f"Predicted {i} step")

    def add_trace(self, x, y, z, name=None, color=None) -> None:
        """
        Add a new 3D (lat, lon, time) trace to the plot
        :param x: [batch,]
        :param y: [batch,]
        :param z: [batch]
        :param name: label of the trace in the legend
        :param color: color of the trace
        """
        self.data.append(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            name=name,
            mode='lines+markers',
            marker=dict(
                size=4,
                symbol='circle',
                color=color,
            ),
            line=dict(
                width=3,
                color=color,
            ),
            opacity=.6
        ))

    def show(self) -> None:
        """
        Show the plot (as HTML)
        """
        fig = go.Figure(data=self.data, layout=self.layout)
        fig.show()

    def save(self, fn=None) -> None:
        """
        Save the plot (as HTML)
        :param fn: the saving filename, without HTML postfix
        """
        if fn is None:
            fn = "trajectory_plot"
        fig = go.Figure(data=self.data, layout=self.layout)
        full_fn = f"{relpath_under('figs', create_dir=True)}/{fn}.html"
        logger.info(f"Saving trajectories to {full_fn}...")
        fig.write_html(full_fn)


if __name__ == '__main__':
    def visualize_diff_0_step(option):
        X = np.arange(0, 1, 0.01)
        Y = np.vstack([np.sin(X), np.cos(X), np.diff(np.tan(X), prepend=-0.01)]).T
        noise = np.random.normal(loc=0.0, scale=[0.01, 0.001, 0.0001], size=(100, 3))
        Y_ = Y + noise
        if option == "visualize_diff":
            fig, ax = visualize_diff(Y_, Y, .2)
            fig.show()
        elif option == "TrajectoryPlotter":
            tp = TrajectoryPlotter()
            tp.compare(Y_, Y, .2)
            tp.save(fn='0step')
        else:
            raise NotImplementedError

    # visualize_diff_0_step(option="visualize_diff")
    # visualize_diff_0_step(option="TrajectoryPlotter")

    def visualize_diff_3_step(option):
        X = np.arange(0, 1, 0.01)
        Y = np.vstack([np.sin(X), np.cos(X), np.diff(np.tan(X), prepend=-0.01)]).T
        Y = np.stack([Y[:-3], Y[1:-2], Y[2:-1], Y[3:]], -1).transpose([0, 2, 1])
        noise = np.random.normal(loc=0.0, scale=[0.01, 0.001, 0.0001], size=(97, 4, 3))
        Y_ = Y + noise
        if option == "visualize_diff":
            fig, ax = visualize_diff(Y_, Y, .2)
            fig.show()
        elif option == "TrajectoryPlotter":
            tp = TrajectoryPlotter()
            tp.compare(Y_, Y, .2)
            tp.save(fn='3step')
        else:
            raise NotImplementedError

    # visualize_diff_3_step(option="visualize_diff")
    # visualize_diff_3_step(option="TrajectoryPlotter")

    def gaussian_mixture(coef_1, coef_2):
        """
        Calculate two-Gaussian mixture intensity based on the given coefficients,
        with spatial domain [0,1]x[0,1] and resolution of [0.01x0.01]

        :param coef_1: coef of first Gaussian
        :param coef_2: coef of second Gaussian
        :return: [101x101] intensity
        """
        from utils import arange
        mesh = arange(100, [[0., 1.], [0., 1.]], np)
        pdf1 = multivariate_normal.pdf(mesh, mean=np.array([0.2, 0.2]),
                                       cov=np.array([[0.05, 0], [0, 0.05]])).reshape(101, 101) * coef_1
        pdf2 = multivariate_normal.pdf(mesh, mean=np.array([0.5, 0.7]),
                                       cov=np.array([[0.02, 0], [0, 0.02]])).reshape(101, 101) * coef_2
        return pdf1 + pdf2

    def visualize_lambs_interactive(heatmap):
        x_range = np.arange(0.0, 1.01, 0.01)
        y_range = np.arange(0.0, 1.01, 0.01)
        t_range = np.arange(0.0, 40.1, 0.50)
        lambs = [gaussian_mixture(np.cos(t) + 1.0, np.sin(t) + 1.0) for t in t_range]
        lambs1 = [lamb_st + np.random.normal(loc=0.0, scale=0.05, size=lamb_st.shape) for lamb_st in lambs]
        lambs2 = [lamb_st + np.random.normal(loc=0.0, scale=0.2, size=lamb_st.shape) for lamb_st in lambs]

        fig1 = plot_lambst_interactive([lambs, lambs1, lambs2], x_range, y_range, t_range, cmin=None, cmax=None,
                                       scaler=None, heatmap=heatmap, colorscale=rose_vivid, show=False,
                                       subplot_titles=["Ground truth", "Predict by 1", "Predict by 2"])
        full_fn = f"{relpath_under('figs', create_dir=True)}/interactive.html"
        logger.info(f"Saving intensities to {full_fn}...")
        fig1.write_html(full_fn)

        fig2 = plot_lambst_interactive(lambs, x_range, y_range, t_range, cmin=None, cmax=None,
                                       scaler=None, heatmap=heatmap, colorscale=rose_vivid, show=False)
        full_fn = f"{relpath_under('figs', create_dir=True)}/interactive_single.html"
        logger.info(f"Saving intensities to {full_fn}...")
        fig2.write_html(full_fn)

    visualize_lambs_interactive(heatmap=False)
