from typing import Dict, List

import pytest
from conftest import relpath, update_params, get_params

import numpy as np
import torch
import plotly.graph_objects as go

from autoint_mlp import model
from integration.autoint import BaselineSequential, MixSequential
from copy import deepcopy

from loguru import logger


warmup = False


@pytest.fixture(scope="class", autouse=True)
def log(model, benchmark):
    from loguru import logger
    logger.info(pytest.fn_params)


def adapt_model(model: MixSequential, model_type: str) -> torch.nn.Module:
    """
    Non-fixture function for creating model of a given type using
    another model's parameters
    :param model: input model that provide the parameters
    :param model_type: the new model type, among (baseline, model, mix[n]) <br>
        - baseline: PyTorch recursive implementation <br>
        - model: our method <br>
        - mix: Differentiate higher order by PyTorch, but use our as recursive base cases.
               n is the level where it goes to base cases.
    :return: the new model with the same set of parameters
    """
    if model_type == "model":
        return model.layers
    elif model_type == "baseline":
        return BaselineSequential(deepcopy(model.layers))
    elif model_type.startswith("mix"):
        n = int(model_type[3:])
        model.layers.reset()
        model = deepcopy(model)
        model.layers.threshold = n
        return model
    else:
        raise NotImplementedError


@pytest.fixture(
    scope="class",
    params=get_params('benchmark')
)
def benchmark(model, device, request):
    import torch
    import datetime
    import pickle
    import os
    from loguru import logger
    from utils import AverageMeter

    update_params('benchmark', request)
    fn = relpath('data/external') + '.pkl'

    global warmup
    warmup = request.param['derivative'] == 'warmup'

    # Try to use the previous benchmark result
    if not request.param['rerun'] and not warmup:
        if os.path.exists(fn):
            with open(fn, 'rb') as handle:
                time_elapsed = pickle.load(handle)
                logger.info('Previous benchmark found.')
                return time_elapsed
        else:
            logger.info('Previous benchmark not found. Rerunning...')

    time_elapsed = {}

    for model_type in pytest.config[__file__]['model_types']:
        model_ = adapt_model(model, model_type)
        time_elapsed[model_type] = []
        for order in pytest.config[__file__]['orders']:
            time_meter = AverageMeter()
            for _ in range(pytest.config[__file__]['n_repeat']):
                optimizer = torch.optim.Adam(model_.parameters(), lr=5e-3)
                x = torch.rand(64, pytest.config[__file__]['model']['inp_dim']).to(device)

                optimizer.zero_grad()
                a = datetime.datetime.now()

                if request.param['derivative'] == 'univariate' or warmup:
                    loss = model_.dnforward(x, [0] * order)
                elif request.param['derivative'] == 'mixed':
                    loss = model_.dnforward(x, list(range(order)))
                else:  # half
                    loss = model_.dnforward(x, list(np.random.randint(0, order // 2 + 1, order)))

                if request.param['computation'] == 'backward':
                    loss = sum(loss)
                    loss.backward()
                    optimizer.step()

                b = datetime.datetime.now()
                time_meter.update((b - a).total_seconds())
            time_elapsed[model_type].append(time_meter.val)

    if not warmup:
        with open(fn, 'wb') as handle:
            pickle.dump(time_elapsed, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return time_elapsed


class TestClass:

    def test_line_plot(self, benchmark: Dict[str, List]) -> None:
        """
        Compare the baseline and our model's time usage using line plot without transformation

        :param benchmark: the benchmark results in model_name -> avg_time per order format
        """
        global warmup
        if warmup:
            logger.info("Warming up...")
            return

        fig = go.Figure()  # Pairwise comparison fig
        full_fig = go.Figure()  # Full comparison fig

        # Add baseline trace
        x = pytest.config[__file__]['orders']
        y = benchmark['baseline']
        baseline_trace = go.Scatter(x=x, y=y, mode='lines', name='baseline')
        fig.add_trace(baseline_trace)
        full_fig.add_trace(baseline_trace)

        der = pytest.fn_params['benchmark']['derivative']
        com = pytest.fn_params['benchmark']['computation']

        fig.update_layout(
            width=550,
            height=450,
            title=f"Comparing baseline and efficient impl <br> of AutoInt, {der} derivative, {com}",
            xaxis=dict(
                tickmode='linear',
                tick0=1,
                dtick=1
            ),
            xaxis_title="order of derivative",
            yaxis_title="forward time (seconds)",
            font=dict(
                size=16
            )
        )

        full_fig.update_layout(
            width=550,
            height=450,
            title=f"Comparing baseline and efficient impl <br> of AutoInt, {der} derivative, {com}",
            xaxis=dict(
                tickmode='linear',
                tick0=1,
                dtick=1
            ),
            xaxis_title="order of derivative",
            yaxis_title="forward time (seconds)",
            font=dict(
                size=16
            )
        )

        for model_type in benchmark:
            if model_type == 'baseline':
                continue
            y = benchmark[model_type]
            fig_ = deepcopy(fig)
            fig_.add_trace(go.Scatter(x=x, y=y, mode='lines', name='ours'))
            full_fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=model_type))

            fig_.write_html(f"{relpath('figs', True)}/{model_type}.html")

        full_fig.write_html(f"{relpath('figs', True)}/full.html")

    def test_log_line_plot(self, benchmark: Dict[str, List]) -> None:
        """
        Compare the baseline and our model's time usage using line plot with log transformation

        :param benchmark: the benchmark results in model_name -> avg_time per order format
        """
        global warmup
        if warmup:
            logger.info("Warming up...")
            return

        fig = go.Figure()  # Pairwise comparison fig
        full_fig = go.Figure()  # Full comparison fig

        # Add baseline trace
        x = pytest.config[__file__]['orders']
        y = benchmark['baseline']
        baseline_trace = go.Scatter(x=x, y=np.log(y), mode='lines', name='baseline')
        fig.add_trace(baseline_trace)
        full_fig.add_trace(baseline_trace)

        der = pytest.fn_params['benchmark']['derivative']
        com = pytest.fn_params['benchmark']['computation']

        fig.update_layout(
            width=550,
            height=450,
            title=f"Comparing baseline and efficient impl <br> of AutoInt, {der} derivative, {com}",
            xaxis=dict(
                tickmode='linear',
                tick0=1,
                dtick=1
            ),
            xaxis_title="order of derivative",
            yaxis_title="log forward time (seconds)",
            font=dict(
                size=16
            )
        )

        full_fig.update_layout(
            width=550,
            height=450,
            title=f"Comparing baseline and efficient impl <br> of AutoInt, {der} derivative, {com}",
            xaxis=dict(
                tickmode='linear',
                tick0=1,
                dtick=1
            ),
            xaxis_title="order of derivative",
            yaxis_title="log forward time (seconds)",
            font=dict(
                size=16
            )
        )

        for model_type in benchmark:
            if model_type == 'baseline':
                continue
            y = benchmark[model_type]
            fig_ = deepcopy(fig)
            fig_.add_trace(go.Scatter(x=x, y=np.log(y), mode='lines', name='ours'))
            full_fig.add_trace(go.Scatter(x=x, y=np.log(y), mode='lines', name=model_type))

            fig_.write_html(f"{relpath('figs', True)}/{model_type}_log.html")

        full_fig.write_html(f"{relpath('figs', True)}/full_log.html")

    def test_ratio_plot(self, benchmark: Dict[str, List]) -> None:
        """
        Compare the baseline and our model's time usage using ratio bar plot

        :param benchmark: the benchmark results in model_name -> avg_time per order format
        """
        global warmup
        if warmup:
            logger.info("Warming up...")
            return

        fig = go.Figure()  # Pairwise comparison fig
        full_fig = go.Figure()  # Full comparison fig

        # Add baseline trace
        x = pytest.config[__file__]['orders']
        y_baseline = benchmark['baseline']
        baseline_trace = go.Bar(x=x, y=np.ones_like(x), name='baseline', opacity=0.9)
        fig.add_trace(baseline_trace)
        full_fig.add_trace(baseline_trace)

        der = pytest.fn_params['benchmark']['derivative']
        com = pytest.fn_params['benchmark']['computation']

        fig.update_layout(
            barmode='overlay',
            width=550,
            height=450,
            title=f"Comparing baseline and efficient impl <br> of AutoInt, {der} derivative, {com}",
            xaxis=dict(
                tickmode='linear',
                tick0=1,
                dtick=1
            ),
            xaxis_title="order of derivative",
            yaxis_title="relative time",
            font=dict(
                size=16
            )
        )

        full_fig.update_layout(
            barmode='overlay',
            width=550,
            height=450,
            title=f"Comparing baseline and efficient impl <br> of AutoInt, {der} derivative, {com}",
            xaxis=dict(
                tickmode='linear',
                tick0=1,
                dtick=1
            ),
            xaxis_title="order of derivative",
            yaxis_title="relative time",
            font=dict(
                size=16
            )
        )

        for model_type in benchmark:
            if model_type == 'baseline':
                continue
            y = benchmark[model_type]
            fig_ = deepcopy(fig)
            fig_.add_trace(go.Bar(x=x, y=np.array(y) / np.array(y_baseline), name='ours', opacity=0.9))
            full_fig.add_trace(go.Bar(x=x, y=np.array(y) / np.array(y_baseline), name=model_type, opacity=0.9))

            fig_.write_html(f"{relpath('figs', True)}/{model_type}_ratio.html")

        full_fig.write_html(f"{relpath('figs', True)}/full_ratio.html")
