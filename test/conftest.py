from typing import List, Dict

import pytest
from typeguard import typechecked
from copy import deepcopy
from utils import relpath_under, serialize_config
import inspect
import os
import wandb


@pytest.fixture(scope='module', autouse=True)
def device():
    from utils import get_device

    return get_device(free=False, min_ram=8000)


def get_params(key: str, caller_fn: str = ''):
    from utils import load_config, dict_to_list
    from loguru import logger
    if caller_fn == '':  # No filename given
        caller_fn = inspect.stack()[1].filename  # Absolute path
        
    assert '.py' in caller_fn
    fn = caller_fn[:caller_fn.rindex('.py')]
    fn = fn[fn.rindex('/') + 1:]
    logger.debug(f'Loading {key} config from configs/{fn}.yaml')
    configs: Dict[str, Dict[str, List]] = load_config(fn)
    pytest.config[caller_fn] = deepcopy(configs)

    for fixture_name in pytest.config[caller_fn]:
        if type(configs[fixture_name]) is not dict:  # Non-fixture parameters
            del configs[fixture_name]
            continue
        for config_name in configs[fixture_name]:
            if type(configs[fixture_name][config_name]) is list:
                if fixture_name not in pytest.fn_params:
                    pytest.fn_params[fixture_name] = {}
                pytest.fn_params[fixture_name][config_name] = ''
            else:
                configs[fixture_name][config_name] = [configs[fixture_name][config_name]]   # Make it a list

    params = {fixture_name: dict_to_list(configs[fixture_name]) for fixture_name in configs}
    try:
        return params[key]
    except KeyError:
        logger.error(f'{fn} does not have config for {key}')
        return []


def pytest_configure():
    """
    Load the test config for current test, as a double dictionary mapping 
    fixture name -> config name -> list of config options
    All combinations of listed config option will be tested.
    """
    from utils import default_fmt
    from loguru import logger
    import sys

    # NOTICE: test config should always be dictionary of lists
    fn = sys.argv[-1]  # Current test file name
    if '.py' in fn:
        fn = fn[:fn.rindex('.py')]
        # Add a file logger
        logger.add(f'{relpath_under("logs", fn)}.log', format=default_fmt, level="DEBUG")
        
    if '/' in fn:
        fn = fn[fn.rindex('/') + 1:]
    pytest.fn = fn
    pytest.fn_params = {}  # Parameters that need to be in the filename
    pytest.config = {}  # Configs for different test files
    pytest.result = {}


def put_result(name: str, value) -> None:
    """
    Store the result in the dictionary with the given config
    :param name: key of the result
    :param value: value of the result, can be any type
    """
    if str(pytest.fn_params) not in pytest.result:
        pytest.result[str(pytest.fn_params)] = {}
    pytest.result[str(pytest.fn_params)][name] = value


def pytest_unconfigure():
    """
    Run after all tests, to parse and store the test results
    """
    from loguru import logger
    if hasattr(pytest, 'config'):
        logger.debug(pytest.config)
    if hasattr(pytest, 'fn') and hasattr(pytest, 'result'):
        logger.info(pytest.fn)
        logger.info(pytest.result)


def update_params(fixture_name: str, request) -> None:
    """
    Update the (currently used) params to be in the file names using request.param

    :param fixture_name: The fixture whose params to update
    :param request: pytest param's request
    """
    for key in request.param:
        if key not in pytest.fn_params[fixture_name]:
            continue
        else:
            pytest.fn_params[fixture_name][key] = request.param[key]
            
            
def wandb_init(caller_fn: str) -> None:
    """
    Init wandb, and upload the current fn_params (listed config params) and non-list config params to wandb
    """
    if wandb.run is not None:
        wandb.finish()  # Finish previous run
    config = pytest.config[caller_fn]
    wandb_config = {}
    for fixture_name in config:
        for key in config[fixture_name]:
            if type(config[fixture_name][key]) is list:
                wandb_config[f'{key} ({fixture_name})'] = pytest.fn_params[fixture_name][key]
            else:
                wandb_config[f'{key} ({fixture_name})'] = config[fixture_name][key]
                
    wandb.init(project=pytest.fn, entity='point-process', config=wandb_config)
    # wandb.init(mode="disabled")
    
    
def wandb_discard(id) -> None:
    """
    Discard the current wandb run
    """
    api = wandb.Api()
    run = api.run(f"point-process/{pytest.fn}/{id}")
    run.delete()


def relpath(prefix: str, create_dir: bool = False) -> str:
    """
    Get the relative path of the caller to the project root, and go down one level by
    appending the config parameters at the end of path.

    :param prefix: the folder under which structure is created
    :param create_dir: whether to create a folder whose name is the parameters
    :return: the relative path to the bottom-most folder
    """
    fn_ = inspect.stack()[1].filename
    new_path = f'{relpath_under(prefix, fn_, True)}/{serialize_config(pytest.fn_params)}'
    if create_dir and not os.path.exists(new_path):
        os.makedirs(new_path)
    return new_path


def log_config() -> None:
    """
    Log the config for current test
    """
    from loguru import logger
    logger.info(pytest.fn_params)


def plot_training_progress(train_losses, val_losses, file_name) -> None:
    import plotly.graph_objects as go
    import numpy as np

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='train',  marker_size=4))
    fig.add_trace(go.Scatter(x=np.arange(9, 1000, 10)[:len(val_losses)], y=val_losses, mode='lines', 
                                name='val', marker_size=4))

    fig.update_layout(
        title=r"Training progress",
        xaxis_title="x",
        yaxis_title="f",
    )

    fig.write_html(f"{file_name}.html")
