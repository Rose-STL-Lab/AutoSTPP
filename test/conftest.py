from typing import List, Dict

import pytest
from typeguard import typechecked
from copy import deepcopy
from utils import relpath_under, serialize_config
import inspect
import os


@pytest.fixture(scope='module', autouse=True)
def device():
    from utils import get_device

    return get_device(free=False, min_ram=8000)


@typechecked
def pytest_configure() -> Dict[str, Dict[str, List]]:
    """
    Load the test config for current test, as a dictionary of lists.
    All combinations of listed config option will be tested.

    :return: a double dictionary mapping
        fixture name -> config name -> list of config options
    """
    from utils import load_config, dict_to_list
    from loguru import logger
    import sys

    # NOTICE: test config should always be dictionary of lists
    fn = sys.argv[-1]  # Current test file name
    if '.py' not in fn:
        logger.error('Test config file not found')
        raise FileNotFoundError
    fn = fn.split('.')[0]

    # Add a file logger
    fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
    logger.add(f'{relpath_under("logs", fn)}.log', format=fmt, level="DEBUG")

    configs: Dict[str, Dict[str, List]] = load_config(fn)
    pytest.config = deepcopy(configs)

    pytest.fn_params = {}  # Parameters that need to be in the filename
    for fixture_name in pytest.config:
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

    pytest.params = {fixture_name: dict_to_list(configs[fixture_name]) for fixture_name in configs}
    return configs


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
