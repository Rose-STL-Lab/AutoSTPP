from typing import List, Dict

import pytest
from typeguard import typechecked


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
    from utils import load_config, dict_to_list, project_root
    from loguru import logger
    import sys

    # NOTICE: test config should always be dictionary of lists
    fn = sys.argv[-1]  # Current test file name
    if '.py' not in fn:
        logger.error('Test config file not found')
    fn = fn.split('.')[0]

    # Add a file logger
    fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
    logger.add(f'{project_root()}/logs/{fn}.log', format=fmt, level="DEBUG")

    configs: Dict[str, Dict[str, List]] = load_config(fn)
    pytest.params = {module_name: dict_to_list(configs[module_name]) for module_name in configs}
    pytest.config = configs
    return configs
