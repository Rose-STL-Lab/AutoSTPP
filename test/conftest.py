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
    from utils import load_config, dict_to_list, project_root, relpath_under, serialize_config
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
    logger.add(f'{project_root()}/logs/{fn}.log', format=fmt, level="DEBUG")

    configs: Dict[str, Dict[str, List]] = load_config(fn)

    pytest.fn_params = {}  # Parameters that need to be in the filename
    for fixture_name in configs:
        for config_name in configs[fixture_name]:
            if type(configs[fixture_name][config_name]) is list:
                if fixture_name not in pytest.fn_params:
                    pytest.fn_params[fixture_name] = {}
                pytest.fn_params[fixture_name][config_name] = ''
            else:
                configs[fixture_name][config_name] = [configs[fixture_name][config_name]]   # Make it a list

    # Methods added to pytest module
    def relpath(prefix: str):
        import inspect
        fn_ = inspect.stack()[1].filename
        return f'{relpath_under(prefix, fn_)}/{serialize_config(pytest.fn_params)}'

    def update_params(fixture_name_, request):
        for key in request.param:
            if key not in pytest.fn_params[fixture_name_]:
                continue
            else:
                pytest.fn_params[fixture_name_][key] = request.param[key]

    setattr(pytest, 'update_params', update_params)
    setattr(pytest, 'relpath', relpath)

    pytest.params = {fixture_name: dict_to_list(configs[fixture_name]) for fixture_name in configs}
    pytest.config = configs
    return configs
