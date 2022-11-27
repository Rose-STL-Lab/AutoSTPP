from datetime import datetime
from typing import Dict, Any, List

from tqdm.auto import tqdm as tqdm_, trange as trange_
from tqdm.contrib import tenumerate as tenumerate_
from loguru import logger

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import io
import torch

import subprocess
import re
import time
import sys

import plotly.io as pio
import yaml
import inspect
import os
import dotenv

import itertools

eps = 1e-10
STEP = 1e-3
t_start = 0.0
t_end = 50.0
x = np.arange(0.0, 1.0 + eps, STEP)
x *= (t_end - t_start)
plt.set_loglevel('warning')
pio.renderers.default = "png"

dotenv.load_dotenv(override=True)  # Loading environment variable

tqdm = lambda *argv: tqdm_(*argv, position=0, leave=True, ncols=100)  # Override tqdm for PyCharm compatibility
trange = lambda *argv: trange_(*argv, position=0, leave=True, ncols=100)
tenumerate = lambda *argv: tenumerate_(*argv, position=0, leave=True, ncols=100)

default_fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def imshow(fig):
    """
    Display a plotly PNG figure in PyCharm by converting it to a Matplotlib PNG figure
    """
    width = 800  # pixels
    height = 600
    margin = 0  # pixels
    dpi = 300.  # dots per inch

    img_bytes = fig.to_image(format="png", height=height, width=width, scale=10.0)
    fp = io.BytesIO(img_bytes)
    with fp:
        i = mpimg.imread(fp, format='png')

    fig_size = ((width + 2 * margin) / dpi, (height + 2 * margin) / dpi)  # inches
    left = margin / dpi / fig_size[0]  # axes ratio
    bottom = margin / dpi / fig_size[1]

    fig = plt.figure(figsize=fig_size, dpi=dpi)
    fig.subplots_adjust(left=left, bottom=bottom, right=1. - left, top=1. - bottom)

    plt.imshow(i)
    plt.show()


def gen_fn(**kwargs):
    """
    Generate the file name used for save model / output \n
    Using the main file name, time, and additional kwargs

    :param kwargs: additional param e.g. lr=0.3 to contain in the file name
    :return: str file name
    """
    file_name = sys.modules['__main__'].__file__.split("/")[len(__file__.split("/"))].split(".")[0]
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    res = file_name + '_' + time_str
    for key, value in kwargs.items():
        res += '_{0}={1}'.format(key, value)
    return res


def get_device(free=True, min_ram=8000):
    """
    Get GPU device with maximum amount of free memory \n
    Halt if no qualified GPU found

    :param free: proceed only if there are unused GPU
    :param min_ram: the minimum RAM (in MiB) required to proceed
    :return: torch.device
    """
    command = 'nvidia-smi'
    try:
        p = subprocess.check_output(command)
    except FileNotFoundError as err:
        logger.exception(f'{command} utility not found.')
        raise err

    # Parse Nvidia interface for GPU info
    p = str(p).split('\n')
    ram_using = re.findall(r'\b\d+MiB+ */', str(p))
    ram_total = re.findall(r'/ *\b\d+MiB', str(p))
    ram_using = [int(''.join(filter(str.isdigit, ram))) for ram in ram_using]
    ram_total = [int(''.join(filter(str.isdigit, ram))) for ram in ram_total]

    try:
        assert len(ram_using) == len(ram_total)
    except AssertionError as err:
        logger.exception(f'parse {command} failure.')
        raise err

    try:
        assert len(ram_using) > 0
    except AssertionError as err:
        logger.exception('GPU not found.')
        raise err

    ram_free = [rt - ru for ru, rt in zip(ram_using, ram_total)]
    idx = int(np.argmax(ram_free))

    # No free GPU found, sleep until found
    flag = True
    while (free and ram_using[idx] > 10) or ram_free[idx] < min_ram:
        if flag:
            flag = False
            logger.info(f'free CUDA device not found, waiting...')
        time.sleep(10)
    if not flag:
        logger.info(f'free CUDA device found')

    logger.debug(f'CUDA device {idx} name: {torch.cuda.get_device_name(idx)}')
    logger.debug(f'RAM total: {ram_total[idx]} | using: {ram_using[idx]} | free: {ram_free[idx]}')
    logger.debug(f'Number of CUDAs(cores): {torch.cuda.device_count()}')

    return torch.device(f'cuda:{idx}')


def adapt(arr, device):  # Adapt to device, add batch number and dim number = 1
    return torch.tensor(arr).float().unsqueeze(0).unsqueeze(-1).to(device)


def gt(lamb_func, his_t):
    """
    Compute ground truth intensity

    :param lamb_func: 1D lambda function
    :param his_t: np array [seq_len,]
    """
    y = np.array([lamb_func(t, his_t)[0] for t in x])
    return y


def loglike(his_t, y):
    """
    Estimate LL given his_t and intensity
    """
    F = sum(y) * STEP * (t_end - t_start)
    lamb = y[np.array(his_t[his_t < t_end] / (t_end - t_start) / STEP, dtype=int)]
    ll = sum(np.log(lamb)) - F
    return ll


def evaluate(lamb_func, predict, model, seqs):
    """
    Calculate λ MAPE and log likelihood w.r.t. a dataset
    """
    lls = []
    mapes = []
    for seq in tqdm(seqs):
        his_t = seq.numpy()
        y = gt(lamb_func, his_t)
        y_predict = predict(model, his_t, x)

        ll = loglike(his_t, y_predict)
        mape = np.mean(abs((y_predict - y) / y))

        lls.append(ll)
        mapes.append(mape)

    return sum(mapes) / len(mapes), sum(lls) / len(lls)


def plot_predict_intensity(lamb_func, predict, model, his_t, color='blue'):
    """
    Plot λt vs. time
    """
    width, _ = plt.figaspect(.1)
    _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(width, width / 3),
                                 gridspec_kw={'height_ratios': [5, 1]})

    # Calculate and Plot the intensity
    y = gt(lamb_func, his_t)
    ax1.plot(x, y, color, label=f'λ(t)')
    ax1.set_xlim([t_start, t_end])
    ax1.legend()
    ll = loglike(his_t, y)
    logger.info(f'ground truth: {ll}')

    y_predict = predict(model, his_t, x)
    ll = loglike(his_t, y_predict)
    logger.info(f'predict: {ll}')

    # Calculate MAPE 
    mape = np.mean(abs((y_predict - y) / y))
    logger.info(f'λ MAPE: {mape}')

    ax1.plot(x, y_predict, 'red', label='predict')
    ax1.set_xlim([t_start, t_end])
    ax1.legend()

    # Plot the events
    idx = np.logical_and(his_t >= t_start, his_t < t_end)
    x_ = np.zeros(np.sum(idx) + 2)
    x_[1:-1] = his_t[idx]
    x_[0] = t_start
    x_[-1] = t_end
    y_ = np.ones_like(x_)
    y_[0] = 0
    y_[-1] = 0

    ax2.stem(x_, y_, use_line_collection=True, label=f'Events')
    ax2.set_xlim([t_start, t_end])
    ax2.invert_yaxis()


def nested_stack(tensors):
    """
    Stack a nested list of tensors by considering each level as a new dimension at dim0

    :param tensors: a nested list of tensors
    :return: the concatenated tensor
    """
    if type(tensors) == torch.Tensor:
        return tensors
    elif type(tensors) == float:
        return torch.tensor(tensors)
    else:
        assert type(tensors) == list
        return torch.stack([nested_stack(tl) for tl in tensors], 0)


def eval_loss(model, test_loader):
    model.eval()
    sll_meter = AverageMeter()
    tll_meter = AverageMeter()
    loss_meter = AverageMeter()

    for index, data in enumerate(test_loader):
        st_x, st_y, _, _, _ = data
        loss, sll, tll = model(st_x, st_y)

        loss_meter.update(loss.item())
        sll_meter.update(sll.mean().item())
        tll_meter.update(tll.mean().item())

    return loss_meter.avg, sll_meter.avg, tll_meter.avg


def project_root() -> str:
    """
    Get the project root's absolute path, assuming this util file is under src/

    :return: the project root's absolute path
    """
    home_fn = os.path.abspath(__file__) + "/../../"  # Project root
    home_fn = os.path.abspath(home_fn)
    return home_fn


def load_config(caller_fn: str = '') -> Dict[str, Any]:
    """
    Load the config yaml file (in `configs` folder) for the corresponding model.

    :param caller_fn: name of the config yaml file (without postfix) to load, default to the caller file name
    :return: dictionary with config entries
    """
    if caller_fn == '':  # No filename given
        caller_fn = inspect.stack()[1].filename  # Absolute path
    caller_fn = os.path.basename(caller_fn).split('.')[0]
    home_fn = project_root()
    with open(f'{home_fn}/configs/{caller_fn}.yaml', "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)


def relpath_under(prefix: str, fn: str = None, create_dir: bool = False) -> str:
    """
    Get the relative path of the caller to the project root, and
    under a folder, create a sub-folder with the same relative path
    for saving items (including intermediate directories)

    :param prefix: the folder under which structure is created
    :param fn: the caller file name, find automatically if not provided
    :param create_dir: whether to create a folder, default to false
    :return: the relative path to the bottom-most folder
    """
    if fn is None:
        fn = inspect.stack()[1].filename  # Absolute path of the caller
    fn = fn.split('.')[0]
    home_fn = project_root()
    relpath = os.path.relpath(fn, home_fn)
    new_path = f'{home_fn}/{prefix}/{relpath}'
    if create_dir and not os.path.exists(new_path):
        os.makedirs(new_path)
    return new_path


def dict_to_list(config: Dict[str, List]) -> List[Dict[str, Any]]:
    """
    Convert config dictionary to list of all possible options

    :param config: a dictionary mapping config name -> list of config options
    :return: a list of dictionary which each config entry maps to single value
    """
    for key in config:
        config[key] = [{key: item} for item in config[key]]  # Append key to each item

    product = list(itertools.product(*config.values()))
    return [{k: v for d in tup for k, v in d.items()} for tup in product]


def serialize_config(config: Dict) -> str:
    """
    Convert a json-like config to a file name
    :param config: a double dictionary mapping
        fixture name -> config name -> list of config options
    :return: a short str of the config, with following special characters: {}[]_,=
    """
    return str(config).replace(':', '=').replace(' ', '').replace('\'', '')[1:-1]


def scale(tensor, bound):
    """
    Scale a uniform tensor using MinMax's
    :param tensor: tensor shape (N, d)
    :param bound: a list of d two-tuples defining MinMax
    :return: the scaled tensor with shape (N, d)
    """
    d = len(bound)
    assert len(bound) == len(tensor[0])
    for i in range(d):
        tensor[:, i] = tensor[:, i] * (bound[i][1] - bound[i][0]) + bound[i][0]
    return tensor


def arange(N: int, bound, lib: Any = np):
    """
    Higher (d) dimensional version of np/torch arange

    :param N: number of data points for each dimension
    :param bound: a list of `d` two-tuples defining MinMax
    :param lib: the tensor library, numpy or torch
    :return: ((N+1)**d, d) border-inclusive tensor uniformly covering the bounded region
    """
    assert lib == torch or lib == np, "Unsupported library"
    d = len(bound)
    ticks = lib.arange(0., 1. + 1. / N, 1. / N)
    X = lib.meshgrid(*([ticks] * d))
    X = lib.vstack([X_.flatten() for X_ in X]).T
    if bound is not None:
        X = scale(X, bound)
    return X


def importer() -> str:
    """
    Get the import filename of the current module
    """
    flag = False
    for frame in inspect.stack():
        if 'importlib' in frame.filename:
            flag = True
        elif flag:
            return frame.filename
    logger.error("Unable to find importer filename")
    raise FileNotFoundError


if __name__ == '__main__':
    # logger.info(get_device(min_ram=0))
    X_flat = arange(100, [[0., 1.], [0., 2.], [0., 3.]], lib=np)
    logger.info(X_flat)
    logger.info(X_flat.shape)
