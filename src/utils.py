from datetime import datetime
from typing import Dict, Any, List

from tqdm import tqdm as tqdm_, trange as trange_
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
import functools
import warnings
import resource
import importlib

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
deprecated_func_names = []


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
    
    :param lamb_func: _description_
    :param predict: a function of the form 
        <quote>    
        Compute model intensities at different time
        :param his_t: np array [seqlen,], the event time history
        :param x: np array [N,], a batch of times
        </quote>
    :param model: _description_
    :param seqs: _description_
    :return: the mean of MAPE and LL
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


def plot_predict_intensity(lamb_func, predict, model, his_t, color='blue', t_end=t_end):
    """
    Plot λt vs. time
    """
    width, _ = plt.figaspect(.1)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(width, width / 3),
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
    return fig


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
    if '.py' in fn:
        fn = fn[:fn.rindex('.py')]
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


def arange(N, bound, lib: Any = np):
    """
    Higher (d) dimensional version of np/torch arange

    :param N: number of data points for each dimension, can be an int or a list of `d` ints
    :param bound: MinMax for each dimension, must be a list of `d` two-tuples
    :param lib: the tensor library, numpy or torch
    :return: ((N+1)**d, d) border-inclusive tensor uniformly covering the bounded region
    """
    assert lib == torch or lib == np, "Unsupported library"
    d = len(bound)
    if type(N) is int:
        N = [N] * d
    else:
        assert len(N) == d, "N and bound's dimension must match"
    ticks = []
    for n in N:
        assert type(n) == int, "Number of data points must be an int"
        assert n >= 1, "Number of data points must be at least 1"
        tick = lib.arange(0., 1. + 0.1 / n, 1. / n)  # Add a small number to include the upper bound
        ticks.append(tick)
    X = lib.meshgrid(*ticks)
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


def deprecated(func):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        global deprecated_func_names
        if func.__name__ not in deprecated_func_names:
            deprecated_func_names.append(func.__name__)
            warnings.simplefilter('always', DeprecationWarning)  # turn off filter
            warnings.warn("Call to deprecated function {}.".format(func.__name__),
                          category=DeprecationWarning,
                          stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
        
    return new_func


def get_minmax(seqs, roll=True, device=torch.device("cuda:0")):
    """
    Get MinMax scalers of space and delta_t such that spaces are mapped to [0, 1]^d
    and delta_t has an average of 1

    :param seqs: a list of sequences of np shape [N, 3], time is the first dimension
    :param roll: whether to roll the time to the last dimension
    :return: a tuple of (min, max) of torch shape [3]
    """
    for i, seq in enumerate(seqs):
        seqs[i][:, 0] = np.diff(seq[:, 0], axis=0, prepend=0)
    if roll:
        seqs = [np.roll(seq, -1, -1) for seq in seqs]        
    temp = np.vstack(seqs)
    min = torch.tensor(np.min(temp, 0)).float().to(device)
    max = torch.tensor(np.max(temp, 0)).float().to(device)
    if roll:
        min[-1] = 0.
        max[-1] = np.mean(temp[:, -1])
    else:
        min[0] = 0.
        max[0] = np.mean(temp[:, 0])
    return min, max


def scale_ll(dataloader, nll, sll, tll, scales=None):
    """
    Scale the log likelihoods to the original scale
    
    :param dataloader: dataloader of a SlidingWindowWrapper (has `min` and `max` attribute)
    :param nll: the negative total log likelihood
    :param sll: the spatial log likelihood
    :param tll: the temporal log likelihood
    :param scales: a list of (s0_scale, s1_scale, t_scale), if dataloader is not given
    """
    if dataloader is not None:
        assert hasattr(dataloader.dataset, 'max') and hasattr(dataloader.dataset, 'min')
        s0_scale, s1_scale, t_scale = dataloader.dataset.max - dataloader.dataset.min
    else:
        assert scales is not None
        s0_scale, s1_scale, t_scale = scales
    if type(t_scale) == torch.Tensor:
        t_scale = t_scale.item()
        s0_scale = s0_scale.item()
        s1_scale = s1_scale.item()
    t_scale = np.log(t_scale)
    s_scale = np.log(s0_scale * s1_scale)
    return nll + s_scale + t_scale, sll - s_scale, tll - t_scale


def get_update_time(model_fn: str):
    """
    Given a model file, return the file's last update time

    :param model_fn: the absolute/relative path to the model file
    """
    return datetime.fromtimestamp(os.path.getmtime(model_fn))


def find_ckpt_path(hash_str, aim_path='.aim'):
    aim_path = os.path.abspath(aim_path)
    largest_N = -1
    largest_N_path = None
    
    # Walk through all subdirectories of `.aim` recursively, skipping certain directories
    skip_dirs = ['.aim/check_ins', '.aim/locks', '.aim/meta', '.aim/progress', '.aim/seqs']
    for dirpath, _, filenames in os.walk(aim_path):
        if any(skip_dir in dirpath for skip_dir in skip_dirs):
            # Skip directories in `skip_dirs`
            continue
        if hash_str in dirpath:
            for f in filenames:
                if f.endswith('.ckpt'):
                    checkpoint_path = os.path.join(dirpath, f)
                    checkpoint_info = f[:-5].split('-')
                    N = int(checkpoint_info[1].split('=')[1])
                    if N > largest_N:
                        largest_N = N
                        largest_N_path = checkpoint_path

    return largest_N_path


def load_class(info: dict):
    assert 'init_args' in info
    assert 'class_path' in info
    class_path = info['class_path']
    init_args = info['init_args']
    
    # Split the class path into module and class names
    module_name, class_name = class_path.rsplit(".", 1)

    # Import the module
    module = importlib.import_module(module_name)
    
    # Load the class and initialize with the given arguments
    cls = getattr(module, class_name)
    instance = cls(**init_args)
    
    return instance


def increase_u_limit():
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, rlimit[1]))


if __name__ == '__main__':
    # logger.info(get_device(min_ram=0))
    # X_flat = arange(100, [[0., 1.], [0., 2.], [0., 3.]], lib=np)
    # logger.info(X_flat)
    # logger.info(X_flat.shape)
    logger.info(find_ckpt_path('e855a0'))
