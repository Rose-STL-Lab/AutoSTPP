import pytest
from conftest import update_params, get_params
from utils import importer

params = {}


@pytest.fixture(
    scope="class",
    params=get_params('model', importer())  # The importer file name 
)
def model(device, request):
    from torch import nn
    from integration.autoint import MixSequential, ReQU, ReQUFlip, Sine, SineFlip

    update_params("model", request)

    # Load activation function
    act_dict = {
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "sine": Sine(),
        "requ": ReQU(),
        "requflip": ReQUFlip(),
        "sineflip": SineFlip()
    }
    act_layer = act_dict[request.param['act']]

    # Construct MLP layers
    assert request.param['n_layers'] >= 1
    layers = [nn.Linear(request.param['inp_dim'], request.param['hid_dim']), ]
    for _ in range(request.param['n_layers'] - 1):
        layers.append(act_layer)
        layers.append(nn.Linear(request.param['hid_dim'], request.param['hid_dim']))
    layers.append(act_layer)
    layers.append(nn.Linear(request.param['hid_dim'], request.param['out_dim']))

    return MixSequential(*layers).to(device)


@pytest.fixture(
    scope="class"
)
def cuboid(device, model):
    from copy import deepcopy
    from integration.autoint import Cuboid
    from integration.autoint import ReQU, ReQUFlip, Sine, SineFlip

    M = model
    L = deepcopy(model)
    for i, layer in enumerate(M.layers):  # Flip all activation function in M
        if type(layer) == ReQU:
            L.layers[i] = ReQUFlip()
        if type(layer) == Sine:
            L.layers[i] = SineFlip()

    return Cuboid(L, M)
