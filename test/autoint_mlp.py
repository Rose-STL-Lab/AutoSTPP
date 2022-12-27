import pytest
from conftest import update_params, get_params
from utils import importer

params = {}


@pytest.fixture(
    scope="class",
    params=get_params('model', importer())  # The importer file name 
)
def model(device, request):
    # Load activation function
    from integration.autoint import MixSequential, act_dict
    from torch import nn
    update_params("model", request)

    act_layer = act_dict[request.param['act']]

    # Construct MLP layers
    assert request.param['n_layers'] >= 1
    if 'posmix' in request.param:
        layers = [act_dict[request.param['posmix']], nn.Linear(request.param['inp_dim'], request.param['hid_dim']), ]
    else:
        layers = [nn.Linear(request.param['inp_dim'], request.param['hid_dim']), ]
    for _ in range(request.param['n_layers'] - 1):
        layers.append(act_layer)
        layers.append(nn.Linear(request.param['hid_dim'], request.param['hid_dim'], bias=request.param['bias']))
    layers.append(act_layer)
    layers.append(nn.Linear(request.param['hid_dim'], request.param['out_dim'], bias=request.param['bias']))

    return MixSequential(*layers).to(device)


@pytest.fixture(
    scope="class",
    params=get_params('cuboid', importer())  # The importer file name 
)
def cuboid(device, model):
    from torch.nn import Linear
    from integration.autoint import Cuboid
    from integration.autoint import MixSequential, Neg

    M = model
    layers = []
    for i, layer in enumerate(M.layers):
        if isinstance(layer, Linear):
            layers.append(Linear(layer.in_features, layer.out_features))
        else:
            layers.append(layer)
    layers.append(Neg())
    L = MixSequential(*layers).to(device)

    return Cuboid(L, M)
