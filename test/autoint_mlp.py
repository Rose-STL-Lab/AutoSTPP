import pytest
from conftest import update_params

params = {}


@pytest.fixture(
    scope="class",
    params=pytest.params['model']
)
def model(device, request):
    import torch
    from torch import nn
    from integration.autoint import MixSequential, ReQU, ReQUFlip

    update_params("model", request)

    class Sine(nn.Module):
        def __init__(self):
            super().__init__()  # init the base class

        @staticmethod
        def forward(x):
            return torch.sin(x)

    # Load activation function
    act_dict = {
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "sine": Sine(),
        "requ": ReQU(),
        "requflip": ReQUFlip()
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
