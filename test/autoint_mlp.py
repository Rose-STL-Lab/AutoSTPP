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


@pytest.fixture(scope="class")
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


@pytest.fixture(
    scope="class",
    params=get_params('cat_linear_model', importer())  # The importer file name 
)
def cat_linear_model(device, request):
    # Load activation function
    from integration.autoint import MixSequential, CatLinear, act_dict
    from torch import nn
    update_params("cat_linear_model", request)

    act_layer = act_dict[request.param['act']]

    # Construct MLP layers
    assert request.param['n_layers'] >= 1
    layers = [CatLinear(request.param['inp_dim'], request.param['hid_dim'], request.param['num_group'])]
    for _ in range(request.param['n_layers'] - 1):
        layers.append(act_layer)
        layers.append(CatLinear(request.param['hid_dim'], request.param['hid_dim'], request.param['num_group'],
                                bias=request.param['bias']))
    layers.append(act_layer)
    layers.append(CatLinear(request.param['hid_dim'], request.param['out_dim'], request.param['num_group'],
                            bias=request.param['bias']))
    layers.append(nn.Linear(request.param['num_group'] * request.param['out_dim'], request.param['out_dim'],
                            bias=request.param['bias']))

    return MixSequential(*layers).to(device)


@pytest.fixture(scope="class")
def add_cuboid(device, model, cat_linear_model):
    from torch.nn import Linear
    from integration.autoint import Cuboid, CatLinear
    from integration.autoint import AddMixSequential, MixSequential, Neg

    M = AddMixSequential(model, cat_linear_model)
    model_layers = []
    cat_linear_model_layers = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Linear):
            if isinstance(layer, CatLinear):
                model_layers.append(CatLinear(layer.in_features, layer.out_features, 
                                              layer.num_group))
            else:
                model_layers.append(layer.__class__(layer.in_features, layer.out_features))
        else:
            model_layers.append(layer)
    model_layers.append(Neg())
    for i, layer in enumerate(cat_linear_model.layers):
        if isinstance(layer, Linear):
            if isinstance(layer, CatLinear):
                cat_linear_model_layers.append(CatLinear(layer.in_features, layer.out_features, 
                                                         layer.num_group))
            else:
                cat_linear_model_layers.append(layer.__class__(layer.in_features, layer.out_features))
        else:
            cat_linear_model_layers.append(layer)
    cat_linear_model_layers.append(Neg())
    L0 = MixSequential(*model_layers).to(device)
    L1 = MixSequential(*cat_linear_model_layers).to(device)
    L = AddMixSequential(L0, L1)

    return Cuboid(L, M)


@pytest.fixture(
    scope="class",
    params=get_params('sum_prodnet_cuboid', importer())  # The importer file name 
)
def sum_prodnet_cuboid(device, request):
    import torch
    from integration.autoint import Cuboid, ProdNet, SumNet, CatNet, Prod, MultSequential
    
    cuboids = torch.nn.ModuleList([])
    for _ in range(request.param['n_kernel']):
        if request.param['composite']:
            L_prod_nets = [MultSequential(CatNet(bias=True, neg=True), torch.nn.Linear(3, 3), Prod())
                            for _ in range(request.param['n_prodnet'])]
            M_prod_nets = [MultSequential(CatNet(bias=True), torch.nn.Linear(3, 3), Prod())
                            for _ in range(request.param['n_prodnet'])]
        else:
            L_prod_nets = [ProdNet(out_dim=1, bias=True, neg=True) 
                            for _ in range(request.param['n_prodnet'])]
            M_prod_nets = [ProdNet(out_dim=1, bias=True) for _ in range(request.param['n_prodnet'])]
        cuboid = Cuboid(L=SumNet(*L_prod_nets), M=SumNet(*M_prod_nets)).to(device)
        cuboids.append(cuboid)
    return cuboids
