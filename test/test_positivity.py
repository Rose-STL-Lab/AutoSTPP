"""
Test composite the traditional activation function with 
a positive mixed catlinear heart 
"""


def composite_models():
    """
    Non-fixture method to return two models
    """
    from integration.autoint import BaselineSequential, MultSequential, CatLinear, ReflectSoft
    from torch import nn
    
    model = BaselineSequential(
        CatLinear(1, 64, 3),
        nn.Tanh(),
        nn.Linear(64 * 3, 128),
        ReflectSoft(),
        nn.Linear(128, 128),
        ReflectSoft(),
        nn.Linear(128, 1)
    )
    ours = MultSequential(*model)
    model.project()
    return model, ours


def prodnet():
    """
    Non-fixture method to return a prodnet model
    """
    from integration.autoint import SumNet, ProdNet
    
    model = SumNet(
        ProdNet(out_dim=1, bias=True),
        ProdNet(out_dim=1, bias=True),
        ProdNet(out_dim=1, bias=True)
    )
    model.project()
    return model


def seq_prodnets1():
    from integration.autoint import BaselineSequential, MultSequential, ProdNet, ReflectExp
    from torch import nn
    
    model = BaselineSequential(
        ProdNet(out_dim=128, bias=False),
        ReflectExp(),
        nn.Linear(128, 1)
    )
    ours = MultSequential(*model)
    model.project()
    return model, ours


# def seq_prodnets2():
#     from integration.autoint import BaselineSequential, MultSequential, ProdNet
#     from torch import nn
    
#     model = BaselineSequential(
#         nn.Linear(3, 3),
#         ProdNet(out_dim=128, bias=False),
#         nn.Linear(128, 1)
#     )
#     ours = MultSequential(*model)
#     model.project()
#     return model, ours


def composite_prodnets():
    from torch import nn
    from integration.autoint import SumNet, ProdNet, ReflectExp, MultSequential, BaselineSequential
    
    model = SumNet(
        BaselineSequential(
            ProdNet(out_dim=128, bias=False),
            ReflectExp(),
            nn.Linear(128, 1)
        ),
        BaselineSequential(
            ProdNet(out_dim=1, bias=True)
        )
    )
    ours = SumNet(MultSequential(*model.nets[0]), MultSequential(*model.nets[1]))    
    model.project()
    return model, ours


def composite_catnet():
    from torch import nn
    from integration.autoint import MultSequential, CatNet, Prod, ReflectExp
    
    model = MultSequential(
        CatNet(),
        nn.Linear(3, 3),
        # Prod(),
        ReflectExp(),
        nn.Linear(3, 1)
    )
    model.project()
    return model


def test_composite_impl():
    """
    Test implementation of ours vs baseline
    """
    import torch
    
    for model, ours in [seq_prodnets1(), composite_models(), composite_prodnets()]:
        x = torch.rand(2, 3)
        
        # from loguru import logger
        # logger.critical(f'[0, 1, 2]: {model.dnforward(x, [0, 1, 2])}, {ours.dnforward(x, [0, 1, 2])}')
        # logger.critical(f'[0, 1]: {model.dnforward(x, [0, 1])}, {ours.dnforward(x, [0, 1])}')
        # logger.critical(f'[0]: {model.dnforward(x, [0])}, {ours.dnforward(x, [0])}')
        # logger.critical(f'[0, 2]: {model.dnforward(x, [0, 2])}, {ours.dnforward(x, [0, 2])}')
        # logger.critical(f'[1, 1]: {model.dnforward(x, [1, 1])}, {ours.dnforward(x, [1, 1])}')
        # logger.critical(f'[1]: {model.dnforward(x, [1])}, {ours.dnforward(x, [1])}')
        # logger.error(ours.pd)
        
        assert torch.allclose(model.dnforward(x, [0, 1, 2]), ours.dnforward(x, [0, 1, 2]))
        assert torch.allclose(model.dnforward(x, [0, 1]), ours.dnforward(x, [0, 1]))
        assert torch.allclose(model.dnforward(x, [0]), ours.dnforward(x, [0]))
        assert torch.allclose(model.dnforward(x, [0, 2]), ours.dnforward(x, [0, 2]))
        assert torch.allclose(model.dnforward(x, [1, 1]), ours.dnforward(x, [1, 1]))
        assert torch.allclose(model.dnforward(x, [2, 2]), ours.dnforward(x, [2, 2]))
    
    
def test_prodnet_impl():
    """
    Test implementation of the product net
    """
    import torch
    model = prodnet().nets[0]
    
    x = torch.rand([1, 3]) * 2
    pred = model.dnforward(x, [0, 1])
    real = model.x_seq.dnforward(x[..., 0:1], [0]) * \
           model.y_seq.dnforward(x[..., 1:2], [0])
    real *= model.t_seq.forward(x[..., 2:3])
    assert torch.allclose(real, pred)
    
    
def test_cmp_catnet_prodnet():
    """
    Compare prodnet and catnet
    """
    import torch
    from integration.autoint import ProdNet, CatNet, MultSequential, Prod, Scale
    prodnet = ProdNet(out_dim=1, bias=True)
    catnet = CatNet(bias=True)
    catnet.x_seq = prodnet.x_seq
    catnet.y_seq = prodnet.y_seq
    catnet.t_seq = prodnet.t_seq
    catnet = MultSequential(catnet, Prod(), Scale(3.))
        
    # Test equivalence 
    x = torch.rand(2, 3)
    assert torch.allclose(prodnet.forward(x), catnet.forward(x))
    assert torch.allclose(prodnet.dnforward(x, [0, 1, 2]), catnet.dnforward(x, [0, 1, 2]))
    assert torch.allclose(prodnet.dnforward(x, [0, 1]), catnet.dnforward(x, [0, 1]))
    assert torch.allclose(prodnet.dnforward(x, [0]), catnet.dnforward(x, [0]))
    assert torch.allclose(prodnet.dnforward(x, [0, 2]), catnet.dnforward(x, [0, 2]))
    assert torch.allclose(prodnet.dnforward(x, [1, 1]), catnet.dnforward(x, [1, 1]))
    assert torch.allclose(prodnet.dnforward(x, [2, 2]), catnet.dnforward(x, [2, 2]))
    
    
def test_composite():
    """
    Test the positivity of mixed derivative and 
    the possible negativity of univariate derivative
    """
    import torch
    from loguru import logger
    
    neg_dims = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 0], [1, 1], [2, 2],
                [0, 1, 0], [1, 2, 1], [2, 0, 2]]
    flags = {}
    for _ in range(200):
        # model = composite_models()[1]
        model = prodnet()
        # model = composite_prodnets()[1]
        
        # model[0].weight = nn.Parameter(torch.eye(3))
        
        x = torch.rand([1, 3]) * 2
        x = abs(x)  # Assume input is always positive
        
        dims = [0, 1, 2]
        dnf = model.dnforward(x, dims)
        logger.debug(f'f^{dims}: {dnf}')
        assert dnf >= 0., f'f^{dims} is sometimes negative (should be always positive)'
        
        for dims in neg_dims:
            if str(dims) not in flags:
                flags[str(dims)] = False
            dnf = model.dnforward(x, dims)
            logger.debug(f'f^{dims}: {dnf}')
            if dnf < 0:
                flags[str(dims)] = True
            
    for dims in flags:
        assert flags[str(dims)], f'f^{dims} is always positive (should be sometimes negative)'


def test_composite_catnet():
    """
    Test the positivity of mixed derivative and 
    the possible negativity of univariate derivative
    """
    import torch
    from loguru import logger
    
    neg_dims = [[0, 1], [0, 2], [1, 2], [0, 0], [1, 1], [2, 2],
                [0, 1, 0], [1, 2, 1], [2, 0, 2]]
    flags = {}
    for _ in range(1000):
        model = composite_catnet()
        
        x = torch.rand([1, 3]) * 2
        x = abs(x)  # Assume input is always positive
        
        dims = [0, 1, 2]
        dnf = model.dnforward(x, dims)
        logger.debug(f'f^{dims}: {dnf}')
        assert dnf >= 0., f'f^{dims} is sometimes negative (should be always positive)'
        
        for dims in neg_dims:
            if str(dims) not in flags:
                flags[str(dims)] = False
            dnf = model.dnforward(x, dims)
            logger.debug(f'f^{dims}: {dnf}')
            if dnf < 0:
                flags[str(dims)] = True
            
    for dims in flags:
        assert flags[str(dims)], f'f^{dims} is always positive (should be sometimes negative)'
