"""
Test the AddMixSequential layer
Motivation: force the mixed derivative (d/dx d/dy d/dz) to be positive 
while the univariate derivative (d/dx d/dx d/dx) is unconstrained

The Mixed derivative is kept positive by adding a network with positive 
triple mixed derivative activation with a network with possibly negative 
univariate derivative activation but, since input does not interact, has 
zero mixed derivative.
"""


def test_mixseq():
    """
    Test MixSequential alone
    """
    from integration.autoint import MultSequential, ReflectExp
    from torch import nn
    import torch
        
    flag = [False, False, False]
    for _ in range(100):
        model = MultSequential(
            nn.Linear(3, 128),
            ReflectExp(),
            # nn.Linear(128, 128),
            # ReflectExp(),
            nn.Linear(128, 1),
        )
        model.project()
        x = torch.rand([1, 3]) - 0.5
        assert model.dnforward(x, [0, 1, 2]) >= 0
        assert model.dnforward(x, [0, 0, 0]) >= 0
        assert model.dnforward(x, [1, 1, 1]) >= 0
        
        # Derivative of intensity over any axis can be negative
        # print(model.dnforward(x, [0, 1, 2, 0]))
        if model.dnforward(x, [0, 1, 2, 0]) < 0:
            flag[0] = True
        if model.dnforward(x, [0, 1, 2, 1]) < 0:
            flag[1] = True
        if model.dnforward(x, [0, 1, 2, 2]) < 0:
            flag[2] = True
    # print(flag)
    assert flag[0]
    assert flag[1]
    assert flag[2]
    

def cat_linear_models():
    """
    Non-fixture method to return two models
    """
    from integration.autoint import BaselineSequential, MultSequential, CatLinear
    from torch import nn
    
    model = BaselineSequential(
        CatLinear(1, 128, 3),
        nn.Tanh(),
        CatLinear(128, 128, 3),
        nn.Tanh(),
        CatLinear(128, 1, 3),
        nn.Linear(3, 1)
    )
    ours = MultSequential(*model)
    model.project()
    return model, ours
    
    
def test_cat_linear_impl():
    """
    Test implementation of CatLinear
    """
    import torch
    
    model, ours = cat_linear_models()
    x = torch.rand(2, 3)
    
    assert torch.allclose(model.dnforward(x, [0, 1, 2]), ours.dnforward(x, [0, 1, 2]))
    assert torch.allclose(model.dnforward(x, [0, 1]), ours.dnforward(x, [0, 1]))
    assert torch.allclose(model.dnforward(x, [0]), ours.dnforward(x, [0]))
    assert torch.allclose(model.dnforward(x, [0, 2]), ours.dnforward(x, [0, 2]))
    assert torch.allclose(model.dnforward(x, [1, 1]), ours.dnforward(x, [1, 1]))
    assert torch.allclose(model.dnforward(x, [2, 2]), ours.dnforward(x, [2, 2]))


def test_cat_linear():
    """
    Test CatLinear eliminate mix derivative, kept positive first derivative 
    and sometimes negative second univariate derivative 
    """
    import torch
    
    flag = False
    flag2 = False
    for _ in range(100):
        model = cat_linear_models()[0]
        N = 2
        x = torch.rand(N, 3)
        assert torch.allclose(model.dnforward(x, [0, 1, 2]), torch.zeros(N, 1))
        if torch.any(model.dnforward(x, [0, 0]) < 0):
            flag = True
        if torch.any(model.dnforward(x, [1, 1]) < 0):
            flag2 = True
    assert flag
    assert flag2


def addmixseq():
    """
    AddMixSequential model
    """
    from integration.autoint import AddMixSequential, MixSequential, ReflectExp
    from torch import nn
    import torch
    
    model = AddMixSequential(
        MixSequential(
            nn.Linear(3, 128),
            ReflectExp(),
            nn.Linear(128, 128),
            ReflectExp(),
            nn.Linear(128, 1)
        ),
        MixSequential(*cat_linear_models()[0], nn.Linear(1, 1))
    )
    model.seq[-1].layers[-1].weight = nn.Parameter(torch.Tensor([[10.]]))
    return model


def test_addmixseq():
    """
    Test the positivity of mixed derivative and 
    the possible negativity of univariate derivative
    """
    import torch
    
    flag = False
    flag2 = False
    for _ in range(100):
        model = addmixseq()
        model.project()
        x = torch.rand([1, 3]) * 2
        try:
            assert model.dnforward(x, [0, 1, 2]) >= 0   # Positive
        except AssertionError:
            print(model.dnforward(x, [0, 1, 2]))
            raise
        if model.dnforward(x, [0, 0, 0]) < 0:  # Negative
            flag = True
        if model.dnforward(x, [1, 1]) < 0:  # Negative
            flag2 = True
    assert flag
    assert flag2
