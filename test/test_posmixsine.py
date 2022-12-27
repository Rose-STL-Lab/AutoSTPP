def test_sine_model():
    from integration.autoint import MultSequential, Sine
    from torch import nn
    import torch
    
    model = MultSequential(
        nn.Linear(3, 3),
        Sine()
    )
    model_wo_sine = MultSequential(
        model[0]
    )
    model[0].weight
    
    x = torch.tensor([[1., 1., 1.]])

    expect = 1.
    for i in range(3):
        expect = expect * model_wo_sine.dnforward(x, [i])
    expect = expect * -torch.cos(model_wo_sine.forward(x))
    result = model.dnforward(x, [0, 1, 2])
    assert torch.allclose(result, expect)
    
    x = torch.tensor([[1., 1., 1.]])
    expect = 1.
    for i in [0, 0, 0]:
        expect = expect * model_wo_sine.dnforward(x, [i])
    expect = expect * -torch.cos(model_wo_sine.forward(x))
    result = model.dnforward(x, [0, 0, 0])


def test_posmixsine():
    from integration.autoint import BaselineSequential, PosMixSine, ReflectExp
    from torch import nn
    import torch
    
    flag = False
    flag2 = False
    for _ in range(100):
        model = BaselineSequential(
            PosMixSine(),
            nn.Linear(3, 128),
            ReflectExp(),
            nn.Linear(128, 128),
            ReflectExp(),
            nn.Linear(128, 1)
        )
        model[0].weight = nn.Parameter(torch.eye(3) * 2)
        model.project()
        from loguru import logger
        x = torch.rand([1, 3]) * 2
        try:
            assert model.dnforward(x, [0, 1, 2]) >= 0   # Positive
        except AssertionError:
            print(model.dnforward(x, [0, 1, 2]))
            raise
        if model.dnforward(x, [0, 0, 0]) < 0:  # Negative
            flag = True
        if model.dnforward(x, [0, 1, 1]) < 0:  # Negative
            flag2 = True
    assert flag
    assert flag2
    
    
def test_posmixsine_impl():
    from integration.autoint import BaselineSequential, MultSequential, ReflectExp, PosMixSine
    from torch import nn
    import torch
    
    model = BaselineSequential(
        ReflectExp(),
        nn.Linear(3, 3),
        PosMixSine(),
        nn.Linear(3, 128),
        ReflectExp(),
        nn.Linear(128, 128),
        ReflectExp(),
        nn.Linear(128, 1)
    )
    model.project()
    ours = MultSequential(*model)
    
    x = torch.rand(2, 3)
    
    assert torch.allclose(model.dnforward(x, [0, 1, 2]), ours.dnforward(x, [0, 1, 2]))
    assert torch.allclose(model.dnforward(x, [0, 1]), ours.dnforward(x, [0, 1]))
    assert torch.allclose(model.dnforward(x, [0]), ours.dnforward(x, [0]))
    assert torch.allclose(model.dnforward(x, [0, 2]), ours.dnforward(x, [0, 2]))
    assert torch.allclose(model.dnforward(x, [1, 1]), ours.dnforward(x, [1, 1]))
    assert torch.allclose(model.dnforward(x, [2, 2]), ours.dnforward(x, [2, 2]))
        

def test_wo_posmix():
    from integration.autoint import MultSequential, ReflectExp
    from torch import nn
    import torch
    
    for _ in range(100):
        model = MultSequential(
            nn.Linear(3, 128),
            ReflectExp(),
            nn.Linear(128, 128),
            ReflectExp(),
            nn.Linear(128, 1)
        )
        # model[0].weight = nn.Parameter(torch.eye(3))
        model.project()
        x = torch.rand([1, 3]) * 2
        assert model.dnforward(x, [0, 1, 2]) >= 0
        assert model.dnforward(x, [0, 0, 0]) >= 0
        assert model.dnforward(x, [1, 1, 1]) >= 0
