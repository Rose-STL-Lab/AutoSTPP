from itertools import combinations
from typing import List, Generator
import itertools
from copy import deepcopy

import numpy as np
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from torch import nn, autograd
from torch.nn import functional as F

from utils import nested_stack, deprecated
from loguru import logger

patch_typeguard()
batch = None  # Suppress PyCharm type warning


class AddMixSequential(nn.Module):
    
    def __init__(self, *args):
        super().__init__()
        for arg in args:
            assert isinstance(arg, MixSequential)
        self.seq: List[MultSequential] = nn.ModuleList(args)
        
    def forward(self, x):
        res = sum([seq(x) for seq in self.seq])
        return res
    
    def dnforward(self, x, dim):
        """
        Calculate d(n)x / dx1 dx2 ... dxn as a summation of all submodules

        :param x: (..., dim)
        :param dims: unordered list [x1, x2, ..., xn], can be repetitive
        :return: (..., dim), d(n)x / dx1 dx2 ... dxn
        """
        res = sum([seq.dnforward(x, dim) for seq in self.seq])
        return res
    
    def project(self) -> None:
        for seq in self.seq:
            seq.project()
            
            
class Exp(nn.Module):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def forward(x):
        return torch.exp(x)
            
            
class Neg(nn.Module):
    def __init__(self):
        super().__init__()  # init the base class

    @staticmethod
    def forward(x):
        return -x


class PointReflect(nn.Module):
    def __init__(self, shape: int, func):
        super().__init__()  # init the base class
        assert shape % 2 == 0
        self.shape = shape
        self.func = func

    def forward(self, x):
        assert x.shape[-1] == self.shape
        half = self.shape // 2
        return torch.cat([self.func(x[..., :half]), -self.func(-x[..., half:])], -1)
        

class Re4U(nn.Module):
    def __init__(self):
        super().__init__()  # init the base class

    @staticmethod
    def forward(x):
        return torch.where(x > 0., x ** 4 / 24. + x ** 3 / 6. + x ** 2 / 2. + x, 
                           torch.exp(x) - 1.)
    

class ReQU(nn.Module):
    def __init__(self):
        super().__init__()  # init the base class

    @staticmethod
    def forward(x):
        return torch.where(x > 0., 0.125 * x ** 2 + 0.5 * x + np.log(2.),
                           torch.log(torch.exp(x) + torch.tensor(1.)))
        

class LogInt(nn.Module):
    def __init__(self):
        super().__init__()  # init the base class

    @staticmethod
    def forward(x):
        return torch.where(x > 1., x**3 / 36 + x**3 * torch.log(x) / 6 + 0.75 * x**2 + 11 * x / 12 + 1 / 72,
                           torch.where(x > 0., x**4 / 24 + x**3 / 6 + x**2 / 2 + x,
                                       torch.exp(x) - 1.))


class ReflectExp(nn.Module):
    def __init__(self):
        super().__init__()  # init the base class

    @staticmethod
    def forward(x):
        return torch.where(x > 0., -torch.exp(-x) + x**2 + 1., torch.exp(x) - 1.)
    
    
class ReflectExpShift(nn.Module):
    def __init__(self):
        super().__init__()  # init the base class
    
    b = 0.5  # Shift bias

    def forward(self, x):
        b = ReflectExpShift.b
        return torch.where(x > 0., -torch.exp(b - x) + (x - b)**2 + 2., 
                           torch.exp(x - b)) - np.exp(-b) + 1.
    
    def dforward(self, f, x):
        b = ReflectExpShift.b
        return torch.where(x > 0., torch.exp(b - x) + (x - b) * 2, torch.exp(x - b))
    
    def d2forward(self, f, x):
        b = ReflectExpShift.b
        return torch.where(x > 0., -torch.exp(b - x) + 2, torch.exp(x - b))
    
    def d3forward(self, f, x):
        b = ReflectExpShift.b
        return torch.where(x > 0., torch.exp(b - x), torch.exp(x - b))
    

class ReflectExpInt(nn.Module):
    def __init__(self):
        super().__init__()  # init the base class

    @staticmethod
    def forward(x):
        return torch.where(x > 0., torch.exp(-x) + x**3 / 3. + 2 * x - 1., torch.exp(x) - 1.)


class ReflectSoft(nn.Module):
    def __init__(self):
        super().__init__()  # init the base class
    
    s = 0.8  # Scale factor

    @staticmethod
    def forward(x):
        bias = 2 - np.sqrt(3.)
        result = torch.where(x > 0., 
                             (x * (x + 6) - 6 * torch.log(torch.exp(x) + bias)) / 6. + 2 * np.log(1 + bias), 
                             torch.log(1 + torch.exp(x + np.log(bias))))
        return ReflectSoft.s * result
        
    @staticmethod
    def dforward(f, x):
        bias = np.sqrt(3.) - 2
        result = torch.where(x > 0., 
                             bias * torch.exp(-x) / (bias * torch.exp(-x) - 1) + x / 3.,
                             bias * torch.exp(x) / (bias * torch.exp(x) - 1))
        return ReflectSoft.s * result
        
    @staticmethod
    def d2forward(f, x):
        bias = np.sqrt(3.) - 2
        result = torch.where(x > 0., 
                             (bias * torch.exp(-x) + 1)**2 / 4. / (bias * torch.exp(-x) - 1)**2 + 1 / 12.,
                             -(bias * torch.exp(x) + 1)**2 / 4. / (bias * torch.exp(x) - 1)**2 + 1 / 4.) 
        return ReflectSoft.s * result
      
    @staticmethod
    def d3forward(f, x):
        bias = np.log(2 - np.sqrt(3.))
        result = torch.where(x > 0., 
                             torch.exp(x - bias) * (torch.exp(x - bias) - 1) / (1 + torch.exp(x - bias))**3,
                             -torch.exp(x + bias) * (torch.exp(x + bias) - 1) / (1 + torch.exp(x + bias))**3)
        return ReflectSoft.s * result
        
        
class Sine(nn.Module):
    def __init__(self):
        super().__init__()  # init the base class

    @staticmethod
    def forward(x):
        return torch.sin(x)
    
    
class PosMixSine(nn.Module):
    def __init__(self):
        super().__init__()  # init the base class

    @staticmethod
    def forward(x):
        assert x.shape[-1] == 3
        # logger.critical(x[..., 0:1] * x[..., 1:2] * x[..., 2:3] / 3.)
        return torch.sin(x) + x[..., 0:1] * x[..., 1:2] * x[..., 2:3] / 3.


class Prod(nn.Module):
    def __init__(self):
        super().__init__()  # init the base class

    @staticmethod
    def forward(x):
        assert x.shape[-1] == 3
        return x[..., 0:1] * x[..., 1:2] * x[..., 2:3] / 3.
    
    
class Scale(nn.Module):
    def __init__(self, weight=1., bias=0.):
        super().__init__()
        assert isinstance(weight, float)
        assert isinstance(bias, float)
        self.weight = weight
        self.bias = bias
    
    def forward(self, x):
        return self.weight * x + self.bias
    
    
class SumNet(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.nets = nn.ModuleList([])
        for net in args:
            assert isinstance(net, nn.Module)
            assert hasattr(net, 'forward')
            assert hasattr(net, 'dnforward')
            self.nets.append(net)
            
    def forward(self, x):
        result = 0.
        for net in self.nets:
            result += net.forward(x)
        return result
    
    def dnforward(self, x, dims):
        result = 0.
        for net in self.nets:
            result += net.dnforward(x, dims)
        return result
    
    def project(self):
        for net in self.nets:
            net.project()


class ProdNet(nn.Module):
    def __init__(self, neg=False, bias=True, out_dim=1, hidden_size=128, num_layers=2, activation=nn.Tanh()):
        
        super().__init__()  # init the base class
        self.out_dim = out_dim
        
        def create_layers(num_layers):
            layers = [nn.Linear(1, hidden_size, bias), activation]
            for _ in range(num_layers - 1):
                layers += [nn.Linear(hidden_size, hidden_size, bias), activation]
            layers += [nn.Linear(hidden_size, out_dim, bias)]
            return layers
        
        self.x_seq = MultSequential(*create_layers(num_layers))
        self.y_seq = MultSequential(*create_layers(num_layers))
        self.t_seq = MultSequential(*create_layers(num_layers))
        
        if neg:
            self.x_seq.append(Neg())
            self.y_seq.append(Neg())
            self.t_seq.append(Neg())
        
    def dnforward(self, x, dims):
        prod = None
        n_count = dims.count(0)
        if n_count > 0:
            prod = self.x_seq.dnforward(x[..., 0:1], [0] * n_count)
        else:
            prod = self.x_seq(x[..., 0:1])
        n_count = dims.count(1)
        if n_count > 0:
            prod *= self.y_seq.dnforward(x[..., 1:2], [0] * n_count)
        else:
            prod *= self.y_seq(x[..., 1:2])
        n_count = dims.count(2)
        if n_count > 0:
            prod *= self.t_seq.dnforward(x[..., 2:3], [0] * n_count)
        else:
            prod *= self.t_seq(x[..., 2:3])
        return prod
            
    def forward(self, x):
        return self.x_seq(x[..., 0:1]) * \
            self.y_seq(x[..., 1:2]) * \
            self.t_seq(x[..., 2:3])
        
    def project(self):
        self.x_seq.project()
        self.y_seq.project()
        self.t_seq.project()
        
        
class CatNet(nn.Module):
    def __init__(
        self, 
        neg=False, 
        bias=True, 
        num_layers=2,
        hidden_size=128,
        activation=nn.Tanh()
    ):
        super().__init__()  # init the base class
        # Always 3 -> 3
        self.x_seq, self.y_seq, self.t_seq = [MultSequential(
            nn.Linear(1, hidden_size, bias=bias),
            activation,
            *itertools.chain(*[[nn.Linear(hidden_size, hidden_size, bias=bias), activation] 
                               for _ in range(num_layers - 1)]),
            nn.Linear(hidden_size, 1, bias=bias),
            *([] if not neg else [Neg()])
        ) for _ in range(3)]
        
    def dnforward(self, x, dims):
        to_cat = []
        n_count = dims.count(0)
        if len(dims) > n_count:
            to_cat.append(torch.zeros_like(x[..., 0:1]))
        elif n_count > 0:
            to_cat.append(self.x_seq.dnforward(x[..., 0:1], [0] * n_count))
        else:
            to_cat.append(self.x_seq(x[..., 0:1]))
        n_count = dims.count(1)
        if len(dims) > n_count:
            to_cat.append(torch.zeros_like(x[..., 0:1]))
        elif n_count > 0:
            to_cat.append(self.y_seq.dnforward(x[..., 1:2], [0] * n_count))
        else:
            to_cat.append(self.y_seq(x[..., 1:2]))
        n_count = dims.count(2)
        if len(dims) > n_count:
            to_cat.append(torch.zeros_like(x[..., 0:1]))
        elif n_count > 0:
            to_cat.append(self.t_seq.dnforward(x[..., 2:3], [0] * n_count))
        else:
            to_cat.append(self.t_seq(x[..., 2:3]))
        return torch.cat(to_cat, dim=-1)
            
    def forward(self, x):
        return torch.cat([self.x_seq(x[..., 0:1]), self.y_seq(x[..., 1:2]), 
                          self.t_seq(x[..., 2:3])], dim=-1)
        
    def project(self):
        self.x_seq.project()
        self.y_seq.project()
        self.t_seq.project()


# Activation functions
act_dict = {
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "sine": Sine(),
    "relu": nn.ReLU(),
    "requ": ReQU(),
    "elu": nn.ELU(),
    "re4u": Re4U(),
    "logint": LogInt(),
    "reflectexp": ReflectExp(),
    "reflectsoft": ReflectSoft(),
    "reflectexpint": ReflectExpInt(),
    "reflectexpshift": ReflectExpShift(),
    "posmixsine": PosMixSine(),
}

# Derivative functions
MAX_GRAD_DIM = 100
# TODO: Implement this in respective classes
grad_dict = {
    'Tanh': [lambda f, _: 1 - f ** 2,
             lambda f, _: -2 * f * (1 - f ** 2),
             lambda f, _: 8 * f ** 2 - 6 * f ** 4 - 2,
             lambda f, _: 24 * f ** 5 - 40 * f ** 3 + 16 * f],
    'Sigmoid': [lambda f, _: f * (1 - f),
                lambda f, _: 2 * f ** 3 - 3 * f ** 2 + f,
                lambda f, _: - 6 * f ** 4 + 12 * f ** 3 - 7 * f ** 2 + f],
    'Sine': [lambda _, x: torch.cos(x),
             lambda f, _: -f,
             lambda _, x: -torch.cos(x),
             lambda f, _: f] * (MAX_GRAD_DIM // 4),
    'ReQU': [lambda _, x: torch.where(x > 0., 0.25 * x + 0.5, 1. / (1. + torch.exp(-x))),
             lambda _, x: torch.where(x > 0., 0.25 * torch.ones_like(x), torch.exp(-x) / (1 + torch.exp(-x)) ** 2),
             lambda _, x: torch.where(x > 0., torch.zeros_like(x),
                                      - torch.exp(x) * (torch.exp(x) - 1.) / (1. + torch.exp(x)) ** 3),
             lambda _, x: torch.where(x > 0., torch.zeros_like(x),
                                      torch.exp(x) * (-4 * torch.exp(x) + torch.exp(2 * x) + 1) /
                                      (1. + torch.exp(x)) ** 4)],
    'Softplus': [lambda _, x: torch.exp(0.5 * x) / (1. + torch.exp(0.5 * x)),
                 lambda _, x: 0.5 * torch.exp(0.5 * x) / (1. + torch.exp(0.5 * x)) ** 2,
                 lambda _, x: 0.25 * (torch.exp(0.5 * x) - torch.exp(x)) / (1. + torch.exp(0.5 * x)) ** 3],
    'ELU': [lambda _, x: torch.where(x > 0., torch.ones_like(x), torch.exp(x))] + 
           [lambda _, x: torch.where(x > 0., torch.zeros_like(x), torch.exp(x))] * MAX_GRAD_DIM,
    'Re4U': [lambda _, x: torch.where(x > 0., x ** 3 / 6. + x ** 2 / 2. + x + 1., torch.exp(x)),
                lambda _, x: torch.where(x > 0., x ** 2 / 2. + x + 1., torch.exp(x)),
                lambda _, x: torch.where(x > 0., x + 1, torch.exp(x))],
    'LogInt': [lambda _, x: torch.where(x > 1., 0.25 * x**2 + 0.5 * x**2 * torch.log(x) + 1.5 * x + 11. / 12,
                                        torch.where(x > 0., x**3 / 6 + x**2 / 2 + x + 1., torch.exp(x))),
                lambda _, x: torch.where(x > 1., torch.log(x) * x + x + 1.5,
                                         torch.where(x > 0., x**2 / 2 + x + 1., torch.exp(x))),
                lambda _, x: torch.where(x > 1., torch.log(x) + 2,
                                         torch.where(x > 0., x + 1., torch.exp(x)))],
    'ReflectExp': [lambda _, x: torch.where(x > 0., torch.exp(-x) + 2 * x, torch.exp(x)),
                   lambda _, x: torch.where(x > 0., -torch.exp(-x) + 2, torch.exp(x))] +
                  [lambda _, x: torch.where(x > 0., torch.exp(-x), torch.exp(x)),
                   lambda _, x: torch.where(x > 0., -torch.exp(-x), torch.exp(x))] * (MAX_GRAD_DIM // 2),
    'ReflectExpInt': [lambda _, x: torch.where(x > 0., -torch.exp(-x) + x**2 + 2, torch.exp(x)),
                      lambda _, x: torch.where(x > 0., torch.exp(-x) + x * 2, torch.exp(x)),
                      lambda _, x: torch.where(x > 0., -torch.exp(-x) + 2, torch.exp(x))],
    'Neg': [lambda _, x: -torch.ones_like(x)] + 
           [lambda _, x: torch.zeros_like(x)] * MAX_GRAD_DIM,
    'ReLU': [lambda _, x: torch.where(x > 0., torch.ones_like(x), torch.zeros_like(x))] +
            [lambda _, x: torch.zeros_like(x)] * MAX_GRAD_DIM,
    'ReflectSoft': [ReflectSoft.dforward, ReflectSoft.d2forward, ReflectSoft.d3forward],
    'ReflectExpShift': [ReflectExpShift.dforward, ReflectExpShift.d2forward, ReflectExpShift.d3forward],
    'PosMixSine': [lambda _, x: torch.cos(x),
                   lambda _, x: -torch.sin(x),
                   lambda _, x: -torch.cos(x),
                   lambda _, x: torch.sin(x)] * (MAX_GRAD_DIM // 4)
}


class MultSequential(nn.Sequential):
    """
    The optimized integral-net
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.x = None
        self.f = []
        self.dnf = {}
        self.grad_dict = grad_dict

    @staticmethod
    def hash(dims: List[int]) -> str:
        """
        Hash a list of deriving dimensions to str dict key
        :param dims: an unordered list of integers
        :return: str of ordered dims (such that [1,2] and [2,1] refer to the same derivative)
        """
        return str(sorted(dims))  # Hash the dimension

    @staticmethod
    def partition(ns: List[int], m: int) -> Generator:
        """
        Finding all k-subset partitions
        Algorithm U, Knuth, the Art of Computer Programming
        Implemented by Adeel Zafar Soomro
        https://codereview.stackexchange.com/questions/1526/finding-all-k-subset-partitions

        :param ns: an unordered list of integers
        :param m: number of ways
        :return: generate ways to partition list as m subsets
        """

        def visit(n, a):
            ps = [[] for _ in range(m)]
            for j in range(n):
                ps[a[j + 1]].append(ns[j])
            return ps

        def f(mu, nu, sigma, n, a):
            if mu == 2:
                yield visit(n, a)
            else:
                for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                    yield v
            if nu == mu + 1:
                a[mu] = mu - 1
                yield visit(n, a)
                while a[nu] > 0:
                    a[nu] = a[nu] - 1
                    yield visit(n, a)
            elif nu > mu + 1:
                if (mu + sigma) % 2 == 1:
                    a[nu - 1] = mu - 1
                else:
                    a[mu] = mu - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                while a[nu] > 0:
                    a[nu] = a[nu] - 1
                    if (a[nu] + sigma) % 2 == 1:
                        for v in b(mu, nu - 1, 0, n, a):
                            yield v
                    else:
                        for v in f(mu, nu - 1, 0, n, a):
                            yield v

        def b(mu, nu, sigma, n, a):
            if nu == mu + 1:
                while a[nu] < mu - 1:
                    yield visit(n, a)
                    a[nu] = a[nu] + 1
                yield visit(n, a)
                a[mu] = 0
            elif nu > mu + 1:
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                while a[nu] < mu - 1:
                    a[nu] = a[nu] + 1
                    if (a[nu] + sigma) % 2 == 1:
                        for v in f(mu, nu - 1, 0, n, a):
                            yield v
                    else:
                        for v in b(mu, nu - 1, 0, n, a):
                            yield v
                if (mu + sigma) % 2 == 1:
                    a[nu - 1] = 0
                else:
                    a[mu] = 0
            if mu == 2:
                yield visit(n, a)
            else:
                for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                    yield v

        n = len(ns)
        a = [0] * (n + 1)
        for j in range(1, m + 1):
            a[n - m + j] = j - 1
        return f(m, n, 0, n, a)
    
    @staticmethod
    def partition2(ns: List[int], b: int) -> Generator:
        """
        Finding ways to put n elements in b bins, allowing for empty bins

        :param ns: a list of n elements
        :param b: number of bins
        :return: generate ways to put n elements in b bins
        """
        masks = np.identity(b, dtype=int)
        for c in itertools.combinations_with_replacement(masks, len(ns)):
            split = sum(c)

            def draw(n, lst):
                """
                Ways to draw n numbers from lst
                e.g. draw 2 from [1,2,3] = [[1,2], [1,3], [2,3]]
                """
                if n == 0:
                    yield []
                else:
                    for i, x in enumerate(lst):
                        for y in draw(n - 1, lst[i + 1:]):
                            yield [x] + y
            
            res = [{'drawn': [], 'left': ns}]
            for i, num_in_bin in enumerate(split):
                while len(res[0]['drawn']) <= i:
                    sample = res.pop(0)
                    for drawn in draw(num_in_bin, sample['left']):
                        left_ = deepcopy(sample['left'])
                        drawn_ = deepcopy(sample['drawn'])
                        for num in drawn:
                            left_.remove(num)
                        drawn_.append(drawn)
                        res.append({'drawn': drawn_, 'left': left_})
            yield [dic['drawn'] for dic in res]

    def reset(self) -> None:
        """
        Reinitialize all buffer, when x changes
        """
        del self.dnf
        self.f = []
        self.dnf = {}

    def forward(self, x):
        """
        Compute f and store intermediate steps

        :param x: (batch, dim)
        """
        if x is self.x:
            if len(self.f) > 0:
                return self.f[-1]
        else:
            self.reset()
            self.x = x

        self.f = [x, ]
        inp = x
        for module in self:
            inp = module(inp)
            self.f.append(inp)

        return self.f[-1]
    
    @deprecated
    def dforward(self, x, dim):
        """
        Calculate 1st derivative

        :param x: (..., dim)
        :param dim: the dimension to derive
        :raises NotImplementedError: if activation derivative does not exist
        :return: (..., dim)
        """
        assert type(dim) == int
        key = str(dim)
        
        _ = self.forward(x)
        base = torch.zeros_like(x)
        base[..., dim] = 1
        self.dnf[key] = pd = [base, ]
            
        # Perform chain rule: df(g(x))/dx = f'(g(x)) g'(x)
        for i, module in enumerate(self):
            tp = type(module).__name__
            if tp == 'Linear':
                pd.append(pd[-1] @ module.weight.T)
            elif tp == 'NonNegLinear':
                pd.append(pd[-1] @ nn.functional.relu(module.weight).T)
            elif tp == 'PadLinear':
                pd.append(pd[-1] @ module.padded_weight().T)
            elif tp in self.grad_dict: 
                pd.append(self.grad_dict[tp][0](self.f[i + 1], self.f[i]) * pd[-1])
            else:
                raise NotImplementedError
        
        return pd[-1]

    def dnforward(self, x, dims):
        """
        Start recursion for d(n)x / dx1 dx2 ... dxn

        :param x: (..., dim)
        :param dims: unordered list [x1, x2, ..., xn], can be repetitive
        :return: (..., dim), d(n)x / dx1 dx2 ... dxn
        """
        assert type(dims) == list
        key = self.hash(dims)
        N = len(dims)
        if x is self.x:
            if key in self.dnf:
                return self.dnf[key][-1]
        else:
            self.x = x
            self.reset()

        assert N > 0
        if N == 1:  # Base case
            if len(self.f) == 0:
                _ = self.forward(x)  # Prepare f values
            base = torch.zeros_like(x)
            assert dims[0] < x.shape[-1], "Deriving dimension exceeds output dimension"
            base[..., dims[0]] = 1
            self.dnf[key] = pd = [base, ]
        else:
            for dim in combinations(dims, N - 1):
                if self.hash(list(dim)) not in self.dnf:
                    _ = self.dnforward(x, list(dim))  # Prepare derivatives of lower order
            self.dnf[key] = pd = [torch.zeros_like(x), ]

        for i, module in enumerate(self):
            tp = type(module).__name__
            if hasattr(module, 'dnforward'):
                if i == 0:
                    pd.append(module.dnforward(x, dims))
                else:
                    raise NotImplementedError('Other AutoInt model can only be the first layer')
            elif tp == 'Scale':
                pd.append(pd[-1] * module.weight)
            elif tp == 'Linear':
                pd.append(pd[-1] @ module.weight.T)
            elif tp == 'NonNegLinear':
                pd.append(pd[-1] @ F.relu(module.weight).T)
            elif tp == 'PadLinear':
                pd.append(pd[-1] @ module.padded_weight().T)
            elif tp == 'CatLinear':
                pd.append(pd[-1] @ module.cat_weight().T)
            elif tp == 'Prod':
                """
                df1(x,y,z:X)f2(X)f3(X)/dx = 
                df1'(X)/dx f2(X)f3(X) + f1(X) df2'(X)/dx f3(X) + f1(X)f2(X) df3'(X)/dx
                When f1=x, f2=y, f3=z, df1'(X)/dx = 1, df2'(X)/dx = 0, df3'(X)/dx = 0
                yz + 0xz + 0xy = yz
                
                df1(X)f2(X)f3(X)/dxdy = 
                df1(X)/dxdy f2(X) f3(X) + df1(X)/dx df2(X)/dy f3(X) + df1(X)/dx f2(X) df3(X)/dy +
                df2(X)/dxdy f1(X) f3(X) + df1(X)/dy df2(X)/dx f3(X) + f1(X) df2(X)/dx df3(X)/dy +
                df3(X)/dxdy f1(X) f2(X) + df1(X)/dy f2(X) df3(X)/dx + f1(X) df2(X)/dy df3(X)/dx
                When f1=x, f2=y, f3=z, z
                
                df1(X)f2(X)f3(X)/dxdydz
                df1(X)/dxdydz f2(X) f3(X) + ...
                When f1=x, f2=y, f3=z, 1
                """
                term = 0.
                f_term = self.f[i]
                for ways in self.partition2(dims, 3):
                    for way in ways:
                        temp = 1.
                        assert len(way) == 3
                        for j, dims_ in enumerate(way):
                            if len(dims_) == 0:
                                temp = f_term[:, j:j + 1] * temp
                            else:
                                df_term = self.dnf[self.hash(dims_)][i]
                                temp = df_term[:, j:j + 1] * temp
                        term += temp / 3.
                pd.append(term)
            elif tp in self.grad_dict:
                acs = self.grad_dict[tp]
                if N == 1:
                    term_sum = acs[0](self.f[i + 1], self.f[i]) * pd[-1]  # Base case
                else:
                    term_sum = 0.
                    """
                    ac(n)   * df/dx1 * df/dx2 * df/dx3 ... * df/dxn + 
                    ...
                    ac''    * (d2f/dx1dx2.. * df/dxn + ... ) +
                    ac'     * dnf/dx1dx2dx3...dxn
                    ways to partition x1...xn to k sets time ac(k)
                    """
                    for order in range(N):
                        if order == 0:
                            term = pd[-1]
                        else:
                            term = 0.
                            for part in self.partition(dims, order + 1):
                                temp = 1.
                                for dim in part:
                                    temp = temp * self.dnf[self.hash(list(dim))][i]
                                term += temp
                        assert order < len(acs), "Activation high-order derivative not implemented"
                        term_sum += term * acs[order](self.f[i + 1], self.f[i])
                if tp == 'PosMixSine':  # Handle the + f(x)*f(y)*f(z) part
                    f_term = self.f[i]
                    for ways in self.partition2(dims, 3):
                        for way in ways:
                            temp = 1.
                            assert len(way) == 3
                            for j, dims_ in enumerate(way):
                                if len(dims_) == 0:
                                    temp = f_term[:, j:j + 1] * temp
                                else:
                                    df_term = self.dnf[self.hash(dims_)][i]
                                    temp = df_term[:, j:j + 1] * temp
                            term_sum += temp / 3.
                pd.append(term_sum)
            else:
                raise NotImplementedError(f"Unknown layer type: {tp}")

        return pd[-1]
    
    def project(self) -> None:
        """Employ negative / non-negative constraint"""
        with torch.no_grad():
            for layer in self:
                if hasattr(layer, 'project'):
                    layer.project()
                elif isinstance(layer, PadLinear):
                    pass  # No need to clamp because t is "leaked"
                elif isinstance(layer, CatLinear):
                    for weight in layer.weights:
                        weight.clamp_(min=0.0, max=None)
                elif isinstance(layer, nn.Linear):
                    layer.weight.clamp_(min=0.0, max=None)


class BaselineSequential(nn.Sequential):
    """
    The baseline integral-net implemented using PyTorch's built-in graph
    """
    def __init__(self, *args):
        super().__init__(*args)

    def dnforward(self, x, dims):
        """
        Start recursion for d(n)x / dx1 dx2 ... dxn

        :param x: (..., dim)
        :param dims: unordered list [x1, x2, ..., xn], can be repetitive
        :return: (..., dim), d(n)x / dx1 dx2 ... dxn
        """
        if x.is_leaf:
            x.requires_grad = True

        if len(dims) == 1:
            df = self  # Base case
        else:
            df = lambda x_: self.dnforward(x_, dims[:-1])  # Derivative with one fewer order

        d2f = df(x)
        assert dims[-1] < x.shape[-1], "Deriving dimension exceeds input dimension"
        return autograd.grad(d2f, x, torch.ones_like(d2f), create_graph=True)[0][:, dims[-1:]]
    
    def project(self) -> None:
        """Employ negative / non-negative constraint"""
        with torch.no_grad():
            for layer in self:
                if hasattr(layer, 'dnforward'):
                    layer.project()
                elif isinstance(layer, PadLinear):
                    pass  # No need to clamp because t is "leaked"
                elif isinstance(layer, CatLinear):
                    for weight in layer.weights:
                        weight.clamp_(min=0.0, max=None)
                elif isinstance(layer, nn.Linear):
                    layer.weight.clamp_(min=0.0, max=None)


class MixSequential(nn.Module):
    """
    Combining the baseline and the optimized solution by
    using the optimized solution as base cases (when number of dims < threshold)
    """

    def __init__(self, *args, threshold=3):
        super().__init__()
        self.layers = MultSequential(*args)  # Composition rather than inheritance
        self.threshold = threshold

    def forward(self, x):
        return self.layers.forward(x)

    def dnforward(self, x, dims):
        """
        Start recursion for d(n)x / dx1 dx2 ... dxn

        :param x: (..., dim)
        :param dims: unordered list [x1, x2, ..., xn], can be repetitive
        :return: (..., dim), d(n)x / dx1 dx2 ... dxn
        """
        if x.is_leaf:
            x.requires_grad = True

        if len(dims) <= self.threshold:
            return self.layers.dnforward(x, dims[:self.threshold])  # Base case
        else:
            df = self.dnforward(x, dims[:-1])  # Derivative with one fewer order

        return autograd.grad(df, x, torch.ones_like(df), create_graph=True)[0][..., dims[-1:]]
    
    def project(self) -> None:
        """Employ negative / non-negative constraint"""
        self.layers.project()


class PadLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, last_t=False):
        """
        Allowing an input feature to be unchanged by the linear layer 
        and kept as an output feature
        
        :param in_features: number of input features
        :param out_features: number of output features
        :param bias: whether to use bias
        :param last_t: whether the unchanged input feature is the first or the last
        """
        super().__init__(in_features, out_features, bias)
        self.last_t = last_t

    def padded_weight(self):
        """
        Return the weight matrix with a padded row and column
        allowing a feature to be unchanged by the linear layer and kept as an output feature
        
        :return: a torch matrix of size (self.out_features + 1, self.in_features + 1)
        """
        weight = torch.zeros(self.out_features + 1, self.in_features + 1).to(self.weight.device)
        if self.last_t:
            weight[:-1, :-1] = self.weight
            weight[-1, -1] = 1.
        else:
            weight[1:, 1:] = self.weight
            weight[0, 0] = 1.
        return weight

    def padded_bias(self):
        """
        Return the bias with a padded value
        allowing a feature to be unchanged by the linear layer
        
        :return: a torch vector of size (self.out_features + 1)
        """
        bias = torch.zeros(self.out_features + 1).to(self.weight.device)
        if self.bias is None:
            return bias
        if self.last_t:
            bias[:-1] = self.bias
        else:
            bias[1:] = self.bias
        return bias

    # Override
    def forward(self, X):
        return F.linear(X, self.padded_weight(), self.padded_bias())
    
    
class CatLinear(nn.Linear):
    def __init__(self, in_features, out_features, num_group, bias=True, last_t=False):
        """
        Let all input features not to interact with input features in other groups
        and the output features are the concatenation of all groups' outputs
        
        :param in_features: number of input features in each group
        :param out_features: number of output features (dim will be multiplied by in_features)
        :param num_group: number of groups
        :param bias: whether to use bias
        :param last_t: whether the unchanged input feature is the first or the last
        """
        super().__init__(in_features, out_features, bias)
        self.weights = nn.ParameterList([nn.Parameter(deepcopy(self.weight)) 
                                         for _ in range(num_group)])
        self.weight = None
        if bias:
            self.biases = nn.ParameterList([nn.Parameter(deepcopy(self.bias)) 
                                            for _ in range(num_group)])
            self.bias = None  # Nullify the unused weight and bias
        self.num_group = num_group
        self.last_t = last_t

    def cat_weight(self):
        """
        Return the weight matrix as a concatenation of all groups' weight matrices
        
        :return: a torch matrix of size (out_features * num_group, self.in_features * num_group)
        """
        weight = torch.zeros(self.out_features * self.num_group, self.in_features * self.num_group)
        weight = weight.to(self.weights[0].device)
        for i in range(self.num_group):
            weight[self.out_features * i:self.out_features * (i + 1), 
                   self.in_features * i:self.in_features * (i + 1)] = self.weights[i]
        return weight

    def cat_bias(self):
        """
        Return the bias vector as a concatenation of all groups' bias vector
        
        :return: a torch vector of size (out_features * num_group)
        """
        if self.bias is None:
            return None
        bias = torch.zeros(self.out_features * self.num_group).to(self.weights[0].device)
        for i in range(self.num_group):
            bias[self.out_features * i:self.out_features * (i + 1)] = self.biases[i]
        return bias

    # Override
    def forward(self, X):
        return F.linear(X, self.cat_weight(), self.cat_bias())


class NonNegLinear(nn.Linear):
    def __init__(self, *args):
        super().__init__(*args)

    # Override
    def forward(self, X):
        return F.linear(X, F.relu(self.weight), self.bias)


class Cuboid(nn.Module):

    def __init__(self, L=None, M=None):
        super().__init__()
        if L is not None:
            self.L = L
        else:
            self.L = MixSequential(nn.Linear(3, 128),
                                   ReQU(),
                                   nn.Linear(128, 128),
                                   ReQU(),
                                   nn.Linear(128, 128),
                                   ReQU(),
                                   nn.Linear(128, 1),
                                   Neg()
                                   )
        if M is not None:
            self.M = M
        else:
            self.M = MixSequential(nn.Linear(3, 128),
                                   ReQU(),
                                   nn.Linear(128, 128),
                                   ReQU(),
                                   nn.Linear(128, 128),
                                   ReQU(),
                                   nn.Linear(128, 1)
                                   )

    @typechecked
    def cuboid(self,
               xa: TensorType["batch"],
               xb: TensorType["batch"],
               ya: TensorType["batch"],
               yb: TensorType["batch"],
               za: TensorType["batch"],
               zb: TensorType["batch"]) -> TensorType[24, 3, "batch"]:
        """
        Helper function for triple integrate the triple derivative.
        All vertices (24 each) of cuboid integration areas are prepared for batch processing

        :param xa: (batch_size,), lower x bound for integration
        :param xb: (batch_size,), upper x bound for integration
        :param ya: (batch_size,), lower y bound for integration
        :param yb: (batch_size,), upper y bound for integration
        :param za: (batch_size,), lower time bound for integration
        :param zb: (batch_size,), upper time bound for integration
        :return: (24, 3, batch_size), the integration endpoints
        """
        return nested_stack([[xa, ya, zb],
                             [xa, yb, zb],
                             [xb, ya, zb],
                             [xb, yb, zb],
                             [xa, ya, za],
                             [xa, yb, za],
                             [xb, ya, za],
                             [xb, yb, za],
                             [xa, yb, za],
                             [xa, yb, zb],
                             [xb, yb, za],
                             [xb, yb, zb],
                             [xa, ya, za],
                             [xa, ya, zb],
                             [xb, ya, za],
                             [xb, ya, zb],
                             [xb, ya, za],
                             [xb, ya, zb],
                             [xb, yb, za],
                             [xb, yb, zb],
                             [xa, ya, za],
                             [xa, ya, zb],
                             [xa, yb, za],
                             [xa, yb, zb]]).to(next(self.parameters()).device)

    @typechecked
    def rectangle(self,
                  xa: TensorType["batch"],
                  xb: TensorType["batch"],
                  ya: TensorType["batch"],
                  yb: TensorType["batch"],
                  z: TensorType["batch"]) -> TensorType[4, 3, "batch"]:
        """
        Helper function for double integrate the triple derivative at a certain time.
        All endpoints (4 each) of rectangle integration areas are prepared for batch processing

        :param xa: (batch_size,), lower x bound for integration
        :param xb: (batch_size,), upper x bound for integration
        :param ya: (batch_size,), lower y bound for integration
        :param yb: (batch_size,), upper y bound for integration
        :param z:  (batch_size,), the fixed z (time) as a constant during integration
        :return: (4, 3, batch_size), the integration endpoints
        """
        xa = xa * torch.ones_like(z)
        xb = xb * torch.ones_like(z)
        ya = ya * torch.ones_like(z)
        yb = yb * torch.ones_like(z)

        return nested_stack([[xa, ya, z],
                             [xa, yb, z],
                             [xb, ya, z],
                             [xb, yb, z]]).to(next(self.parameters()).device)

    @typechecked
    def forward(self, st: TensorType["batch", 3]) -> TensorType["batch", 1]:
        """
        A closed evaluation of the third derivative (λ_st)

        :param st: (batch_size, 3), time is the last dimension
        :return: (batch_size, 1), the third derivative
        """
        return self.M.dnforward(st, [0, 1, 2]) * torch.tensor(3.) - \
               self.L.dnforward(st, [0, 1, 2]) * torch.tensor(3.)

    @typechecked
    def lamb_t_stpp(self,
                    s: TensorType["batch", 2],
                    t: TensorType["batch", 1]) -> TensorType["batch", 1]:
        """
        Shorthand for the closed evaluation of the first derivative (λ_t)
        over the space [0,1]x[0,1], assuming the intensity is centered at s (origin)

        :param s: (batch_size, 2)
        :param t: (batch_size, 1), the time
        :return:  (batch_size, 1), the first derivative
        """
        x = s[:, 0]
        y = s[:, 1]
        t = t.squeeze(-1)  # all squeeze to (batch_size,)
        xa = -x
        xb = 1. - x
        ya = -y
        yb = 1. - y
        return self.lamb_t(xa, xb, ya, yb, t)

    @typechecked
    def lamb_t(self,
               xa: TensorType["batch"],
               xb: TensorType["batch"],
               ya: TensorType["batch"],
               yb: TensorType["batch"],
               t:  TensorType["batch"]) -> TensorType["batch", 1]:
        """
        A closed evaluation of intensity over the space [xa,xb]x[ya,yb] at time t

        :param xa: (batch_size,), lower x bound
        :param xb: (batch_size,), upper x bound
        :param ya: (batch_size,), lower y bound
        :param yb: (batch_size,), upper y bound
        :param t:  (batch_size,), the time
        :return:   (batch_size, 1), the first derivative
        """
        # [4, 3, N] -> [4, N, 3] -> [4*N, 3]
        inp = self.rectangle(xa, xb, ya, yb, t).permute(0, 2, 1).contiguous().view(-1, 3)
        m = self.M.dnforward(inp, [2]) * 3
        m = m.view(4, -1, 1)
        l = self.L.dnforward(inp, [2]) * 3
        l = l.view(4, -1, 1)
        return l[2] - l[0] + m[3] - m[2] + l[1] - l[3] + m[0] - m[1]

    @typechecked
    def int_lamb_stpp(self,
                      s:  TensorType["batch", 2],
                      ta: TensorType["batch", 1],
                      tb: TensorType["batch", 1]) -> TensorType["batch", 1]:
        """
        Shorthand for the closed evaluation of the integral over the cuboid
        over the space [0,1]×[0,1] (related to s) and time [ta, tb]

        :param s:  (batch_size, 2), the origin locations
        :param ta: (batch_size, 1), starting time for integration
        :param tb: (batch_size, 1), ending time for integration
        :return:   (batch_size, 1), the first derivative
        """
        x = s[:, 0]
        y = s[:, 1]
        xa = -x
        xb = 1. - x
        ya = -y
        yb = 1. - y
        ta = ta.squeeze(-1)
        tb = tb.squeeze(-1)  # all squeeze to (batch_size,)
        return self.int_lamb(xa, xb, ya, yb, ta, tb)

    @typechecked
    def int_lamb(self,
                 xa: TensorType["batch"],
                 xb: TensorType["batch"],
                 ya: TensorType["batch"],
                 yb: TensorType["batch"],
                 ta: TensorType["batch"],
                 tb: TensorType["batch"]) -> TensorType["batch", 1]:
        """
        A closed evaluation of the integral over the cuboid over the space [xa,xb]×[ya,yb]
         and time [ta, tb]

        :param xa: (batch_size,), lower x bound
        :param xb: (batch_size,), upper x bound
        :param ya: (batch_size,), lower y bound
        :param yb: (batch_size,), upper y bound
        :param ta: (batch_size,), starting time for integration
        :param tb: (batch_size,), ending time for integration
        :return:   (batch_size, 1), the first derivative
        """
        # [24, 3, N] -> [3, 2, 4, 3, N]
        endpoints = self.cuboid(xa, xb, ya, yb, ta, tb).view(3, 2, 4, 3, -1).transpose(-1, -2)

        m = self.M(endpoints).sum(0)
        m = m[0] - m[1]
        l = self.L(endpoints).sum(0)
        l = l[0] - l[1]

        return l[2] - l[0] + m[3] - m[2] + l[1] - l[3] + m[0] - m[1]

    def project(self) -> None:
        """
        Clamp all layers' weights to non-negative.
        """
        self.M.project()
        self.L.project()
