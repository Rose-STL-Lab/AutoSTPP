from itertools import combinations
from typing import List, Generator

import numpy as np
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from torch import nn, autograd
from torch.nn import functional as F

from src.utils import nested_stack

patch_typeguard()
batch = None  # Suppress PyCharm type warning


class ReQU(nn.Module):
    def __init__(self):
        super().__init__()  # init the base class

    @staticmethod
    def forward(x):
        return torch.where(x > 0., 0.125 * x ** 2 + 0.5 * x + np.log(2.),
                           torch.log(torch.exp(x) + torch.tensor(1.)))


class ReQUFlip(nn.Module):
    def __init__(self):
        super().__init__()  # init the base class

    @staticmethod
    def forward(x):
        return torch.where(-x > 0., 0.125 * x ** 2 - 0.5 * x + np.log(2.),
                           torch.log(torch.exp(-x) + torch.tensor(1.)))


class MultSequential(nn.Sequential):
    """
    The optimized integral-net
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.x = None
        self.f = []
        self.dnf = {}

    acDict = {  # activation derivative
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
                 lambda f, _: f] * 25,
        'ReQU': [lambda _, x: torch.where(x > 0., 0.25 * x + 0.5, 1. / (1. + torch.exp(-x))),
                 lambda _, x: torch.where(x > 0., 0.25 * torch.ones_like(x), torch.exp(-x) / (1 + torch.exp(-x)) ** 2),
                 lambda _, x: torch.where(x > 0., torch.zeros_like(x),
                                          - torch.exp(x) * (torch.exp(x) - 1.) / (1. + torch.exp(x)) ** 3)]
    }

    acDict['ReQUFlip'] = [lambda _, x: ac(_, -x) for ac in acDict['ReQU']]  # ReQUFlip is ReQU flipped about y-axis

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
            if tp == 'Linear':
                pd.append(pd[-1] @ module.weight.T)
            elif tp == 'NonNegLinear':
                pd.append(pd[-1] @ F.relu(module.weight).T)
            elif tp == 'PadLinear':
                pd.append(pd[-1] @ module.padded_weight().T)
            elif tp in self.acDict:
                if N == 1:
                    term_sum = self.acDict[tp][0](self.f[i + 1], self.f[i]) * pd[-1]  # Base case
                else:
                    term_sum = 0.
                    # ac(n)   * df/dx1 * df/dx2 * df/dx3 ... * df/dxn
                    # ...
                    # ac''    * (d2f/dx1dx2.. * df/dxn + ... )
                    # ac'     * dnf/dx1dx2dx3...dxn
                    # ways to partition x1...xn to k sets time ac(k)
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
                        assert order < len(self.acDict[tp]), "Activation high-order derivative not implemented"
                        term_sum += term * self.acDict[tp][order](self.f[i + 1], self.f[i])
                pd.append(term_sum)
            else:
                raise NotImplementedError

        return pd[-1]


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


class PadLinear(nn.Linear):
    def __init__(self, *args):
        super().__init__(*args)

    def padded_weight(self):
        weight = torch.eye(self.out_features + 1, self.in_features + 1).to(self.weight.device)
        weight[1:, 1:] = self.weight
        return weight

    def padded_bias(self):
        bias = torch.zeros(self.out_features + 1).to(self.bias.device)
        bias[1:] = self.bias
        return bias

    # Override
    def forward(self, X):
        return F.linear(X, self.padded_weight(), self.padded_bias())


class NonNegLinear(nn.Linear):
    def __init__(self, *args):
        super().__init__(*args)

    # Override
    def forward(self, X):
        return F.linear(X, F.relu(self.weight), self.bias)


class Cuboid(nn.Module):

    def __init__(self):
        super().__init__()
        self.L = MixSequential(nn.Linear(3, 128),
                               ReQUFlip(),
                               nn.Linear(128, 128),
                               ReQUFlip(),
                               nn.Linear(128, 128),
                               ReQUFlip(),
                               nn.Linear(128, 1)
                               )
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
                  z:  TensorType["batch"]) -> TensorType[4, 3, "batch"]:
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
    def lamb_t(self, s: TensorType["batch", 2],
                     t: TensorType["batch", 1]) -> TensorType["batch", 1]:
        """
        A closed evaluation of the first derivative (λ_t) over the space [0,1]x[0,1], assuming the intensity
        is centered at s (origin)

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

        m = self.M.dnforward(self.rectangle(xa, xb, ya, yb, t).transpose(-1, -2), [2]) * 3
        l = self.L.dnforward(self.rectangle(xa, xb, ya, yb, t).transpose(-1, -2), [2]) * 3
        return l[2] - l[0] + m[3] - m[2] + l[1] - l[3] + m[0] - m[1]

    @typechecked
    def int_lamb(self, s:  TensorType["batch", 2],
                       ta: TensorType["batch", 1],
                       tb: TensorType["batch", 1]) -> TensorType["batch", 1]:
        """
        A closed evaluation of the integral over the cuboid over the space [0,1]x[0,1] (related to s)
         and time [ta, tb]

        :param s:  (batch_size, 2), the origin locations
        :param ta: (batch_size, 1), starting time for integration
        :param tb: (batch_size, 1), ending time for integration
        :return:   (batch_size, 1), the first derivative
        """
        x = s[:, 0]
        y = s[:, 1]
        ta = ta.squeeze(-1)
        tb = tb.squeeze(-1)  # all squeeze to (batch_size,)
        xa = -x
        xb = 1. - x
        ya = -y
        yb = 1. - y

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
        with torch.no_grad():
            self.M.layers[0].weight.clamp_(min=0.0, max=None)
            self.M.layers[2].weight.clamp_(min=0.0, max=None)
            self.M.layers[4].weight.clamp_(min=0.0, max=None)
            self.M.layers[6].weight.clamp_(min=0.0, max=None)
            self.L.layers[0].weight.clamp_(min=None, max=0.0)
            self.L.layers[2].weight.clamp_(min=None, max=0.0)
            self.L.layers[4].weight.clamp_(min=None, max=0.0)
            self.L.layers[6].weight.clamp_(min=None, max=0.0)
