from itertools import combinations

import numpy as np
import torch
from src.experiment.autoint3d import MLPnD
from torch import nn, autograd
from torch.nn import functional as F
import logging

from src.utils import nested_stack

logger = logging.getLogger(__name__)


class ReQU(nn.Module):
    def __init__(self):
        super().__init__()  # init the base class

    def forward(self, x):
        return torch.where(x > 0., 0.125 * x ** 2 + 0.5 * x + np.log(2.), torch.log(torch.exp(x) + 1.))


class ReQUflip(nn.Module):
    def __init__(self):
        super().__init__()  # init the base class

    def forward(self, x):
        return torch.where(-x > 0., 0.125 * x ** 2 - 0.5 * x + np.log(2.), torch.log(torch.exp(-x) + 1.))


class MultSequential(nn.Sequential):

    def __init__(self, *args):
        super().__init__(*args)
        # self.x, self.f, self.df, self.d2f, self.d3f = None, [], {}, {}, {}
        self.x = None
        self.f = []
        self.dnf = {}

    acDict = {  # activation derivative
        'Tanh': [lambda f, _: 1 - f ** 2,
                 lambda f, _: -2 * f * (1 - f ** 2),
                 lambda f, _: 8 * f ** 2 - 6 * f ** 4 - 2,
                 lambda f, _: 24 * f ** 5 - 40 * f ** 3 + 16 * f],
        'Sigmoid': [lambda f, _: f * (1 - f),
                    lambda f, _: (f - 2 * f ** 2) * (1 - f)],
        'Sine': [lambda _, x: torch.cos(x),
                 lambda f, _: -f,
                 lambda _, x: -torch.cos(x),
                 lambda f, _: f] * 25,
        'ReQU': [lambda _, x: torch.where(x > 0., 0.25 * x + 0.5, 1. / (1. + torch.exp(-x))),
                 lambda _, x: torch.where(x > 0., 0.25 * torch.ones_like(x), torch.exp(-x) / (1 + torch.exp(-x)) ** 2),
                 lambda _, x: torch.where(x > 0., torch.zeros_like(x),
                                          - torch.exp(x) * (torch.exp(x) - 1.) / (1. + torch.exp(x)) ** 3)]
    }

    acDict['ReQUflip'] = [lambda _, x: ac(_, -x) for ac in acDict['ReQU']]

    @staticmethod
    def hash(dims):
        """
        Hash a list of deriving dimensions to str dict key
        :param dims: an unordered list of integers
        :return: str of ordered dims (such that [1,2] and [2,1] refer to the same derivative)
        """
        return str(sorted(dims))  # Hash the dimension

    @staticmethod
    def partition(ns, m):
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

    def reset(self):
        """
        Reinitialize all buffer, when x changes
        """
        # self.f, self.df, self.d2f, self.d3f = [], {}, {}, {}
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

    # @DeprecationWarning
    # def dforward(self, x, dim):
    #     """
    #     Start recursion for a partial derivative
    #
    #     :param x: (batch, dim)
    #     :param dim: the deriving dimension
    #     """
    #     assert type(dim) == int
    #     key = self.hash([dim])  # Hash the dimension
    #
    #     if x is self.x:
    #         if key in self.df:
    #             return self.df[key][-1]
    #     else:
    #         self.x = x
    #         self.reset()
    #
    #     if len(self.f) == 0:
    #         _ = self.forward(x)  # Prepare f values
    #
    #     base = torch.zeros_like(x)
    #     base[..., dim] = 1
    #     self.df[key] = pd = [base, ]
    #
    #     # Perform chain rule: df(g(x))/dx = f'(g(x)) g'(x)
    #     for i, module in enumerate(self):
    #         tp = type(module).__name__
    #         if tp == 'Linear':
    #             pd.append(pd[-1] @ module.weight.T)
    #         elif tp == 'NonNegLinear':
    #             pd.append(pd[-1] @ F.relu(module.weight).T)
    #         elif tp == 'PadLinear':
    #             pd.append(pd[-1] @ module.padded_weight().T)
    #         elif tp in self.acDict:
    #             pd.append(self.acDict[tp][0](self.f[i + 1], self.f[i]) * pd[-1])
    #         else:
    #             raise NotImplementedError
    #
    #     return pd[-1]
    #
    # @DeprecationWarning
    # def d2forward(self, x, dims):
    #     """
    #     Start recursion for d2f / dxi dxj
    #
    #     :param x: (batch, dim)
    #     :param dims: list [i, j], can be repetitive
    #     """
    #     assert type(dims) == list and len(dims) == 2
    #     key = self.hash(dims)
    #
    #     if x is self.x:
    #         if key in self.d2f:
    #             return self.d2f[key][-1]
    #     else:
    #         self.x = x
    #         self.reset()
    #
    #     for dim in dims:
    #         if self.hash([dim]) not in self.df:
    #             _ = self.dforward(x, dim)  # Prepare f, df values
    #
    #     self.d2f[key] = pd2 = [torch.zeros_like(x), ]
    #
    #     # ac''(f) * df/dx * df/dy + ac'(f) * d2f/dxdy
    #     for i, module in enumerate(self):
    #         tp = type(module).__name__
    #         if tp == 'Linear':
    #             pd2.append(pd2[-1] @ module.weight.T)
    #         elif tp in self.acDict:
    #             term1 = self.acDict[tp][1](self.f[i + 1], self.f[i])
    #             for d in dims:
    #                 term1 *= self.df[self.hash([d])][i]
    #             term2 = self.acDict[tp][0](self.f[i + 1], self.f[i]) * pd2[-1]
    #             pd2.append(term1 + term2)
    #         else:
    #             raise NotImplementedError
    #
    #     return pd2[-1]
    #
    # def d3forward(self, x, dims):
    #     """
    #     Start recursion for d3f / dxi dxj dxk
    #
    #     :param x: (batch, dim)
    #     :param dims: list [i, j, k], can be repetitive
    #     """
    #     assert type(dims) == list and len(dims) == 3
    #     key = self.hash(dims)
    #
    #     if x is self.x:
    #         if key in self.d3f:
    #             return self.d3f[key][-1]
    #     else:
    #         self.x = x
    #         self.reset()
    #
    #     for dim in combinations(dims, 2):
    #         if self.hash(dim) not in self.d2f:
    #             _ = self.d2forward(x, list(dim))  # Prepare f, df, d2f values
    #
    #     self.d3f[key] = pd3 = [torch.zeros_like(x), ]
    #
    #     # ac''' *  df/dx * df/dy * df/dz
    #     # ac''  * (d2f/dxdy * df/dz + d2f/dxdz * df/dy + d2f/dydz * df/dx)
    #     # ac'   *  d3f/dxdydz
    #     for i, module in enumerate(self):
    #         tp = type(module).__name__
    #         if tp == 'Linear':
    #             pd3.append(pd3[-1] @ module.weight.T)
    #         elif tp in self.acDict:
    #             term1 = self.acDict[tp][2](self.f[i + 1], self.f[i])
    #             for d in dims:
    #                 term1 *= self.df[self.hash([d])][i]
    #
    #             d1, d2, d3 = dims
    #             term2 = self.d2f[self.hash([d1, d2])][i] * self.df[self.hash([d3])][i]
    #             term2 += self.d2f[self.hash([d1, d3])][i] * self.df[self.hash([d2])][i]
    #             term2 += self.d2f[self.hash([d2, d3])][i] * self.df[self.hash([d1])][i]
    #             term2 *= self.acDict[tp][1](self.f[i + 1], self.f[i])
    #
    #             term3 = self.acDict[tp][0](self.f[i + 1], self.f[i]) * pd3[-1]
    #             pd3.append(term1 + term2 + term3)
    #         else:
    #             raise NotImplementedError
    #
    #     return pd3[-1]

    def dnforward(self, x, dims):
        """
        Start recursion for d(n)x / dx1 dx2 ... dxn

        :param x: (batch, dim)
        :param dims: unordered list [x1, x2, ..., xn], can be repetitive
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
            base[..., dims[0]] = 1
            self.dnf[key] = pd = [base, ]
        else:
            for dim in combinations(dims, N - 1):
                if self.hash(dim) not in self.dnf:
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
                        term_sum += term * self.acDict[tp][order](self.f[i + 1], self.f[i])
                pd.append(term_sum)
            else:
                raise NotImplementedError

        return pd[-1]


class BaselineSequential(nn.Sequential):

    def __init__(self, *args):
        super().__init__(*args)

    def dnforward(self, x, dims):
        """
        Start recursion for a nth order partial derivative

        :param x: (batch, dim)
        :param dims: the list of deriving dimension, with length n
        """
        if x.is_leaf:
            x.requires_grad = True

        if len(dims) == 1:
            df = self  # Base case
        else:
            df = lambda x_: self.dnforward(x_, dims[:-1])  # Derivative with one fewer order

        d2f = df(x)
        return autograd.grad(d2f, x, torch.ones_like(d2f), create_graph=True)[0][:, dims[-1:]]


class MixSequential(nn.Module):

    def __init__(self, *args, thres=3):
        super().__init__()
        self.layers = MultSequential(*args)
        self.thres = thres

    def load_state_dict(self, state_dict, strict=True):
        return self.layers.load_state_dict(state_dict, strict)

    def forward(self, x):
        return self.layers.forward(x)

    def dnforward(self, x, dims):
        """
        Start recursion for a nth order partial derivative

        :param x: (batch, dim)
        :param dims: the list of deriving dimension, with length n
        """
        if x.is_leaf:
            x.requires_grad = True

        if len(dims) <= self.thres:
            return self.layers.dnforward(x, dims[:self.thres])  # Base case
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
        self.L = MLPnD(MixSequential, 3, ReQUflip())
        self.M = MLPnD(MixSequential, 3, ReQU())

    def cuboid(self, xa, xb, ya, yb, za, zb):
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

    def rectangle(self, xa, xb, ya, yb, z):
        xa = xa * torch.ones_like(z)
        xb = xb * torch.ones_like(z)
        ya = ya * torch.ones_like(z)
        yb = yb * torch.ones_like(z)

        return nested_stack([[xa, ya, z],
                             [xa, yb, z],
                             [xb, ya, z],
                             [xb, yb, z]]).to(next(self.parameters()).device)

    def forward(self, st):
        """
        A closed evaluation of the third derivative (λ_st)

        :param st: (batch_size, 3), time is the last dimension
        :return (batch_size) the third derivative
        """
        assert len(st.shape) == 2 and st.shape[-1] == 3
        return self.M.dnforward(st, [0, 1, 2]) * 3 - self.L.dnforward(st, [0, 1, 2]) * 3

    def lamb_t(self, s, t):
        """
        A closed evaluation of the first derivative (λ_t) over the space [0,1]x[0,1], assuming the intensity
        is centered at s (origin)

        :param s: (batch_size, 2)
        :param t: (batch_size), the time
        :return: (batch_size) the first derivative
        """
        assert len(s.shape) == 2 and s.shape[1] == 2
        assert len(t.shape) == 1 and t.shape[0] == s.shape[0]
        x = s[:, 0]
        y = s[:, 1]
        xa = -x
        xb = 1. - x
        ya = -y
        yb = 1. - y

        m = self.M.dnforward(self.rectangle(xa, xb, ya, yb, t).transpose(-1, -2), [2]) * 3
        l = self.L.dnforward(self.rectangle(xa, xb, ya, yb, t).transpose(-1, -2), [2]) * 3
        return l[2] - l[0] + m[3] - m[2] + l[1] - l[3] + m[0] - m[1]

    def int_lamb(self, s, ta, tb):
        """
        A closed evaluation of the integral over the cuboid over the space [0,1]x[0,1] (related to s)
         and time [ta, tb]

        :param s:  (batch_size, 2), the origin locations
        :param ta: (batch_size), the starting time
        :param tb: (batch_size), the ending time
        :return: (batch_size) the first derivative
        """
        assert len(ta.shape) == 1 and len(tb.shape) == 1 and len(tb) == len(ta)

        x = s[:, 0]
        y = s[:, 1]
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

    def project(self):
        """
        Clamp all weights to non-negative
        """
        with torch.no_grad():
            self.M.layers.layers[0].weight.clamp_(0.0)
            self.M.layers.layers[2].weight.clamp_(0.0)
            self.M.layers.layers[4].weight.clamp_(0.0)
            self.M.layers.layers[6].weight.clamp_(0.0)
            self.L.layers.layers[0].weight.clamp_(min=None, max=0.0)
            self.L.layers.layers[2].weight.clamp_(min=None, max=0.0)
            self.L.layers.layers[4].weight.clamp_(min=None, max=0.0)
            self.L.layers.layers[6].weight.clamp_(min=None, max=0.0)
