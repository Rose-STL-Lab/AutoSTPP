from src.data.data import TPPWrapper

import numpy as np
import torch
import logging.config
from torch import nn
from tqdm import tqdm
from tqdm.contrib import tzip
from scipy.optimize import brentq
from utils import get_device, gen_fn

import matplotlib.pyplot as plt

eps = 1e-10  # A negligible positive number
np.random.seed(0)


def relu(x):
    return x * (x > 0)


def plus(x):
    """
    relu with small positive shift
    """
    return x * (x > eps) + eps * (x < eps)


def lamb_func(t, his_t):
    mu1 = .4
    mu2 = 1e-5

    zeta = 1.
    c0 = -1.
    c1 = 1.
    c2 = .2

    delta_t = t - his_t[his_t < t]
    lamb_t = relu(mu1 + np.sum((c2 * delta_t ** 2 + c1 * delta_t + c0) * np.exp(-zeta * delta_t))) + mu2
    L = 2 / lamb_t

    count = delta_t[0::2]
    count = sum(count[count < 10])

    M = mu1 + mu2 + count * 0.677

    return lamb_t, L, M


def delta(a, b, c, d, e):
    """
    Calculate the discriminants of quartic polynomial
    """
    delta0 = c**2-3*b*d+12*a*e
    delta1 = 2*c**3-9*b*c*d+27*b**2*e+27*a*d**2-72*a*c*e
    return delta0, delta1


def cbrt(x):
    """
    Calculate cubic root of a torch tensor
    :param x: torch.tensor
    :return: cubic root of x, regardless of sign
    """
    return x.sign() * x.abs().pow(1/3.)


def quartic(a, b, c, d, e):
    """
    Find the positive root of quartic function
    """
    if type(a) is not torch.Tensor:
        a = torch.tensor(a)
    if type(b) is not torch.Tensor:
        b = torch.tensor(b)
    if type(c) is not torch.Tensor:
        c = torch.tensor(c)
    if type(d) is not torch.Tensor:
        d = torch.tensor(d)
    if type(e) is not torch.Tensor:
        e = torch.tensor(e)

    p = (8*a*c-3*b**2)/(8*a**2)
    q = (b**3-4*a*b*c+8*a**2*d)/(8*a**3)
    d0, d1 = delta(a, b, c, d, e)
    temp1 = d1**2 - 4*d0**3
    Q = cbrt((plus(temp1)**0.5 + d1) / 2)
    temp2 = -2/3*p + (Q + d0/Q)/(3*a)
    S = plus(temp2)**0.5 / 2
    temp3 = -4*S**2 - 2*p + q/S
    # temp4 = -4*S**2 - 2*p - q/S
    x1 = -b/(4*a) - S + plus(temp3)**0.5 / 2
    # x2 = -b/(4*a) - S - plus(temp3)**0.5 / 2
    # x3 = -b/(4*a) + S + plus(temp4)**0.5 / 2
    # x4 = -b/(4*a) + S - plus(temp4)**0.5 / 2
    return x1 * (S > 1e-3) * (x1 > 0.)


def nll_close(his_t, t_start, t_end, mu, zeta, c0, c1, c2):
    """
    Calculate the negative log-likelihood of correcting Hawkes intensity by closed solution. \n
    Intensity function: (c0 + c1 x + c2 x^2) exp(-zeta x) + mu

    :param his_t: (seq_len, ) the temporal sequence
    :param t_start: the start time of observation, <= his_t[0]
    :param t_end: the end time of observation, >= his_t[-1]
    :param mu: scalar torch parameter
    :param zeta: scalar torch parameter
    :param c0: scalar torch parameter
    :param c1: scalar torch parameter
    :param c2: scalar torch parameter
    :return: scalar NLL
    """
    # Calculate outer subtraction
    delta_t = torch.tril(his_t.reshape(-1, 1) - his_t.reshape(1, -1), -1)

    # Calculate lamb before all events
    delta_t = torch.clamp(delta_t, max=50.0)
    lamb = torch.sum(torch.tril((c2 * delta_t ** 2 + c1 * delta_t + c0)
                                * torch.exp(-zeta * delta_t), -1), -1)
    lamb = plus(mu + lamb)

    # - ∑ log λ(t_i)
    term_1 = - torch.sum(torch.log(lamb))

    # + ∫ λ(t) dt = μT - α/β ∑ [exp(-β(T-t_i)) - 1] (0~T)
    term_2 = mu * (t_end - t_start)

    # Calculate the coefficient of the polynomial after each event
    # Pick [i, i] (no tril offset) to include the intensity jump
    a_ = torch.sum(c2 * torch.tril(torch.exp(-zeta * delta_t)), -1)
    b_ = torch.sum((c1 + c2 * 2 * delta_t) * torch.tril(torch.exp(-zeta * delta_t)), -1)
    c_ = torch.sum((c0 + c1 * delta_t + c2 * delta_t ** 2) * torch.tril(torch.exp(-zeta * delta_t)), -1)

    # Approximate mu + (at**2 + bt + c) exp(-z*t) using a quartic function at**4 + bt**3 + ct**2 + dt + e
    a = a_ / 2.
    b = -a_ + b_ / 2.
    c = a_ - b_ + c_ / 2.
    d = b_ - c_
    e = c_ + mu

    # Solve the positive real roots of the quartic function
    # If not exist, then return zero (no negative intervals)
    # roots are the ends of negative intervals
    roots = quartic(a, b, c, d, e)
    t_end = torch.ones(1).to(his_t) * t_end
    term_2 -= torch.sum(roots) * mu  # remove the negative parts of the base intensity

    # Integrate from all roots to next event
    roots = roots + his_t
    roots = torch.cat([torch.zeros(1).to(roots), roots])
    neg_start = torch.cat([his_t, t_end])

    int_start = neg_start.reshape(-1, 1) - his_t.reshape(1, -1)
    int_start = relu(int_start)

    int_end = roots.reshape(-1, 1) - his_t.reshape(1, -1)
    int_end = relu(int_end)

    term_3 = torch.sum(torch.exp(-zeta * int_start) * (-c2 * (zeta * int_start * (zeta * int_start + 2) + 2) -
                                                       zeta * (c1 * zeta * int_start + c1 + c0 * zeta)) -
                       torch.exp(-zeta * int_end) * (-c2 * (zeta * int_end * (zeta * int_end + 2) + 2) -
                                                     zeta * (c1 * zeta * int_end + c1 + c0 * zeta))) / zeta ** 3

    return term_1 + term_2 + term_3


def nll_num_root(his_t, t_start, t_end, mu, zeta, c0, c1, c2):
    """
    Calculate the negative log-likelihood of correcting Hawkes intensity by root-finding. \n
    Intensity function: (c0 + c1 x + c2 x^2) exp(-zeta x) + mu

    :param his_t: (seq_len, ) the temporal sequence
    :param t_start: the start time of observation, <= his_t[0]
    :param t_end: the end time of observation, >= his_t[-1]
    :param mu: scalar torch parameter
    :param zeta: scalar torch parameter
    :param c0: scalar torch parameter
    :param c1: scalar torch parameter
    :param c2: scalar torch parameter
    :return: scalar NLL
    """
    # Calculate outer subtraction
    delta_t = torch.tril(his_t.reshape(-1, 1) - his_t.reshape(1, -1), -1)

    # Calculate lamb before all events
    lamb = torch.sum(torch.tril((c2 * delta_t ** 2 + c1 * delta_t + c0)
                                * torch.exp(-zeta * delta_t), -1), -1)
    lamb = plus(mu + lamb)

    # - ∑ log λ(t_i)
    term_1 = - torch.sum(torch.log(lamb))

    # + ∫ λ(t) dt = μT - α/β ∑ [exp(-β(T-t_i)) - 1] (0~T)
    term_2 = mu * (t_end - t_start)

    # Poor man's approach to find all positive interval
    def lf(t, ht):
        dt = t - ht[ht < t]
        return (mu + torch.sum((c2 * dt ** 2 + c1 * dt + c0) * torch.exp(-zeta * dt))).item()

    ts = torch.cat([torch.zeros(1).to(his_t), his_t])
    N = 3
    dts = (ts[1:] - ts[:-1]).reshape(-1, 1) / N * \
          torch.cat([-torch.ones(1) * eps, torch.arange(N - 1) + eps]).reshape(1, -1).to(ts)
    ts = (ts[:-1].view(-1, 1) + dts).view(-1)

    lambs = np.array([lf(t, his_t) for t in tqdm(ts)])
    lambs[:-1] *= lambs[1:]
    mask = (lambs < 0)[:-1]

    ts1 = ts[:-1][mask]
    ts2 = ts[1:][mask]

    neg_start = []
    neg_end = [t_start]
    flag = True
    for t1, t2 in tzip(ts1, ts2):
        res = brentq(lambda t : lf(t, his_t), t1, t2, xtol=eps)
        if flag:   # positive -> negative
            neg_start.append(res)
        else:   # negative -> positive
            neg_end.append(res)
        flag = not flag

    if flag:  # end with positive value
        neg_start.append(t_end)

    neg_start = torch.tensor(neg_start).to(device)
    neg_end = torch.tensor(neg_end).to(device)

    term_2 -= torch.sum(neg_end[1:] - neg_start[:-1]) * mu
    if not flag:
        term_2 -= (t_end - neg_start[-1]) * mu

    int_start = neg_start.reshape(-1, 1) - his_t.reshape(1, -1)
    int_start = relu(int_start)

    int_end = neg_end.reshape(-1, 1) - his_t.reshape(1, -1)
    int_end = relu(int_end)

    term_3 = torch.sum(torch.exp(-zeta * int_start) * (-c2 * (zeta * int_start * (zeta * int_start + 2) + 2) -
                                                       zeta * (c1 * zeta * int_start + c1 + c0 * zeta)) -
                       torch.exp(-zeta * int_end) * (-c2 * (zeta * int_end * (zeta * int_end + 2) + 2) -
                                                     zeta * (c1 * zeta * int_end + c1 + c0 * zeta))) / zeta ** 3

    return term_1 + term_2 + term_3


class CorrectingHawkes(torch.nn.Module):
    def __init__(self, mu, zeta, c0, c1, c2):
        super(CorrectingHawkes, self).__init__()

        self.zeta = nn.Parameter(torch.ones([]) * np.log(zeta))
        self.mu = nn.Parameter(torch.ones([]) * np.log(mu))

        self.c0 = nn.Parameter(torch.ones([]) * c0)
        self.c1 = nn.Parameter(torch.ones([]) * c1)
        self.c2 = nn.Parameter(torch.ones([]) * c2)

    def get_param(self):
        mu = torch.exp(self.mu)
        zeta = torch.exp(self.zeta)

        return mu, zeta, self.c0, self.c1, self.c2

    def log_param(self):
        mu, zeta, c0, c1, c2 = self.get_param()
        logger.info('mu    : {:6.4f}'.format(mu.item()))
        logger.info('zeta  : {:6.4f}'.format(zeta.item()))
        logger.info('c0    : {:6.4f}'.format(c0.item()))
        logger.info('c1    : {:6.4f}'.format(c1.item()))
        logger.info('c2    : {:6.4f}'.format(c2.item()))

    # Calculate negative log-like given a temporal sequence
    def forward(self, his_t, t_start, t_end):
        mu, zeta, c0, c1, c2 = self.get_param()
        if method == 'numroot':
            return nll_num_root(his_t, t_start, t_end, mu, zeta, c0, c1, c2)
        elif method == 'close':
            return nll_close(his_t, t_start, t_end, mu, zeta, c0, c1, c2)


def main():
    torch.autograd.set_detect_anomaly(True)

    trainset = TPPWrapper(lamb_func, n_sample=1, t_end=5000.0, max_lamb=100., fn='data/temporal/correctingHawkes.db')
    his_t = torch.tensor(trainset.seqs[0].numpy()).to(device)

    model = CorrectingHawkes(1.0, 1.0, 1.0, 1.0, 1.0).to(device)
    example = None
    if method == 'numroot':
        example = nll_num_root(his_t, 0.0, 5000.0, .4, 1., -1., 1., .2)
    elif method == 'close':
        example = nll_close(his_t, 0.0, 5000.0, .4, 1., -1., 1., .2)
    logger.info(f'optimal nll: {example}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    train_loss = []

    epoch = 1
    early_stop = 20  # Stop if no update for 20 epochs
    while len(train_loss) <= early_stop or min(train_loss[-early_stop:]) <= min(train_loss[:-early_stop]) - 1.:
        optimizer.zero_grad()
        loss = model(his_t, 0.0, 5000.0)
        try:
            loss.backward()
        except RuntimeError as e:
            model.log_param()
            raise e
        logger.info(f'Epoch {str(epoch).ljust(8)} \t Train Loss: {loss.item():6.4f}')
        epoch += 1
        train_loss.append(loss.item())
        optimizer.step()

    plt.plot(train_loss)
    plt.savefig(f'figs/{fn}.png')
    logger.info(f'Writing to figs/{fn}.png...')
    plt.show()

    model.log_param()  # Print parameters


def test():
    print(quartic(torch.tensor([0.3, 0.3, 0.3, 0.3]),
                  torch.tensor([-0.1, -0.1, -0.1, -0.1]),
                  torch.tensor([2.1, -2.9, -0.405, -0.35]),
                  torch.tensor([-4, 6, 1.01, 0.9]),
                  torch.tensor([5, -5, -0.01, 0.1])))


if __name__ == '__main__':
    """
    # Close
    CorrectingMLE : 259 - mu    : 0.4250
    CorrectingMLE : 260 - zeta  : 1.2670
    CorrectingMLE : 261 - c0    : -0.9889
    CorrectingMLE : 262 - c1    : 0.4252
    CorrectingMLE : 263 - c2    : 0.9122
    """
    method = 'numroot'
    lr = 0.08
    fn = gen_fn(method = method, lr = lr)
    logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
    logger = logging.getLogger('root')
    logger.info('\n' + '-' * 100)
    plt.set_loglevel('warning')
    device = get_device()
    main()
