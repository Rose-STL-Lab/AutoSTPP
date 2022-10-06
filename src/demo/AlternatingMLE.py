from src.data.data import TPPWrapper

import numpy as np
import torch

eps = 1e-10  # A negligible positive number
np.random.seed(0)

device = torch.device('cuda:0')
print("Device name: ", torch.cuda.get_device_name(0))
print("Number of CUDAs(cores): ", torch.cuda.device_count())


def lamb_func(t, his_t):
    alpha = .2
    beta = .2
    mu = .2

    zeta = 1.
    c0 = -1.
    c1 = 1.
    c2 = 1.

    delta_t = t - his_t[his_t < t]
    dt0 = delta_t[0::2]  # even index entries
    dt1 = delta_t[1::2]  # even index entries

    lamb_t = mu + alpha * (np.sum(np.exp(-beta * dt0)) +
                           np.sum((c2 * dt1 ** 2 + c1 * dt1 + c0) * np.exp(-zeta * dt1)))
    lamb_t = abs(lamb_t)
    L = 2 / lamb_t

    count = delta_t[0::2]
    count = sum(count[count < 10])

    M = abs(mu + alpha * (np.sum(np.exp(-beta * dt0)) + count * 0.677))

    return lamb_t, L, M


def nll(his_t, alpha, beta, mu, zeta, c0, c1, c2):
    N = len(his_t)

    # Calculate outer subtraction
    delta_t = torch.tril(his_t.reshape(-1, 1) - his_t.reshape(1, -1), -1)
    exc_delta_t = delta_t.clone()
    inh_delta_t = delta_t.clone()

    # The event index such that the influence is excitation; given here
    exc_mask = torch.zeros(N, dtype=torch.bool)
    exc_mask[0::2] = 1

    exc_delta_t[:, ~exc_mask] = 0.
    inh_delta_t[:, exc_mask] = 0.

    exc_lamb = torch.sum(torch.tril(torch.exp(-beta * exc_delta_t), -1)[:, exc_mask], -1)
    inh_lamb = torch.sum(torch.tril((c2 * inh_delta_t ** 2 + c1 * inh_delta_t + c0)
                                    * torch.exp(-zeta * inh_delta_t), -1)[:, ~exc_mask], -1)

    lamb = torch.abs(mu + (exc_lamb + inh_lamb) * alpha)

    # - ∑ log λ(t_i)
    term_1 = - torch.sum(torch.log(lamb))

    # + ∫ λ(t) dt = μT - α/β ∑ [exp(-β(T-t_i)) - 1] (0~T)
    t_end = his_t[-1]
    t_start = his_t[0]
    term_2 = mu * (t_end - t_start)
    term_2 -= alpha / beta * torch.sum((torch.exp(-beta * (t_end - his_t[exc_mask])) - torch.tensor(1.)))
    temp = t_end - his_t[~exc_mask]

    term_3 = torch.sum(torch.exp(-zeta * temp) * (-c2 * (zeta * temp * (zeta * temp + 2) + 2) -
                                                  zeta * (c1 * zeta * temp + c1 + c0 * zeta)) + (
                               2 * c2 + zeta * (c1 + c0 * zeta))) / zeta ** 3
    term_3 *= alpha

    return term_1 + term_2 + term_3


def main():
    trainset = TPPWrapper(lamb_func, n_sample=1, t_end=10000.0, max_lamb=100., fn='data/temporal/alterHawkes.db')
    his_t = torch.tensor(trainset.seqs[0].numpy()).to(device)
    print(nll(his_t, .2, .2, .2, 1., -1., 1., 1.))


if __name__ == '__main__':
    main()
