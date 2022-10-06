import numpy as np
import torch


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device("cpu")


def chebyCoef(func, N, upper, lower):
    nodes = torch.cos((torch.arange(N) + 0.5) / torch.ones(N) / N * np.pi).to(device)
    xs = (upper + lower) / 2 + (upper - lower) / 2 * nodes
    f = func(xs)
    u = [torch.ones(N).to(device), nodes]
    c = [torch.mean(f), 2 * torch.mean(f * nodes)]
    for i in range(2, N):
        u.append(2 * nodes * u[i-1] - u[i-2])
        c.append(2 * torch.mean(f * u[i]))
    return torch.stack(c, 0)


def f_approx(x, cs, upper, lower):
    N = len(cs)
    u = (x - (upper + lower) / 2) / (upper - lower) * 2
    us = torch.zeros(N, len(u)).to(device)
    us[0] = 1
    us[1] = u
    for i in range(2, N):
        us[i] = 2*u*us[i-1] - us[i-2]
    return cs @ us


# Given f(x) = ∑ c_i T_i(x), calculate ∫ f from lower to x
def chebyInt(x, cs, upper, lower):
    N = len(cs)
    u = (x - (upper + lower) / 2) / (upper - lower) * 2
    u = torch.cat([torch.tensor([-1.0]).to(device), u]) # Append lower to front
    
    ts = [torch.ones_like(u), u] # T_0(x) = 1, T_1(x) = x
    qs = [u, u**2 / 2]           # Q_0(x) = x, Q_1(x) = x**2 / 2
    
    for i in range(2, N+1):
        ts.append(2*u*ts[i-1] - ts[i-2])
    for i in range(2, N):
        qs.append((ts[i+1]/(i+1) - ts[i-1]/(i-1))/2)
    
    ts = torch.stack(ts, 0)
    qs = torch.stack(qs, 0)
    qs = qs[:, 1:] - qs[:, :1] # Second fundamental theorem of calculus
    
    qs *= (upper - lower) / 2
    
    return cs @ qs