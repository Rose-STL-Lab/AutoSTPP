import torch
from scipy.integrate import quad, tplquad
from utils import tqdm, scale


class Integral1DWrapper(torch.utils.data.Dataset):
    """
    Wrap x, f(x), and F(x)=∫[0, x] f(t)dt for x in R^1
    """

    def __init__(self, bound, fn, sampling_density, device):
        self.X = torch.rand(sampling_density) * (bound[1] - bound[0]) + bound[0]
        self.f = fn(self.X)
        self.F = torch.tensor([quad(fn, bound[0], x)[0] for x in self.X])

        self.X = self.X.float().to(device)
        self.f = self.f.float().to(device)
        self.F = self.F.float().to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.f[idx], self.F[idx]


class Integral3DWrapper(torch.utils.data.Dataset):
    """
    Wrap x, f(x), and F(x)=∫[0, x] f(t)dt for x in R^1
    """

    def __init__(self, bound, fn, sampling_density, device):
        self.X = torch.rand(sampling_density, 3)
        self.X = scale(self.X, bound)

        self.f = fn(self.X[:, 0], self.X[:, 1], self.X[:, 2])
        self.F = torch.tensor([tplquad(fn, bound[0][0], x[0],
                                       bound[1][0], x[1],
                                       bound[2][0], x[2])[0] for x in tqdm(self.X)])

        self.X = self.X.float().to(device)
        self.f = self.f.float().to(device)
        self.F = self.F.float().to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.f[idx], self.F[idx]
