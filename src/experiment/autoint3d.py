from torch import nn
import torch
from scipy.integrate import quad
from tqdm.auto import tqdm
import numpy as np


class Integral1DWrapper(torch.utils.data.Dataset):
    """
    Wrap x, f(x), and F(x)=∫[0, x] f(t)dt for x in R^1
    """
    def __init__(self, range, fn, sampling_density, device):
        self.X = torch.rand(sampling_density) * (range[1] - range[0]) + range[0]
        self.f = fn(self.X)
        self.F = torch.tensor([quad(fn, range[0], x)[0] for x in self.X])

        self.X = self.X.float().to(device)
        self.f = self.f.float().to(device)
        self.F = self.F.float().to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.f[idx], self.F[idx]


class Sine(nn.Module):
    def __init__(self):
        super().__init__()  # init the base class

    def forward(self, input):
        return torch.sin(input)


sin = Sine()


class MLP1D(nn.Module):
    """
    Multilayer Perceptron.
    """
    def __init__(self, model_type):
        super().__init__()
        self.layers = model_type(
            nn.Linear(1, 128),
            sin,
            nn.Linear(128, 128),
            sin,
            nn.Linear(128, 128),
            sin,
            nn.Linear(128, 1)
        )

    '''f(x) forward pass'''
    def forward(self, x):
        if x.shape[-1] != 1:
            x = x.unsqueeze(-1)  # Add one dimension
        return self.layers(x).squeeze()

    ''' fn(x) forward pass'''
    def dnforward(self, x, dims):
        if x.shape[-1] != 1:
            x = x.unsqueeze(-1)  # Add one dimension
        return self.layers.dnforward(x, dims).squeeze()


class MLPnD(nn.Module):
    """
    Multilayer Perceptron.
    """
    def __init__(self, model_type, input_size, ac=sin):
        super().__init__()
        self.input_size = input_size
        self.layers = model_type(
            nn.Linear(input_size, 128),
            ac,
            nn.Linear(128, 128),
            ac,
            nn.Linear(128, 128),
            ac,
            nn.Linear(128, 1)
        )

    '''f(x) forward pass'''
    def forward(self, x):
        return self.layers(x).squeeze(-1)

    ''' f(n)(x) forward pass'''
    def dnforward(self, x, dims):
        return self.layers.dnforward(x, dims).squeeze(-1)


def train_1d(model, dataloader, device):
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in tqdm(range(0, 50)):
        current_loss = []

        for i, (x, integrants, targets) in enumerate(dataloader):
            optimizer.zero_grad()

            d2f_anchor = torch.tensor([np.pi / 2.0, 3 * np.pi / 2.0]).to(device)  # mount f''(π/2) = 0, f''(3π/2) = 0
            df_anchor = torch.tensor([0.0, np.pi]).to(device)  # mount f'(0) = 0, f'(π) = 0
            f_anchor = torch.tensor([np.pi / 2.0, 3 * np.pi / 2.0]).to(device)  # mount f(π/2) = 0, f(3π/2) = 0

            # Learn seventh derivative
            loss = loss_func(model.dnforward(x, [0, 0, 0]), integrants) + \
                   torch.square(model.dnforward(d2f_anchor, [0, 0])).sum() + \
                   torch.square(model.dnforward(df_anchor, [0])).sum() + \
                   torch.square(model(f_anchor)).sum()

            # Use buffer: 0.44 s per epoch -> 0.4 s

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss.append(loss.item())

        print(f'Epoch {epoch} \t Loss: {sum(current_loss) / len(current_loss)}')

        scheduler.step()
