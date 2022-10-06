from torch import nn
import torch
from scipy.integrate import quad
import numpy as np
from torch.utils.data import DataLoader
from tqdm import trange
import plotly.graph_objects as go
import logging.config

from src.integration.autoint import MultSequential
from src.utils import get_device, imshow


class Integral1DWrapper(torch.utils.data.Dataset):
    """
    Wrap x, f(x), and F(x)=∫[0, x] f(t)dt for x in R^1
    """
    def __init__(self, range, fn, sampling_density):
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


def func_to_fit(X):
    return np.sin(X)


def train(model, dataloader):
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in trange(0, 50):
        current_loss = []

        for i, (x, integrants, targets) in enumerate(dataloader):
            optimizer.zero_grad()

            d2f_anchor = torch.tensor([[np.pi / 2.0], [3 * np.pi / 2.0]]).to(device)  # Mount f''(π/2) = 0, f''(3π/2) = 0
            df_anchor = torch.tensor([[0.0], [np.pi]]).to(device)  # Mount f'(0) = 0, f'(π) = 0
            f_anchor = torch.tensor([[np.pi / 2.0], [3 * np.pi / 2.0]]).to(device)  # Mount f(π/2) = 0, f(3π/2) = 0

            # Learn third derivative
            loss = loss_func(model.d3forward(x.unsqueeze(-1), [0, 0, 0]).squeeze(), integrants) + \
                   torch.square(model.d2forward(d2f_anchor, [0, 0])).sum() + \
                   torch.square(model.dforward(df_anchor, 0)).sum() + \
                   torch.square(model(f_anchor)).sum()

            # Use buffer: 0.44 s per epoch -> 0.4 s

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss.append(loss.item())

        train_loss = sum(current_loss) / len(current_loss)
        logger.info(f'Epoch {str(epoch).ljust(8)} \t Train Loss: {train_loss:6.4f}')

        scheduler.step()

    return model


def test(model, dataset):
    model.eval()

    X = dataset.X.cpu().detach().numpy()
    f_gt = -torch.sin(dataset.X).cpu().detach().numpy()
    f_pd = model.dforward(dataset.X.unsqueeze(-1), 0).cpu().squeeze().detach().numpy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=f_gt, mode='markers', name='ground truth', marker_size=4))
    fig.add_trace(go.Scatter(x=X, y=f_pd, mode='markers', name='approximation', marker_size=4))

    fig.update_layout(
        title=r"$f'(x) \text{ after fitting } f'''(x)$",
        xaxis_title="x",
        yaxis_title="f",
    )
    imshow(fig)


def main():
    # Generate data
    dataset = Integral1DWrapper([0, 6], func_to_fit, sampling_density=1024)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=64)

    model = MultSequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        ).to(device)

    model = train(model, dataloader)
    test(model, dataset)


if __name__ == '__main__':
    logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
    logger = logging.getLogger('root')
    logger.info('\n' + '-' * 100)
    device = get_device()
    main()
