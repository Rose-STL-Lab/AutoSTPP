from torch import nn
import torch
from scipy.integrate import quad
import numpy as np
from utils import get_device, AverageMeter, eval_loss, load_config, tqdm, tenumerate


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

    def forward(self, x):
        """
        f(x) forward pass
        """
        return self.layers(x).squeeze(-1)

    def dnforward(self, x, dims):
        """
        dn f(x) / dx[dims[1]] dx[dims[2]] ... forward pass
        """
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


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == '__main__':
    import os
    from models.st_model import AutoIntSTPPSameInfluence
    from data.data import SlidingWindowWrapper
    from torch.utils.data import DataLoader
    from datetime import datetime
    from loguru import logger

    dataset = 'covid_nj_cases'
    npz = np.load(f'data/spatiotemporal/{dataset}.npz', allow_pickle=True)

    train_set = SlidingWindowWrapper(npz['train'], normalized=True)
    val_set = SlidingWindowWrapper(npz['val'], normalized=True, min=train_set.min, max=train_set.max)
    test_set = SlidingWindowWrapper(npz['test'], normalized=True, min=train_set.min, max=train_set.max)

    config = load_config()

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

    device = get_device(free=False, min_ram=0)

    model = AutoIntSTPPSameInfluence(config['hid_dim'], device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    best_eval = np.infty
    sll_meter = AverageMeter()
    tll_meter = AverageMeter()
    loss_meter = AverageMeter()

    time_now = str(datetime.now())
    parent_dir = f'models/AutoInt-STPP-Same-Influence-{dataset}-{time_now}'
    os.mkdir(parent_dir)

    for epoch in range(config['n_epoch']):
        loss_total = 0
        model.train()
        for index, data in tenumerate(train_loader):
            st_x, st_y, _, _, _ = data
            optimizer.zero_grad()
            loss, sll, tll = model(st_x, st_y)

            if torch.isnan(loss):
                logger.error("Numerical error, quiting...")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step()

            # Project to feasible set
            model.project()

            loss_meter.update(loss.item())
            sll_meter.update(sll.mean().item())
            tll_meter.update(tll.mean().item())

        scheduler.step()

        logger.info("In Epoch {} | "
                    "total loss: {:5f} | Space: {:5f} | Time: {:5f}".format(
            epoch, loss_meter.avg, sll_meter.avg, tll_meter.avg
        ))

        if (epoch + 1) % config['n_eval_epoch'] == 0:
            model.eval()
            val_loss, val_s, val_t = eval_loss(model, val_loader)
            logger.info("Evaluate   | Val Loss {:5f} | Space: {:5f} | Time: {:5f}".format(val_loss, val_s, val_t))
            if val_loss < best_eval:
                best_eval = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, f'{parent_dir}/AutoIntSTPP-{epoch}.mod')

    logger.info("training done!")
