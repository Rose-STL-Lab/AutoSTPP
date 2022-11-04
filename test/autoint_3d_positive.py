import pytest

# noinspection PyUnresolvedReferences
from autoint_mlp import model
from conftest import relpath, update_params, log_config

Xa = 0.
Xb = 3.
Ya = 0.
Yb = 3.
Za = 0.
Zb = 3.


def func_to_fit(x, y, z):
    import numpy as np
    return np.sin(x) * np.cos(y) * np.sin(z) + 1


@pytest.fixture(
    scope="class"
)
def cuboid(device, model):
    from copy import deepcopy
    from integration.autoint import Cuboid
    from integration.autoint import ReQU, ReQUFlip, Sine, SineFlip

    L = model
    M = deepcopy(model)
    for i, layer in enumerate(L.layers):  # Flip all activation function in M
        if type(layer) == ReQU:
            M.layers[i] = ReQUFlip()
        if type(layer) == Sine:
            M.layers[i] = SineFlip()

    return Cuboid(L, M)


@pytest.fixture(
    scope="module",
    params=pytest.params['dataloader']
)
def dataloader(device, request):
    import os
    import torch
    from utils import relpath_under
    from torch.utils.data import DataLoader
    from data.toy import Integral3DWrapper

    update_params('dataloader', request)
    dataset_fn = relpath_under('data') + '/dataset.pkl'

    if not request.param['regenerate'] and os.path.exists(dataset_fn):  # Loading
        dataset = torch.load(dataset_fn)
    else:
        dataset = Integral3DWrapper([[Xa, Xb], [Ya, Yb], [Za, Zb]], func_to_fit,
                                    request.param['sampling_density'], device)
        torch.save(dataset, dataset_fn)

    return DataLoader(dataset, shuffle=True, batch_size=request.param['batch_size'])


@pytest.fixture(scope="class")
def integral_mse_loss():
    def _integral_mse_loss(x, _, targets, model):
        """
        Calculate the MSE between dataset integrals and integral network output

        :param x: (N, 2), dataset x
        :param _: not using dataset integrants
        :param targets: dataset integrals
        :param model: the cuboid model
        :return: the MSE loss, scalar
        """
        import torch

        loss_func = torch.nn.MSELoss()
        F = model.int_lamb(torch.ones_like(x[:, 0]) * torch.tensor(Xa), x[:, 0],
                           torch.ones_like(x[:, 1]) * torch.tensor(Ya), x[:, 1],
                           torch.ones_like(x[:, 2]) * torch.tensor(Za), x[:, 2]).squeeze(-1)
        loss = loss_func(F, targets)
        return loss
    return _integral_mse_loss


@pytest.fixture(scope="class")
def gradient_mse_loss():
    def _gradient_mse_loss(x, integrants, _, model):
        """
        Calculate the MSE between dataset integrants and derivative network output

        :param x: (N, 2), dataset x
        :param integrants: dataset integrants
        :param _: not using dataset targets
        :param model: the cuboid model
        :return: the MSE loss, scalar
        """
        import torch

        loss_func = torch.nn.MSELoss()
        f = model(x).squeeze(-1)
        loss = loss_func(integrants, f)
        return loss
    return _gradient_mse_loss


@pytest.fixture(
    scope="class",
    params=pytest.params['trained_model']
)
def trained_model(cuboid, dataloader, device, request):
    import torch
    from loguru import logger
    import os

    model = cuboid
    model_fn = relpath('models') + '.pkl'
    update_params('trained_model', request)
    log_config()

    if not request.param['retrain']:  # try to use the previous trained model
        if os.path.exists(model_fn):
            model.load_state_dict(torch.load(model_fn)['model_state_dict'])
            logger.info('Previous model found and loaded.')
            model.eval()
            return model
        else:
            logger.info('Previous model not found. Retraining...')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    epoch = 0
    loss = torch.tensor(1.0)

    while loss.item() > 5e-5:
        current_loss = []
        epoch += 1

        for i, (x, integrants, targets) in enumerate(dataloader):
            optimizer.zero_grad()

            loss = request.getfixturevalue(request.param['loss_func'])(x, integrants, targets, model)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss.append(loss.item())

        avg_loss = sum(current_loss) / len(current_loss)
        logger.debug(f'Epoch {epoch} \t Loss: {avg_loss:.6f}')
        scheduler.step()

    model.eval()

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, model_fn)

    return model


class TestClass:

    # TODO: increase speed
    def test_integral_fit(self, trained_model, device):
        """
        Test the error of the learned integral function

        :param trained_model: the trained model (on either integral or integrant)
        :param device: GPU device
        """
        import torch
        from utils import arange
        from scipy.integrate import tplquad
        from tqdm import tqdm
        from loguru import logger

        X = arange(10, [[Xa, Xb], [Ya, Yb], [Za, Zb]])
        F_gt = torch.tensor([tplquad(func_to_fit, Xa, x[0],
                                                  Ya, x[1],
                                                  Za, x[2])[0] for x in tqdm(X)])
        X = torch.Tensor(X).to(device)
        F_pd = trained_model.int_lamb(torch.ones_like(X[:, 0]) * torch.tensor(Xa), X[:, 0],
                                      torch.ones_like(X[:, 1]) * torch.tensor(Ya), X[:, 1],
                                      torch.ones_like(X[:, 2]) * torch.tensor(Za), X[:, 2])
        F_pd = F_pd.squeeze(-1).cpu().detach().numpy()

        logger.error(abs(F_gt - F_pd).mean())

    # TODO: add visualization
    def test_integrant_fit(self, trained_model, device):
        """
        Test the error of the learned integrant function

        :param trained_model: the trained model (on either integral integrant)
        :param device: GPU device
        """
        import torch
        from utils import arange
        from loguru import logger

        trained_model.eval()

        X = arange(10, [[Xa, Xb], [Ya, Yb], [Za, Zb]])

        f_gt = func_to_fit(X[:, 0], X[:, 1], X[:, 2])
        f_pd = trained_model(torch.Tensor(X).to(device)).squeeze(-1).cpu().detach().numpy()

        logger.error(abs(f_gt - f_pd).mean())
