import pytest

from autoint_mlp import model
from conftest import get_params, relpath, update_params, log_config, put_result

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

    M = model
    L = deepcopy(model)
    for i, layer in enumerate(M.layers):  # Flip all activation function in M
        if type(layer) == ReQU:
            L.layers[i] = ReQUFlip()
        if type(layer) == Sine:
            L.layers[i] = SineFlip()

    return Cuboid(L, M)


@pytest.fixture(
    scope="module",
    params=get_params('dataloader')
)
def dataset(device, request):
    import os
    import torch
    from utils import relpath_under
    from data.toy import Integral3DWrapper

    update_params('dataloader', request)
    dataset_fn = relpath_under('data') + '/dataset.pkl'

    if not request.param['regenerate'] and os.path.exists(dataset_fn):  # Loading
        dataset = torch.load(dataset_fn)
    else:
        dataset = Integral3DWrapper([[Xa, Xb], [Ya, Yb], [Za, Zb]], func_to_fit,
                                    request.param['sampling_density'], device)
        torch.save(dataset, dataset_fn)

    return dataset


@pytest.fixture(
    scope="module",
    params=get_params('dataloader')
)
def dataloader(dataset, request):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, shuffle=True, batch_size=request.param['batch_size'])


@pytest.fixture(scope="class")
def integral_mse_loss():
    def _integral_mse_loss(x, _, targets, model):
        """
        Calculate the MSE between dataset integrals and integral network output

        :param x: (N, 3), dataset x
        :param _: (N,), not using dataset integrants
        :param targets: (N,) dataset integrals
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

        :param x: (N, 3), dataset x
        :param integrants: (N,), dataset integrants
        :param _: (N,), not using dataset targets
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
    params=get_params('trained_model')
)
def trained_model(cuboid, dataloader, device, request):
    import torch
    from loguru import logger
    import os
    import datetime

    model = cuboid
    update_params('trained_model', request)
    model_fn = relpath('models') + '.pkl'
    log_config()

    if not request.param['retrain']:  # try to use the previous trained model
        if os.path.exists(model_fn):
            model.load_state_dict(torch.load(model_fn)['model_state_dict'])
            logger.info('Previous model found and loaded.')
            model.eval()
            return model
        else:
            logger.info('Previous model not found. Retraining...')

    a = datetime.datetime.now()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    epoch = 0
    losses = []
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
        losses.append(avg_loss)
        if len(losses) > 200 and min(losses[:-100]) - 1e-5 < min(losses[-100:]):  # No improvement for 100 epochs
            break
        logger.debug(f'Epoch {epoch} \t Loss: {avg_loss:.6f}')
        scheduler.step()

    b = datetime.datetime.now()
    put_result('train_time_per_epoch', (b - a).total_seconds() / epoch)
    put_result('n_epoch', epoch)
    put_result('ending_loss', loss.item())

    model.eval()

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, model_fn)

    return model


class TestClass:

    @staticmethod
    def plot(N, f_gt, f_pd, title, file_name):
        import numpy as np
        from visualization.plotter import plot_lambst_interactive

        f_gt = f_gt.reshape((N + 1, N + 1, N + 1)).transpose([2, 0, 1])
        f_pd = f_pd.reshape((N + 1, N + 1, N + 1)).transpose([2, 0, 1])
        x_range = np.arange(Xa, Xb + (Xb - Xa) / N, (Xb - Xa) / N)
        y_range = np.arange(Ya, Yb + (Yb - Ya) / N, (Yb - Ya) / N)
        t_range = np.arange(Za, Zb + (Zb - Za) / N, (Zb - Za) / N)

        fig = plot_lambst_interactive([f_gt, f_pd], x_range, y_range, t_range, show=False,
                                      master_title=title, subplot_titles=['Ground Truth', 'Predicted'])
        fig.write_html(f"{relpath('figs', True)}/{file_name}.html")

    @pytest.mark.skip(reason="No visualization")
    def test_integral_fit_fast(self, dataset, trained_model, device):
        """
        Test the error of the learned integral function using the training dataset

        :param trained_model: the trained model (on either integral or integrant)
        :param device: GPU device
        """
        import torch
        from loguru import logger

        X = dataset.X
        F_gt = dataset.F.cpu().detach().numpy()
        F_pd = trained_model.int_lamb(torch.ones_like(X[:, 0]) * torch.tensor(Xa), X[:, 0],
                                      torch.ones_like(X[:, 1]) * torch.tensor(Ya), X[:, 1],
                                      torch.ones_like(X[:, 2]) * torch.tensor(Za), X[:, 2])
        F_pd = F_pd.squeeze(-1).cpu().detach().numpy()

        logger.error(abs(F_gt - F_pd).mean())

    def test_integral_fit(self, trained_model, device):
        """
        Test the error of the learned integral function using regularly sampled points in 3D

        :param trained_model: the trained model (on either integral or integrant)
        :param device: GPU device
        """
        import torch
        import numpy as np
        import os
        from utils import arange, relpath_under
        from scipy.integrate import tplquad
        from tqdm import tqdm
        from loguru import logger
        from torch.utils.data import DataLoader

        N = 50
        X = arange(N, [[Xa, Xb], [Ya, Yb], [Za, Zb]])

        F_gt_fn = relpath_under('data') + '/F_gt.pkl'
        if not pytest.config[__file__]['dataloader']['regenerate'] and os.path.exists(F_gt_fn):  # Loading
            F_gt = torch.load(F_gt_fn)
        else:
            F_gt = torch.tensor([tplquad(func_to_fit, Xa, x[0],
                                                      Ya, x[1],
                                                      Za, x[2])[0] for x in tqdm(X)])
            torch.save(F_gt, F_gt_fn)
        F_gt = F_gt.cpu().detach().numpy()

        X = torch.Tensor(X).to(device)
        X_loader = DataLoader(X, shuffle=False, batch_size=4096)
        F_pd = []
        for X_ in X_loader:
            F_pd_ = trained_model.int_lamb(torch.ones_like(X_[:, 0]) * torch.tensor(Xa), X_[:, 0],
                                           torch.ones_like(X_[:, 1]) * torch.tensor(Ya), X_[:, 1],
                                           torch.ones_like(X_[:, 2]) * torch.tensor(Za), X_[:, 2])
            F_pd.append(F_pd_.squeeze(-1).cpu().detach().numpy())

        F_pd = np.concatenate(F_pd)
        logger.error(abs(F_gt - F_pd).mean())
        put_result('integral_test_MSE', abs(F_gt - F_pd).mean())

        title = f'Learned integral by training with {pytest.fn_params["trained_model"]["loss_func"]}'
        self.plot(N, F_gt, F_pd, title, 'integral')

    def test_integrant_fit(self, trained_model, device):
        """
        Test the error of the learned integrant function using regularly sampled points in 3D

        :param trained_model: the trained model (on either integral integrant)
        :param device: GPU device
        """
        import numpy as np
        import torch
        from utils import arange
        from loguru import logger
        from torch.utils.data import DataLoader

        N = 50
        X = arange(N, [[Xa, Xb], [Ya, Yb], [Za, Zb]])

        f_gt = func_to_fit(X[:, 0], X[:, 1], X[:, 2])
        X_loader = DataLoader(torch.Tensor(X).to(device), shuffle=False, batch_size=4096)

        f_pd = []
        for X_ in X_loader:
            f_pd.append(trained_model(X_).squeeze(-1).cpu().detach().numpy())
        f_pd = np.concatenate(f_pd)

        logger.error(abs(f_gt - f_pd).mean())
        put_result('integrant_test_MSE', abs(f_gt - f_pd).mean())

        title = f'Learned integrant by training with {pytest.fn_params["trained_model"]["loss_func"]}'
        self.plot(N, f_gt, f_pd, title, 'integrant')
