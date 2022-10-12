import pytest

Xa = 0.
Xb = 6.
params = {}


@pytest.fixture(
    scope="class",
    params=pytest.params['dataloader']
)
def dataloader(device, request):
    import torch
    import numpy as np
    from scipy.integrate import quad
    from torch.utils.data import DataLoader

    batch_size = request.param['batch_size']
    global params
    params.update({'dataloader': request.param})

    class Integral1DWrapper(torch.utils.data.Dataset):
        """
        Wrap x, f(x), and F(x)=∫[0, x] f(t)dt for x in R^1
        """

        def __init__(self, bound, fn, sampling_density):
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

    def func_to_fit(X):
        return np.sin(X)

    dataset = Integral1DWrapper([Xa, Xb], func_to_fit, sampling_density=1024)
    return DataLoader(dataset, shuffle=True, batch_size=batch_size)


@pytest.fixture(
    scope="class",
    params=pytest.params['model']
)
def model(device, request):
    import torch
    from torch import nn
    from integration.autoint import MixSequential

    act = request.param['act']
    n_layers = request.param['n_layers']
    hid_dim = request.param['hid_dim']
    global params
    params.update({'model': request.param})

    class Sine(nn.Module):
        def __init__(self):
            super().__init__()  # init the base class

        @staticmethod
        def forward(x):
            return torch.sin(x)

    act_dict = {
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "sine": Sine()
    }
    act_layer = act_dict[act]

    class MLP1D(nn.Module):
        """
        Multilayer Perceptron.
        """

        def __init__(self):
            super().__init__()
            assert n_layers >= 1
            layers = [nn.Linear(1, hid_dim), ]
            for _ in range(n_layers - 1):
                layers.append(act_layer)
                layers.append(nn.Linear(hid_dim, hid_dim))
            layers.append(act_layer)
            layers.append(nn.Linear(hid_dim, 1))
            self.layers = MixSequential(*layers)

        def forward(self, x):
            """
            f(x) forward pass
            """
            if x.shape[-1] != 1:
                x = x.unsqueeze(-1)  # Add one dimension
            return self.layers(x).squeeze()

        def dnforward(self, x, dims):
            """
            fn(x) forward pass
            """
            if x.shape[-1] != 1:
                x = x.unsqueeze(-1)  # Add one dimension
            return self.layers.dnforward(x, dims).squeeze()

    return MLP1D().to(device)


@pytest.fixture(scope="class", autouse=True)
def log(model, dataloader):  # Run after their execution
    from loguru import logger
    from utils import relpath_under, serialize_config
    import os

    global params
    folder_name = f"{relpath_under('figs')}/{serialize_config(params)}"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    logger.info(params)


@pytest.fixture(
    scope="class",
    params=pytest.params['trained_model']
)
def trained_model(model, dataloader, device, request):
    import torch
    from torch import nn
    import numpy as np
    from loguru import logger
    from utils import relpath_under, serialize_config
    import os

    global params
    model_fn = f'{relpath_under("models")}/{serialize_config(params)}.pkl'
    if not request.param['retrain']:  # try to use the previous trained model
        if os.path.exists(model_fn):
            model.load_state_dict(torch.load(model_fn)['model_state_dict'])
            logger.info('Previous model found and loaded.')
            model.eval()
            return model
        else:
            logger.info('Previous model not found. Retraining...')

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)
    epoch = 0
    loss = torch.tensor(1.0)

    while loss.item() > 5e-5:
        current_loss = []
        epoch += 1

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

        avg_loss = sum(current_loss) / len(current_loss)
        logger.debug(f'Epoch {epoch} \t Loss: {avg_loss}')
        scheduler.step()

    model.eval()

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, model_fn)
    return model


class TestClass:

    @staticmethod
    def plot(x, f_gt, f_pd, title, file_name):
        import plotly.graph_objects as go
        from utils import relpath_under, serialize_config

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=f_gt, mode='markers', name='ground truth'))
        fig.add_trace(go.Scatter(x=x, y=f_pd, mode='markers', name='approximation'))

        fig.update_layout(
            title=title,
            xaxis_title="x",
            yaxis_title="f",
        )

        global params
        folder_name = serialize_config(params)
        fig.write_html(f"{relpath_under('figs')}/{folder_name}/{file_name}.html")

    def test_f(self, trained_model, device):
        import torch
        import numpy as np

        trained_model.eval()

        x = np.arange(Xa, Xb, 0.01)
        f_gt = np.cos(x)
        f_pd = trained_model(torch.Tensor(x).to(device)).cpu().detach().numpy()

        self.plot(x, f_gt, f_pd, r"$f(x) \text{ after fitting } f'''(x)$", "original_f")

        diff = abs(f_pd - f_gt)
        assert np.mean(diff) < 0.01  # Assert the error is smaller than .01

    def test_1st_derivative(self, trained_model, device):
        import torch
        import numpy as np

        trained_model.eval()

        x = np.arange(Xa, Xb, 0.01)
        f_gt = -np.sin(x)
        f_pd = trained_model.dnforward(torch.Tensor(x).to(device), [0]).cpu().detach().numpy()

        self.plot(x, f_gt, f_pd, r"$f'(x) \text{ after fitting } f'''(x)$", "1st_derivative")

        diff = abs(f_pd - f_gt)
        assert np.mean(diff) < 0.01  # Assert the error is smaller than .01

    def test_2nd_derivative(self, trained_model, device):
        import torch
        import numpy as np

        trained_model.eval()

        x = np.arange(Xa, Xb, 0.01)
        f_gt = -np.cos(x)
        f_pd = trained_model.dnforward(torch.Tensor(x).to(device), [0, 0]).cpu().detach().numpy()

        self.plot(x, f_gt, f_pd, r"$f''(x) \text{ after fitting } f'''(x)$", "2nd_derivative")

        diff = abs(f_pd - f_gt)
        assert np.mean(diff) < 0.01  # Assert the error is smaller than .01
