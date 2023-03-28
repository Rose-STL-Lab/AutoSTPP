import pytest

from autoint_mlp import model
from conftest import relpath, update_params, get_params, log_config, wandb_init, wandb_discard

Xa = 0.
Xb = 6.


@pytest.fixture(
    scope="module",
    params=get_params('dataloader')
)
def dataloader(device, request):
    import numpy as np
    from torch.utils.data import DataLoader
    from data.toy import Integral1DWrapper

    update_params('dataloader', request)

    def func_to_fit(X):
        return np.sin(X)

    dataset = Integral1DWrapper([Xa, Xb], func_to_fit, sampling_density=1024, device=device)
    return DataLoader(dataset, shuffle=True, batch_size=request.param['batch_size'])


@pytest.fixture(
    scope="class",
    params=get_params('trained_model')
)
def trained_model(model, dataloader, device, request):
    import torch
    from torch import nn
    import numpy as np
    from loguru import logger
    import wandb
    import os

    model_fn = relpath('models') + '.pkl'
    log_config()
    wandb_init(__file__)
    logger.info(model)
    if not request.param['retrain']:  # try to use the previous trained model
        if os.path.exists(model_fn):
            model.load_state_dict(torch.load(model_fn, map_location=device)['model_state_dict'])
            logger.info('Previous model found and loaded.')
            model.eval()
            return model
        else:
            logger.info('Previous model not found. Retraining...')

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    epoch = 0
    loss = torch.tensor(1.0)

    while loss.item() > 1e-3 and epoch < 1500:
        current_loss = []
        epoch += 1

        for i, (x, integrants, targets) in enumerate(dataloader):
            optimizer.zero_grad()

            x = x.unsqueeze(-1)

            # Mount f''(π/2) = 0, f''(3π/2) = 0
            d2f_anchor = torch.tensor([np.pi / 2.0, 3 * np.pi / 2.0]).unsqueeze(-1).to(device)
            # Mount f'(0) = 0, f'(π) = 0
            df_anchor = torch.tensor([0.0, np.pi]).unsqueeze(-1).to(device)
            # Mount f(π/2) = 0, f(3π/2) = 0
            f_anchor = torch.tensor([np.pi / 2.0, 3 * np.pi / 2.0]).unsqueeze(-1).to(device)

            # Learn third derivative
            loss = loss_func(model.dnforward(x, [0, 0, 0]).squeeze(), integrants) + \
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
            if loss.item() < 10.0:
                wandb.log({'loss': loss.item()})
            
            if torch.isnan(loss):
                id = wandb.run.id
                wandb.finish()
                wandb_discard(id)
                raise ValueError('NaN loss')

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
        import wandb

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=f_gt, mode='markers', name='ground truth'))
        fig.add_trace(go.Scatter(x=x, y=f_pd, mode='markers', name='approximation'))

        mathjax_link = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js'
        fig.update_layout(
            title=title,
            xaxis_title="x",
            yaxis_title="f",
        )
        
        html_fn = f"{relpath('figs', True)}/{file_name}.html"
        fig.write_html(html_fn, include_mathjax=mathjax_link)
        wandb.log({file_name: wandb.Html(html_fn)})

    def test_f(self, trained_model, device):
        import torch
        import numpy as np
        from loguru import logger
        import wandb

        trained_model.eval()

        x = np.arange(Xa, Xb, 0.01)
        f_gt = np.cos(x)
        f_pd = trained_model(torch.Tensor(x).to(device).unsqueeze(-1)) \
            .squeeze(-1).cpu().detach().numpy()

        self.plot(x, f_gt, f_pd, r"$f(x) \text{ after fitting } f'''(x)$", "original_f")

        diff = abs(f_pd - f_gt)
        logger.error(np.mean(diff))
        wandb.log({'f(x) error': np.mean(diff)})
        assert np.mean(diff) < 0.05  # Assert the error is smaller than .05

    def test_1st_derivative(self, trained_model, device):
        import torch
        import numpy as np
        from loguru import logger
        import wandb

        trained_model.eval()

        x = np.arange(Xa, Xb, 0.01)
        f_gt = -np.sin(x)
        f_pd = trained_model.dnforward(torch.Tensor(x).to(device).unsqueeze(-1), [0]) \
            .squeeze().cpu().detach().numpy()

        self.plot(x, f_gt, f_pd, r"$f'(x) \text{ after fitting } f'''(x)$", "1st_derivative")

        diff = abs(f_pd - f_gt)
        logger.error(np.mean(diff))
        wandb.log({'f\'(x) error': np.mean(diff)})
        assert np.mean(diff) < 0.05  # Assert the error is smaller than .05

    def test_2nd_derivative(self, trained_model, device):
        import torch
        import numpy as np
        from loguru import logger
        import wandb

        trained_model.eval()

        x = np.arange(Xa, Xb, 0.01)
        f_gt = -np.cos(x)
        f_pd = trained_model.dnforward(torch.Tensor(x).to(device).unsqueeze(-1), [0, 0]) \
            .squeeze().cpu().detach().numpy()

        self.plot(x, f_gt, f_pd, r"$f''(x) \text{ after fitting } f'''(x)$", "2nd_derivative")

        diff = abs(f_pd - f_gt)
        logger.error(np.mean(diff))
        wandb.log({'f\'\'(x) error': np.mean(diff)})
        assert np.mean(diff) < 0.05  # Assert the error is smaller than .05
