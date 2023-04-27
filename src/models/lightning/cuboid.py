import pytorch_lightning as pl
from aim import Figure
from loguru import logger
from typing import Union, Optional, Callable, Any
from torch.optim.optimizer import Optimizer
from pytorch_lightning.core.optimizer import LightningOptimizer
from typing import List
import numpy as np
from visualization.plotter import plot_lambst_interactive
import torch


class BaseCuboid(pl.LightningModule):
    
    def __init__(
        self, 
        fit_on: str = 'f',
        test_on: str = 'F3',
        project: bool = True,
        bounds: List[List] = [[0., 3.], [0., 3.], [0., 3.]],
        grid_size: int = 50,
        learning_rate: float = 0.005,
        step_size: int = 20,
        gamma: float = 0.5,
    ) -> None:
        """Cuboid is capable of learning a function f(x) and its triple integral simultaneously

        Parameters
        ----------
        fit_on: str, optional
            Whether to fit the model on 
            f(x, y, z), 
            F1 := ∫ f(x, y, z) dx, 
            F2 := ∫∫ f(x, y, z) dxdy, or
            F3 := ∫∫∫ f(x, y, z) dxdydz
        test_on: str, optional
            Whether to test the model on
            f(x, y, z), 
            F1 := ∫ f(x, y, z) dx, 
            F2 := ∫∫ f(x, y, z) dxdy, or
            F3 := ∫∫∫ f(x, y, z) dxdydz
        project: bool, optional
            Whether to project the weights to be non-negative
        bounds: List[List], optional
            X, Y, Z integral limits of the cuboid
        grid_size: int, optional
            Number of grid points in each dimension for visualization
        learning_rate : float, optional
            Learning rate of Cuboid
        step_size : int, optional
            Scheduler step size
        gamma : float, optional
            Scheduler gamma
        """
        super().__init__()
        self.save_hyperparameters()
        self.train_pd = []
        self.train_gt = []
        self.test_pd = []
        self.test_gt = []
        
        self.Xa = self.hparams.bounds[0][0]
        self.Ya = self.hparams.bounds[1][0]
        self.Za = self.hparams.bounds[2][0]
        self.cuboid = None
        
    def on_fit_start(self):
        logger.info(f'model.dtype: {self.dtype}')
        logger.info(self)
        if self.hparams.project:
            self.project()

    def optimizer_step(
        self, 
        epoch: int, 
        batch_idx: int, 
        optimizer: Union[Optimizer, LightningOptimizer], 
        optimizer_closure: Optional[Callable[[], Any]] = None
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.hparams.project:
            self.project()
            
    def on_test_batch_end(
        self, 
        outputs: Optional[Any], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        self.train_pd.append(outputs[self.hparams.fit_on + '_pd'])
        self.train_gt.append(outputs[self.hparams.fit_on + '_gt'])
        self.test_pd.append(outputs[self.hparams.test_on + '_pd'])
        self.test_gt.append(outputs[self.hparams.test_on + '_gt'])

    def on_test_epoch_end(self) -> None:
        """Logs the predictions for the test step as a 3D interactive plot."""
        N = self.hparams.grid_size
        title = 'fit on ' + self.hparams.fit_on + ' and test on ' + self.hparams.test_on
        
        for gt, pd, name in zip(
            [self.train_gt, self.test_gt],
            [self.train_pd, self.test_pd],
            [self.hparams.fit_on, self.hparams.test_on]
        ):
            gt = torch.cat(gt).detach().cpu().numpy()
            pd = torch.cat(pd).detach().cpu().numpy()
            
            try:
                gt = gt.reshape((N + 1, N + 1, N + 1)).transpose([2, 0, 1])
                pd = pd.reshape((N + 1, N + 1, N + 1)).transpose([2, 0, 1])
            except ValueError:
                logger.info("Cannot reshape the predictions to a 3D grid: possibly incomplete test set.")
                return
                
            [[Xa, Xb], [Ya, Yb], [Za, Zb]] = self.hparams.bounds
            x_range = np.arange(Xa, Xb + (Xb - Xa) / N, (Xb - Xa) / N)
            y_range = np.arange(Ya, Yb + (Yb - Ya) / N, (Yb - Ya) / N)
            t_range = np.arange(Za, Zb + (Zb - Za) / N, (Zb - Za) / N)

            fig = plot_lambst_interactive([gt, pd], x_range, y_range, t_range, show=False,
                                          master_title=title, subplot_titles=['Ground Truth', 'Predicted'])

            self.logger.experiment.track(Figure(fig), 
                                         name=f'{name} predictions', 
                                         step=0, 
                                         context={'subset': 'test'},)

    def project(self):
        """
        Employ non-negative constraint
        """
        self.cuboid.project()
        
    def forward(self, batch, mode='f'):
        """
        Calculate the MSE between the predicted integrant / integral and the ground truth integrant / integral

        Parameters
        ----------
        batch : tuple of 3 numpy arrays (x, integrants, integrals)
            The batch of data containing inputs (x), integrants, and integrals.
        mode : str
            Whether to calculate the MSE of f, F1, F2, or F3
        Returns
        -------
        mse : float
            The average mean squared error
        """
        x, f_gt, F1_gt, F2_gt, F3_gt = batch
        loss_func = torch.nn.MSELoss()
        if mode == 'f':
            pd = self.cuboid(x)
            pd = pd.reshape(-1)
            loss = loss_func(pd, f_gt)
        elif mode == 'F1':
            # TODO: implement F1
            pass
        elif mode == 'F2':
            pd = self.cuboid.lamb_t(torch.ones_like(x[:, 0]) * torch.tensor(self.Xa), x[:, 0],
                                    torch.ones_like(x[:, 1]) * torch.tensor(self.Ya), x[:, 1],
                                    x[:, 2])
            pd = pd.reshape(-1)
            loss = loss_func(pd, F2_gt)
        elif mode == 'F3':
            pd = self.cuboid.int_lamb(torch.ones_like(x[:, 0]) * torch.tensor(self.Xa), x[:, 0],
                                      torch.ones_like(x[:, 1]) * torch.tensor(self.Ya), x[:, 1],
                                      torch.ones_like(x[:, 2]) * torch.tensor(self.Za), x[:, 2])
            pd = pd.reshape(-1)
            loss = loss_func(pd, F3_gt)
        else:
            raise ValueError(f"mode {mode} not supported")
        return {
            'loss': loss,
            'f_gt': f_gt,
            'F1_gt': F1_gt,
            'F2_gt': F2_gt,
            'F3_gt': F3_gt,
            mode + '_pd': pd,
        }

    def training_step(self, batch, batch_idx):
        loss = self(batch, self.hparams.fit_on)['loss']
        self.log('train_mse', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch, self.hparams.fit_on)['loss']
        self.log('val_mse', loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        res = self(batch, self.hparams.test_on)
        loss = res['loss']
        test_pd = res[self.hparams.test_on + '_pd']
        test_gt = res[self.hparams.test_on + '_gt']
        res = self(batch, self.hparams.fit_on)
        train_pd = res[self.hparams.fit_on + '_pd']
        train_gt = res[self.hparams.fit_on + '_gt']
        self.log('test_mse', loss.item())
        return {
            "loss": loss.item(),
            self.hparams.fit_on + "_pd": train_pd, 
            self.hparams.fit_on + "_gt": train_gt,
            self.hparams.test_on + "_pd": test_pd, 
            self.hparams.test_on + "_gt": test_gt
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, 
                                                    gamma=self.hparams.gamma)
        return [optimizer], [scheduler]
