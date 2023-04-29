import numpy as np
import torch
import pytorch_lightning as pl
from aim import Figure
from loguru import logger
from typing import Union, Optional, Callable, Any
from torch.optim.optimizer import Optimizer
from pytorch_lightning.core.optimizer import LightningOptimizer
from typing import List
from visualization.plotter import plot_lambst_interactive
from data.synthetic import STHPDataset, STSCPDataset
from abc import abstractmethod


def load_synt(name: str, x_num: int, y_num: int):
    """
    Load hparams and data of synthetic data
    Return None if name is not recognized
    
    :param name: name of the synthetic dataset
    :param x_num: number of x grid points
    :param y_num: number of y grid points
    
    :return synt: the synthetic dataset
    :return t_range: the time range of the dataset
    """
    if name == 'sthp0':
        synt = STHPDataset(s_mu=np.array([0, 0]), 
                            g0_cov=np.array([[.2, 0],
                                             [0, .2]]),
                            g2_cov=np.array([[.5, 0],
                                             [0, .5]]),
                            alpha=.5, beta=1, mu=.2,
                            dist_only=False)
        T = 200
    elif name == 'sthp1':
        synt = STHPDataset(s_mu=np.array([0, 0]), 
                            g0_cov=np.array([[5, 0],
                                            [0, 5]]),
                            g2_cov=np.array([[.1, 0],
                                            [0, .1]]),
                            alpha=.5, beta=.6, mu=.15,
                            dist_only=False)
        T = 200
    elif name == 'sthp2':
        synt = STHPDataset(s_mu=np.array([0, 0]), 
                            g0_cov=np.array([[1, 0],
                                            [0, 1]]),
                            g2_cov=np.array([[.1, 0],
                                            [0, .1]]),
                            alpha=.3, beta=2, mu=1,
                            dist_only=False)
        T = 200
    elif name == 'stscp0':
        synt = STSCPDataset(g0_cov=np.array([[1, 0],
                                             [0, 1]]),
                            g2_cov=np.array([[.85, 0],
                                             [0, .85]]),
                            alpha=.2, beta=.2, mu=1, gamma=0,
                            x_num=x_num, y_num=y_num,
                            max_history=100, dist_only=False)
        T = 100
    elif name == 'stscp1':
        synt = STSCPDataset(g0_cov=np.array([[.4, 0],
                                             [0, .4]]),
                            g2_cov=np.array([[.3, 0],
                                             [0, .3]]),
                            alpha=.3, beta=.2, mu=1, gamma=0,
                            x_num=x_num, y_num=y_num, lamb_max=4, 
                            max_history=100, dist_only=False)
        T = 100
    elif name == 'stscp2':
        synt = STSCPDataset(g0_cov=np.array([[.25, 0],
                                             [0, .25]]),
                            g2_cov=np.array([[.2, 0],
                                             [0, .2]]),
                            alpha=.4, beta=.2, mu=1, gamma=0,
                            x_num=x_num, y_num=y_num, lamb_max=4, 
                            max_history=100, dist_only=False)
        T = 100
    else:
        return None, 0.
        
    synt.load(f'data/raw/spatiotemporal/{name}.data', t_start=0, t_end=10000)
    return synt, T


class BaseSTPointProcess(pl.LightningModule):
    """Spatiotemporal Point Process Model"""
    
    def __init__(
        self,
        learning_rate: float = 0.004,
        step_size: int = 20,
        gamma: float = 0.5,
        name: str = 'sthp0',
        ## Visuailzation params
        start_idx: int = 2,
        vis_bounds: List[List[float]] = None,
        nsteps: List[int] = [101, 101, 201],
        round_time: bool = True,
        trunc: bool = False,
        max_history: int = 20,
        vis_batch_size: int = 8192
    ):
        """Spatiotemporal Point Process Model
        
        Parameters
        ----------
        learning_rate : float, optional
            Learning rate of STPP
        step_size : int, optional
            Scheduler step size
        gamma : float, optional
            Scheduler gamma
        name: str, optional
            Name of the dataset, plot intensity if synthetic
        start_idx: int optional
            The idx of sequence in test set to be plotted
        vis_bounds: List[List[float]], optional
            The 2x2 [[xmin, xmax], [ymin, ymax]] bounds for intensity visualization
        nsteps: List[int], optional
            The number of steps to visualize for each dimension (x, y, t)
        round_time: bool, optional
            Whether to round time range to integers between, then t_nstep will be ignored
        trunc: bool, optional
            Whether to truncate the history for intensity computation
        max_history: int, optional
            The maximum history length to truncate, ignored if trunc is False
        vis_batch_size: int, optional
            The batch size for intensity computation
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.st_x = []
        self.st_y = []
        self.st_x_cum = []
        self.st_y_cum = []
        
    def on_fit_start(self):
        logger.info(f'model.dtype: {self.dtype}')
        logger.info(self)
        self.project()

    def optimizer_step(
        self, 
        epoch: int, 
        batch_idx: int, 
        optimizer: Union[Optimizer, LightningOptimizer], 
        optimizer_closure: Optional[Callable[[], Any]] = None
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.project()

    def training_step(self, batch, batch_idx):
        st_x, st_y, _, _, _ = batch
        nll, sll, tll = self(st_x, st_y)
        
        if torch.isnan(nll):
            logger.error("Numerical error, quiting...")
            
        self.log('train_nll', nll.item())
        self.log('train_sll', sll.item())
        self.log('train_tll', tll.item())
        return nll

    def validation_step(self, batch, batch_idx):
        st_x, st_y, _, _, _ = batch
        nll, sll, tll = self(st_x, st_y)

        self.log('val_nll', nll.item())
        self.log('val_sll', sll.item())
        self.log('val_tll', tll.item())
        return nll

    def test_step(self, batch, batch_idx):
        st_x, st_y, st_x_cum, st_y_cum, loc = batch
        nll, sll, tll = self(st_x, st_y)
        mask = loc[0] == self.hparams.start_idx  # Plot the start_idx sequence only

        self.log('test_nll', nll.item())
        self.log('test_sll', sll.item())
        self.log('test_tll', tll.item())
        return {
            'loss': nll,
            'st_x': st_x[mask],
            'st_y': st_y[mask],
            'st_x_cum': st_x_cum[mask],
            'st_y_cum': st_y_cum[mask]
        }
        
    def on_test_batch_end(
        self, 
        outputs: Optional[Any], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        self.st_x.append(outputs['st_x'])
        self.st_y.append(outputs['st_y'])
        self.st_x_cum.append(outputs['st_x_cum'])
        self.st_y_cum.append(outputs['st_y_cum'])
        
    def on_test_epoch_end(self):
        device = self.st_x[0].device
        ## Stack ST outputs
        st_x = torch.cat(self.st_x, 0).cpu()
        st_y = torch.cat(self.st_y, 0).cpu()
        st_x_cum = torch.cat(self.st_x_cum, 0).cpu()
        st_y_cum = torch.cat(self.st_y_cum, 0).cpu()
        lambs = []
        
        st_x_cum_ = st_x_cum.clone()
        st_x_cum_[:, :, -1] = torch.tensor(np.diff(st_x_cum_[:, :, -1].numpy(), axis=1, prepend=0))
        scales = (st_x_cum_[0, 2, :] - st_x_cum_[0, 1, :]) / (st_x[0, 2, :] - st_x[0, 1, :])
        scales = scales.numpy()
        biases = st_x_cum_[0, 2, :] - st_x[0, 2, :] * scales
        biases = biases.numpy()
        
        if self.hparams.vis_bounds is None:
            ## Discretize space
            xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
        else:
            ## Normalize space
            bounds = np.array(self.hparams.vis_bounds)
            bounds = ((bounds.T - biases) / scales).T
            [xmin, xmax], [ymin, ymax] = bounds
        
        x_nstep, y_nstep, t_nstep = self.hparams.nsteps
        
        ############## Calculate synthetic intensity ##############
        synt, T = load_synt(self.hparams.name, x_nstep, y_nstep)
        his_st = st_y_cum.squeeze(1).detach().cpu().numpy()
        his_st[:, -1] += self.hparams.start_idx * T
        # Calculate the ground truth intensity
        t_start = his_st[0, -1]
        t_end = his_st[-1, -1]
        logger.info(f'Intensity time range : {t_start} ~ {t_end}')
        
        if self.hparams.round_time:
            t_num = int(t_end - t_start) + 1
        else:
            t_num = t_nstep
        t_range = torch.linspace(t_start, t_end, t_num)
        
        if synt is not None:
            lambs_gt, x_range, y_range, t_range = synt.get_lamb_st(x_num=x_nstep, y_num=y_nstep, 
                                                                   t_num=t_num, t_start=t_start, t_end=t_end)
            ## Normalize range
            x_range = (torch.Tensor(x_range) - biases[0]) / scales[0]
            y_range = (torch.Tensor(y_range) - biases[1]) / scales[1]
        else:
            x_range = torch.linspace(xmin, xmax, x_nstep)
            y_range = torch.linspace(ymin, ymax, y_nstep)
    
        ############## Calculate model intensity ##############
        lambs = self.calc_lamb(st_x, st_x_cum, st_y, st_y_cum, scales, biases,
                               x_range, y_range, t_range, device)
        
        ## Denormalize space
        x_range = x_range.numpy() * scales[0] + biases[0]
        y_range = y_range.numpy() * scales[1] + biases[1]
        
        cmin = 0.0
        if synt is not None:
            cmax = max(np.array(lambs_gt).max(), np.array(lambs).max())
            lambs_list = [lambs_gt, lambs]
            subplot_titles = ['Ground Truth', type(self).__name__]
        else:
            cmax = np.array(lambs).max()
            lambs_list = lambs
            subplot_titles = [type(self).__name__]
        
        ## For AutoInt: lambs, x_range, y_range, t_range, his_st_cum[:, :2], his_st_cum[:, 2]
        fig = plot_lambst_interactive(lambs_list, x_range, y_range, t_range, show=False,
                                      cmin=cmin, cmax=cmax,
                                      master_title=self.hparams.name,
                                      subplot_titles=subplot_titles)
        
        self.logger.experiment.track(Figure(fig), name='intensity', step=0, context={'subset': 'test'},)
        
    @abstractmethod
    def calc_lamb(self, st_x, st_x_cum, st_y, st_y_cum, scales, biases,
                  x_range, y_range, t_range, device):
        """Computing the intensity function over 3D [x, y, t] samples

        Parameters
        ----------
        st_x : torch.Tensor
            Normalized spatiotemporal sliding windows, N events, delta-time
        st_x_cum : torch.Tensor
            Normalized spatiotemporal sliding window, 1 event, delta-time
        st_y : torch.Tensor
            Spatiotemporal sliding window, N events, cumulative-time
        st_y_cum : torch.Tensor
            Spatiotemporal sliding window, 1 event, cumulative-time
        scales : np.ndarray
            Normalization scales
        biases : np.ndarray
            Normalization biases
        x_range : torch.Tensor
            Normalized x range for intensity calculation
        y_range : torch.Tensor
            Normalized y range for intensity calculation
        t_range : torch.Tensor
            t range for intensity calculation
        device : torch.device
        """
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, 
                                                    gamma=self.hparams.gamma)
        return [optimizer], [scheduler]
