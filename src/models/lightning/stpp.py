import numpy as np
import torch
import pytorch_lightning as pl
from aim import Figure, Image
from loguru import logger
from typing import Union, Optional, Callable, Any
from torch.optim.optimizer import Optimizer
from pytorch_lightning.core.optimizer import LightningOptimizer
from typing import List
from visualization.plotter import plot_lambst_interactive, plot_lambst_static
from data.synthetic import STHPDataset, STSCPDataset
from abc import abstractmethod
from utils import scale_ll


def load_synt(name: str, x_num: int, y_num: int):
    """
    Load hparams and data of synthetic data
    Return None if name is not recognized
    
    :param name: name of the synthetic dataset
    :param x_num: number of x grid points
    :param y_num: number of y grid points

    :return synt: the synthetic dataset
    :return t_range: the time range of the dataset
    :return cmax: recommended cmax for plotting
    :return bounds: x- and y- bounds for plotting
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
        cmax = 0.4
        bounds = {
            "x_min": -2.5,
            "x_max": 2.5,
            "y_min": -2.5,
            "y_max": 2.5
        }
    elif name == 'sthp1':
        synt = STHPDataset(s_mu=np.array([0, 0]), 
                            g0_cov=np.array([[5, 0],
                                            [0, 5]]),
                            g2_cov=np.array([[.1, 0],
                                            [0, .1]]),
                            alpha=.5, beta=.6, mu=.15,
                            dist_only=False)
        T = 200
        cmax = None
        bounds = {}
    elif name == 'sthp2':
        synt = STHPDataset(s_mu=np.array([0, 0]), 
                            g0_cov=np.array([[1, 0],
                                            [0, 1]]),
                            g2_cov=np.array([[.1, 0],
                                            [0, .1]]),
                            alpha=.3, beta=2, mu=1,
                            dist_only=False)
        T = 200
        cmax = None
        bounds = {}
    elif name == 'stscp0':
        synt = STSCPDataset(g0_cov=np.array([[1, 0],
                                             [0, 1]]),
                            g2_cov=np.array([[.85, 0],
                                             [0, .85]]),
                            alpha=.2, beta=.2, mu=1, gamma=0,
                            x_num=x_num, y_num=y_num,
                            max_history=100, dist_only=False)
        T = 100
        cmax = None
        bounds = {}
    elif name == 'stscp1':
        synt = STSCPDataset(g0_cov=np.array([[.4, 0],
                                             [0, .4]]),
                            g2_cov=np.array([[.3, 0],
                                             [0, .3]]),
                            alpha=.3, beta=.2, mu=1, gamma=0,
                            x_num=x_num, y_num=y_num, lamb_max=4, 
                            max_history=100, dist_only=False)
        T = 100
        cmax = None
        bounds = {}
    elif name == 'stscp2':
        synt = STSCPDataset(g0_cov=np.array([[.25, 0],
                                             [0, .25]]),
                            g2_cov=np.array([[.2, 0],
                                             [0, .2]]),
                            alpha=.4, beta=.2, mu=1, gamma=0,
                            x_num=x_num, y_num=y_num, lamb_max=4, 
                            max_history=100, dist_only=False)
        T = 100
        cmax = 3.5
        bounds = {}
    else:
        return None, 0., None, {}
        
    synt.load(f'data/raw/spatiotemporal/{name}.data', t_start=0, t_end=10000)
    return synt, T, cmax, bounds


class BaseSTPointProcess(pl.LightningModule):
    """Spatiotemporal Point Process Model"""
    
    def __init__(
        self,
        learning_rate: float = 0.004,
        step_size: int = 20,
        gamma: float = 0.5,
        name: str = 'sthp0',
        ## Visuailzation params
        start_idx: Union[int, List[int]] = 2,
        vis_bounds: List[List[float]] = None,
        nsteps: List[int] = [101, 101, 201],
        round_time: bool = True,
        max_history: int = 20,
        vis_batch_size: int = 8192,
        vis_type: List[str] = ['interactive']
    ):
        """ABSTRACT Spatiotemporal Point Process Model
        
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
        start_idx: Union[int, List[int]] optional
            The idx of sequence in test set to be plotted
        vis_bounds: List[List[float]], optional
            The 2x2 [[xmin, xmax], [ymin, ymax]] bounds for intensity visualization
        nsteps: List[int], optional
            The number of steps to visualize for each dimension (x, y, t)
        round_time: bool, optional
            Whether to round time range to integers between, then t_nstep will be ignored
        max_history: int, optional
            The maximum history length to truncate, ignored if trunc is False
        vis_batch_size: int, optional
            The batch size for intensity computation
        vis_type: List[str], optional
            The type of visualization, 'interactive' or 'static', can be neither or both
        """
        super().__init__()
        
        ## Input spatiotemporal X+y by sequence index
        self.st_x = {}
        self.st_y = {}
        self.st_x_cum = {}
        self.st_y_cum = {}
        
        self.scales = [1., 1., 1.]
        self.biases = [0., 0., 0.]
        self.early_stop = False
        
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
        
    def calc_norm(self, st_x, st_x_cum):
        st_x_cum = st_x_cum.clone().detach().cpu()
        st_x = st_x.detach().cpu()
        st_x_cum[:, :, -1] = torch.tensor(np.diff(st_x_cum[:, :, -1].numpy(), axis=1, prepend=0))
        self.scales = (st_x_cum[0, 2, :] - st_x_cum[0, 1, :]) / (st_x[0, 2, :] - st_x[0, 1, :])
        self.scales = self.scales.numpy()
        self.biases = st_x_cum[0, 2, :] - st_x[0, 2, :] * self.scales
        self.biases = self.biases.numpy()

    def training_step(self, batch, batch_idx):
        st_x, st_y, st_x_cum, _, _ = batch
        nll, sll, tll = self(st_x, st_y)
        
        if type(self.scales) is list:
            self.calc_norm(st_x, st_x_cum)
        nll_scaled, sll_scaled, tll_scaled = scale_ll(None, nll, sll, tll, self.scales)
        
        if torch.isnan(nll) and not self.early_stop:
            logger.error("Numerical error, quiting...")
            self.early_stop = True
            
        self.log('train_nll', nll_scaled.item())
        self.log('train_sll', sll_scaled.item())
        self.log('train_tll', tll_scaled.item())
        return nll

    def validation_step(self, batch, batch_idx):
        st_x, st_y, _, _, _ = batch
        nll, sll, tll = self(st_x, st_y)
        nll_scaled, sll_scaled, tll_scaled = scale_ll(None, nll, sll, tll, self.scales)

        self.log('val_nll', nll_scaled.item())
        self.log('val_sll', sll_scaled.item())
        self.log('val_tll', tll_scaled.item())
        return nll

    def test_step(self, batch, batch_idx):
        st_x, st_y, st_x_cum, st_y_cum, loc = batch
        nll, sll, tll = self(st_x, st_y)
        
        if type(self.scales) is list:
            self.calc_norm(st_x, st_x_cum)
        nll_scaled, sll_scaled, tll_scaled = scale_ll(None, nll, sll, tll, self.scales)
            
        self.log('test_nll', nll_scaled.item())
        self.log('test_sll', sll_scaled.item())
        self.log('test_tll', tll_scaled.item())
        
        st_x_dict = {}
        st_y_dict = {}
        st_x_cum_dict = {}
        st_y_cum_dict = {}
        
        indices = loc[0].unique()  # All sequence indices in this batch
        for idx in indices:
            idx = idx.item()
            mask = loc[0] == idx
            st_x_dict[idx] = st_x[mask]
            st_y_dict[idx] = st_y[mask]
            st_x_cum_dict[idx] = st_x_cum[mask]
            st_y_cum_dict[idx] = st_y_cum[mask]
            
        return {
            'loss': nll,
            'st_x': st_x_dict,
            'st_y': st_y_dict,
            'st_x_cum': st_x_cum_dict,
            'st_y_cum': st_y_cum_dict
        }
    
    def on_train_batch_start(self, batch, batch_idx):
        if self.early_stop:
            return -1
        
    def on_test_batch_end(
        self, 
        outputs: Optional[Any], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        indices = outputs['st_x'].keys()
        
        for idx in indices: 
            if idx not in self.st_x:
                self.st_x[idx] = []
                self.st_y[idx] = []
                self.st_x_cum[idx] = []
                self.st_y_cum[idx] = []
            
            self.st_x[idx].append(outputs['st_x'][idx])
            self.st_y[idx].append(outputs['st_y'][idx])
            self.st_x_cum[idx].append(outputs['st_x_cum'][idx])
            self.st_y_cum[idx].append(outputs['st_y_cum'][idx])
        
    def on_test_epoch_end(self):
        # if len(self.hparams.vis_type) == 0:
        #     return  # Skip visualization
        
        device = self.st_x[0][0].device
        indices = self.st_x.keys()
        hessians = []
        maes = []
        
        if type(self.hparams.start_idx) is int:
            start_idx = [self.hparams.start_idx]
        else:
            start_idx = self.hparams.start_idx
        
        if min(start_idx) > max(indices):
            logger.critical("No sequence to plot! The maximum start_idx is %d" % max(indices))
            
        ############## Loading synthetic dataset ##############
        x_nstep, y_nstep, t_nstep = self.hparams.nsteps
        synt, T, cmax, bounds = load_synt(self.hparams.name, x_nstep, y_nstep)
        # synt = None
        # T = 0
        
        # For each ST sequences
        for idx in indices:
            if synt is None and idx not in start_idx:
                continue  # Skip if not the idx to plot
            
            ## Stack ST inputs
            st_x = torch.cat(self.st_x[idx], 0).cpu()
            st_y = torch.cat(self.st_y[idx], 0).cpu()
            st_x_cum = torch.cat(self.st_x_cum[idx], 0).cpu()
            st_y_cum = torch.cat(self.st_y_cum[idx], 0).cpu()
            
            scales = self.scales
            biases = self.biases
            
            ############## Calculate synthetic intensity ##############
            his_st = st_y_cum.squeeze(1).detach().clone().cpu().numpy()
            his_st[:, -1] += idx * T
            
            t_start = his_st[0, -1]
            t_end = his_st[-1, -1]
            logger.info(f'Intensity time range : {t_start} ~ {t_end}')
            
            ## Round time ranges to nearest decimals
            if self.hparams.round_time:
                t_start = float(round(t_start))
                t_end = float(round(t_end))
                t_num = int(t_end - t_start) + 1
                if synt is None:
                    t_num = (t_num - 1) * (t_nstep // t_num) + 1
            else:
                t_num = t_nstep
            t_range = torch.linspace(t_start, t_end, t_num)
            
            if self.hparams.vis_bounds is not None:
                ## Normalize space
                [x_min, x_max], [y_min, y_max] = self.hparams.vis_bounds
                bounds = {
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max
                }
                    
            if synt is not None:
                lambs_gt, x_range, y_range, t_range = synt.get_lamb_st(x_num=x_nstep, y_num=y_nstep, 
                                                                       t_num=t_num, t_start=t_start, t_end=t_end,
                                                                       **bounds)
                ## Normalize range
                t_range -= idx * T
                x_range = (torch.Tensor(x_range) - biases[0]) / scales[0]
                y_range = (torch.Tensor(y_range) - biases[1]) / scales[1]
            else:
                ## Normalized range
                lambs_gt = None
                x_range = torch.linspace(0.0, 1.0, x_nstep)
                y_range = torch.linspace(0.0, 1.0, y_nstep)
        
            ############## Calculate model intensity ##############
            lambs = self.calc_lamb(st_x, st_x_cum, st_y, st_y_cum, scales, biases,
                                   x_range, y_range, t_range, device)
            
            ################### Compute Hessian ###################
            if synt is not None:
                for p, q in zip(lambs, lambs_gt):
                    p = p.flatten() / p.sum()
                    q = q.flatten() / q.sum()
                    dist = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
                    hessians.append(dist)
            
            #################### Compute λ MAE ####################
            if synt is not None:
                for p, q in zip(lambs, lambs_gt):
                    lamb_t = np.sum(p) / x_nstep / y_nstep
                    lamb_t_gt = np.sum(q) / x_nstep / y_nstep
                    maes.append(abs(lamb_t - lamb_t_gt))
                    
            ###################### Plotting #######################
            if idx not in start_idx:
                continue  # Skip if not the idx to plot
            
            ## Denormalize
            x_range = x_range.numpy() * scales[0] + biases[0]
            y_range = y_range.numpy() * scales[1] + biases[1]
            his_st[:, -1] -= idx * T
            
            if cmax is None:
                if lambs_gt is not None:
                    cmax = np.array([lambs_gt, lambs]).max()
                else:
                    cmax = np.array(lambs).max()
            
            if synt is not None:
                lambs_list = [lambs_gt, lambs]
                subplot_titles = ['Ground Truth', type(self).__name__]
            else:
                lambs_list = [lambs]
                subplot_titles = [type(self).__name__]
            
            if 'interactive' in self.hparams.vis_type:
                fig = plot_lambst_interactive(lambs_list if len(lambs_list) > 1 else lambs_list[0],
                                              x_range, y_range, t_range, show=False, cauto=True,
                                              master_title=self.hparams.name,
                                              subplot_titles=subplot_titles)
                self.logger.experiment.track(Figure(fig), name=f'intensity-{idx}', step=0, context={'subset': 'test'},)
            
            if 'static' in self.hparams.vis_type:
                for lambs, title in zip(lambs_list, subplot_titles):
                    fig = plot_lambst_static(lambs, x_range, y_range, t_range, history=(his_st[:, :-1], his_st[:, -1]),
                                             cmax=cmax, fps=12, fn=f'{title}.gif',)
                    self.logger.experiment.track(Image(f'{title}.gif'), name=f'static-{idx}-{title}', 
                                                 context={'subset': 'test'})

        if synt is not None:
            hessian = np.mean(hessians)
            mae = np.mean(maes)
            logger.info(f'λt Mean Absolute Error: {mae}')
            logger.info(f'Hessian distance: {hessian}')
            self.log('lamb_mae', mae)
            self.log('hessian', hessian)
    
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
            Unscaled & 0-shifted t range for intensity calculation
        device : torch.device
        """
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, 
                                                    gamma=self.hparams.gamma)
        return [optimizer], [scheduler]
