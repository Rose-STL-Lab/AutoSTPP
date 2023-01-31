from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from loguru import logger
from copy import deepcopy
from typing import List

from integration.autoint import Cuboid


class AutoIntSTPPSameInfluence(nn.Module):

    def __init__(self, cuboid: Cuboid, device):
        super().__init__()
        self.device = device

        # log background intensity
        self.background = torch.nn.Parameter(torch.ones(1))

        # ∫_0^t λ
        self.F = cuboid.to(device)

        self.project()

    def project(self):
        """
        Employ non-negative constraint
        """
        self.F.project()

    def forward(self, st_x, st_y):
        """
        Calculate NLL for a batch of sliding windows

        :param st_x: [batch, seq_len, 3], the event timings
        :param st_y: [batch, 1, 3], the time to forecast
        :return: nll: scalar, the average negative log likelihood
        """
        # Calculate spatiotemporal distance to previous events
        t_x_cum = torch.cumsum(st_x[..., -1], -1)  # [batch, seq_len]
        t_diff = t_x_cum[:, -1:] - t_x_cum + st_y[..., -1:, -1]  # [batch, seq_len]

        if not torch.all(t_diff >= 0):
            idx = torch.argmin(t_diff)
            logger.debug(t_diff[idx // t_diff.shape[1]])
            raise ValueError('Negative time difference.')

        s_x = st_x[..., :2]
        s_y = st_y[..., :2]
        s_diff = s_y - s_x   # [batch, seq_len, 2]
        st_diff = torch.cat([s_diff, t_diff.unsqueeze(-1)], -1)

        ########## Calculate intensity ############
        # [batch, seq_len]
        batch, seq_len, _ = st_diff.shape
        lambs = self.F.forward(st_diff.view(-1, 3)).view([batch, seq_len])
        
        # Sum up all events' influence
        lambs_sum = torch.sum(lambs, -1) + torch.exp(self.background)
        
        ########## Calculate temporal intensity ############
        # lamb_t = self.F.lamb_t_stpp(s_x.view(-1, 2), 
        #                             t_diff.view(-1, 1)).view([batch, seq_len])
        # lamb_t = torch.sum(lamb_t, -1) + torch.exp(self.background)  # sum up all events' influence
        
        N = 10
        rand_locs = torch.rand([N, *s_x.shape]).to(self.device) - s_x  # Random locations centered at s_x
        inp = torch.cat((rand_locs, t_diff.unsqueeze(0).unsqueeze(-1).repeat(N, 1, 1, 1)), -1)
        shape_ = inp.shape
        lamb_t = self.F.forward(inp.view(-1, 3)).view(shape_[:-1]).mean(0).squeeze(-1)
        lamb_t = torch.sum(lamb_t, -1) + torch.exp(self.background)  # sum up all events' influence
        
        # logger.debug(lamb_t)

        ######### Calculate integral intensity ##########
        # [batch, seq_len]
        # cumulative intensity of every event
        lamb_ints = self.F.int_lamb_stpp(s_x.view(-1, 2), (t_x_cum[:, -1:] - t_x_cum).view(-1, 1),
                                         t_diff.view(-1, 1)).view([batch, seq_len])
        # lamb_ints = torch.where(lamb_ints >= 0, lamb_ints, torch.zeros_like(lamb_ints))

        ######### Calculate loss ########
        lamb_ints = torch.sum(lamb_ints, -1)
        background_int = st_y[..., -1, -1] * torch.exp(self.background)
        lamb_ints += background_int  # Add background intensities' integral

        tll = torch.log(lamb_t).mean() - lamb_ints.mean()
        ll = torch.log(lambs_sum).mean() - lamb_ints.mean()

        if not torch.all(lambs_sum > 0):
            idx = torch.argmax(torch.isnan(torch.log(lambs_sum)).float())
            logger.debug(idx)
            logger.debug(t_diff[idx])
            logger.debug(lambs[idx])
            logger.debug(torch.sum(lambs[idx]))
            logger.debug(torch.log(torch.sum(lambs[idx])))
            logger.debug(lambs_sum)
            logger.debug(torch.log(lambs_sum))
            logger.debug('--------------------------------------------')
            raise ValueError('Negative intensities.')

        sll = ll - tll

        return -ll, sll, tll
    
    
class WeightedAutoIntSTPPSameInfluence(nn.Module):

    def __init__(self, cuboids: List[Cuboid], device):
        super().__init__()
        self.device = device

        # log background intensity
        self.background = torch.nn.Parameter(torch.ones(1))

        # ∫_0^t λ
        self.cuboids = cuboids
        self.weight_net = torch.nn.GRU(3, 128, 1, batch_first=True)
        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, len(cuboids)),
        )
        self.project()

    def project(self):
        """
        Employ non-negative constraint
        """
        for f in self.cuboids:
            f.project()

    def forward(self, st_x, st_y):
        """
        Calculate NLL for a batch of sliding windows

        :param st_x: [batch, seq_len, 3], the event timings
        :param st_y: [batch, 1, 3], the time to forecast
        :return: nll: scalar, the average negative log likelihood
        """
        # Calculate spatiotemporal distance to previous events
        t_x_cum = torch.cumsum(st_x[..., -1], -1)  # [batch, seq_len]
        t_diff = t_x_cum[:, -1:] - t_x_cum + st_y[..., -1:, -1]  # [batch, seq_len]

        if not torch.all(t_diff >= 0):
            idx = torch.argmin(t_diff)
            logger.debug(t_diff[idx // t_diff.shape[1]])
            raise ValueError('Negative time difference.')

        s_x = st_x[..., :2]
        s_y = st_y[..., :2]
        s_diff = s_y - s_x   # [batch, seq_len, 2]
        st_diff = torch.cat([s_diff, t_diff.unsqueeze(-1)], -1)
        
        ########## Calculate weight ############
        output = self.weight_net(st_x)[1][0]
        weights = torch.softmax(self.output_net(output), -1).unsqueeze(1)  # [batch, 1, n_kernel]

        ########## Calculate intensity ############
        # [batch, seq_len]
        batch, seq_len, _ = st_diff.shape
        lambs_n = [f.forward(st_diff.view(-1, 3)).view([batch, seq_len]) for f in self.cuboids]
        lambs_n = torch.stack(lambs_n, -1)  # [batch, seq_len, n_kernel]
        lambs = torch.sum(lambs_n * weights, -1)  # [batch, seq_len]
        
        lambs_sum = torch.sum(lambs, -1) + torch.exp(self.background)
        
        ########## Estimate temporal intensity ############
        def calc_lamb_t():
            with torch.no_grad():
                N = 10
                rand_locs = torch.rand([N, *s_x.shape]).to(self.device) - s_x  # Random locations centered at s_x
                inp = torch.cat((rand_locs, t_diff.unsqueeze(0).unsqueeze(-1).repeat(N, 1, 1, 1)), -1)
                shape_ = inp.shape
                lamb_t_n = [f.forward(inp.view(-1, 3)).view(shape_[:-1]).mean(0).squeeze(-1) for f in self.cuboids]
                lamb_t_n = torch.stack(lamb_t_n, -1)
                lamb_t = torch.sum(lamb_t_n * weights, -1)
                lamb_t = torch.sum(lamb_t, -1) + torch.exp(self.background)  # sum up all events' influence
                return lamb_t
        
        if self.training:
            lamb_t = calc_lamb_t()
        else:
            M = 10
            lamb_t = sum([calc_lamb_t() for _ in range(M)]) / M 
        
        ######### Calculate integral intensity ##########
        # [batch, seq_len]
        # cumulative intensity of every event
        lamb_ints_n = [f.int_lamb_stpp(s_x.view(-1, 2), (t_x_cum[:, -1:] - t_x_cum).view(-1, 1),
                                       t_diff.view(-1, 1)).view([batch, seq_len]) for f in self.cuboids]
        lamb_ints_n = torch.stack(lamb_ints_n, -1)
        lamb_ints = torch.sum(lamb_ints_n * weights, -1)

        ######### Calculate loss ########
        lamb_ints = torch.sum(lamb_ints, -1)
        background_int = st_y[..., -1, -1] * torch.exp(self.background)
        lamb_ints += background_int  # Add background intensities' integral

        tll = torch.log(lamb_t).mean() - lamb_ints.mean()
        ll = torch.log(lambs_sum).mean() - lamb_ints.mean()

        if not torch.all(lambs_sum > 0):
            idx = torch.argmax(torch.isnan(torch.log(lambs_sum)).float())
            logger.debug(idx)
            logger.debug(t_diff[idx])
            logger.debug(lambs[idx])
            logger.debug(torch.sum(lambs[idx]))
            logger.debug(torch.log(torch.sum(lambs[idx])))
            logger.debug(lambs_sum)
            logger.debug(torch.log(lambs_sum))
            logger.debug('--------------------------------------------')
            raise ValueError('Negative intensities.')

        sll = ll - tll

        return -ll, sll, tll


def calc_lamb(model, test_loader, device, scales=np.ones(3), biases=np.zeros(3),
              t_nstep=201, x_nstep=101, y_nstep=101, round_time=True,
              xmax=None, xmin=None, ymax=None, ymin=None, start_idx=2, trunc=False):
    """
    Calculate the uniformly sampled spatiotemporal intensity with a given
    number of spatiotemporal steps  
    """
    model.eval()
    
    # Aggregate data
    st_xs = []
    st_ys = []
    st_x_cums = []
    st_y_cums = []
    for data in test_loader:
        st_x, st_y, st_x_cum, st_y_cum, (idx, _) = data
        mask = idx == start_idx  # Get the 3rd sequence only
        st_xs.append(st_x[mask])
        st_ys.append(st_y[mask])
        st_x_cums.append(st_x_cum[mask])
        st_y_cums.append(st_y_cum[mask])

        if not torch.any(mask):
            continue
        
    # Stack the first sequence
    st_x = torch.cat(st_xs, 0).cpu()
    st_y = torch.cat(st_ys, 0).cpu()
    st_x_cum = torch.cat(st_x_cums, 0).cpu()
    st_y_cum = torch.cat(st_y_cums, 0).cpu()
    lambs = []
    
    # Discretize space
    if xmax is None:
        xmax = 1.0
        xmin = 0.0
        ymax = 1.0
        ymin = 0.0
    else:
        xmax = (xmax - biases[0]) / scales[0]
        xmin = (xmin - biases[0]) / scales[0]
        ymax = (ymax - biases[1]) / scales[1]
        ymin = (ymin - biases[1]) / scales[1]

    x_step = (xmax - xmin) / (x_nstep - 1)
    y_step = (ymax - ymin) / (y_nstep - 1)
    x_range = torch.arange(xmin, xmax + 1e-5, x_step)
    y_range = torch.arange(ymin, ymax + 1e-5, y_step) 
    s_grids = torch.stack(torch.meshgrid(x_range, y_range, indexing='ij'), dim=-1).view(-1, 2).to(device)
    
    # Discretize time
    t_start = st_y_cum[0, -1, -1].item()
    t_end = st_y_cum[-1, -1, -1].item()
    print(f'Intensity time range : {t_start} ~ {t_end}')
    t_step = (t_end - t_start) / (t_nstep - 1)
    if round_time:
        t_range = torch.arange(round(t_start), round(t_end) + 1e-5, t_step)
    else:
        t_range = torch.arange(t_start, t_end + 1e-5, t_step)
        
    # Calculate intensity
    background = model.background.unsqueeze(0).cpu().detach()
    
    # Convert to history
    his_st_cum = torch.vstack((st_x_cum[0], st_y_cum.squeeze())).numpy()

    for t in tqdm(t_range):
        i = sum(his_st_cum[:-1, -1] <= t.item()) - 1  # Index of corresponding history events
        
        his_st_cum_ = his_st_cum[:i + 1]   # History events (unscaled)
        his_st_cum_ = torch.tensor(his_st_cum_).float().to(device)
        
        # st_x__ = st_x[:i + 1, 0].to(device)   # History events (scaled)
        st_x_ = deepcopy(his_st_cum_)
        st_x_[1:, -1] = torch.diff(st_x_[:, -1])
        st_x_ = (st_x_ - torch.tensor(biases).to(device)) / torch.tensor(scales).to(device)  # History events (scaled)
        
        # Truncate the history events
        if trunc:
            his_st_cum_ = his_st_cum_[-20:]
            st_x_ = st_x_[-20:]
        
        s_diff = s_grids.T.unsqueeze(-1) - st_x_[:, :-1].T.unsqueeze(-2)  # Spatial difference
        s_diff = s_diff.permute([1, 2, 0])  # [len(s_grids), len(his_st_cum_), 2]
        
        # assert torch.allclose(s_diff[0], s_grids[0] - st_x_[:, :-1])
        # assert torch.allclose(s_diff[1], s_grids[1] - st_x_[:, :-1])
        
        t_diff = t - his_st_cum_[:, -1]  # [len(his_st_cum_)]
        t_diff = torch.stack([t_diff] * len(s_grids), 0).unsqueeze(-1)  # [len(s_grids), len(his_st_cum_), 1]
        t_diff = t_diff / scales[-1]
        
        st_diff = torch.cat((s_diff, t_diff), -1)  # Spatiotemporal difference
        
        temp = st_diff.view(-1, 3)
        
        # Load in batches
        batch_size = 8192
        lamb = []
        for i in range(0, len(temp), batch_size):
            if isinstance(model, AutoIntSTPPSameInfluence):
                lamb.append(model.F.forward(temp[i:i + batch_size]).cpu().detach().numpy())
            elif isinstance(model, WeightedAutoIntSTPPSameInfluence):
                output = model.weight_net(st_x_)[1][0]
                weights = torch.softmax(model.output_net(output), -1).unsqueeze(0).unsqueeze(1)  # [1, 1, n_kernel]
                lamb_n = [f.forward(temp[i:i + batch_size]) for f in model.cuboids]
                lamb.append(torch.sum(torch.stack(lamb_n, -1) * weights, -1).cpu().detach().numpy())
        lamb = np.concatenate(lamb, 0)
        lamb = lamb.reshape(len(s_grids), -1)
        lamb = lamb.sum(-1).reshape(x_nstep, y_nstep) + np.exp(background.item())
        
        lambs.append(lamb / np.prod(scales))

    x_range = x_range.numpy() * scales[0] + biases[0]
    y_range = y_range.numpy() * scales[1] + biases[1]
    t_range = t_range.numpy()

    return lambs, x_range, y_range, t_range, his_st_cum[:, :2], his_st_cum[:, 2]
