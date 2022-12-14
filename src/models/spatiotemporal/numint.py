import torch
from torch import nn
from loguru import logger

from integration.autoint import Cuboid


class MonteCarloSTPPSameInfluence(nn.Module):

    def __init__(self, cuboid: Cuboid, device):
        super().__init__()
        self.device = device

        # log background intensity
        self.background = torch.nn.Parameter(torch.ones(1))

        # λ
        self.f = nn.Sequential(
            cuboid.M.to(device),
            nn.Softplus()
        )

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
        lambs = self.f.forward(st_diff.view(-1, 3)).view([batch, seq_len])
        lambs_sum = torch.sum(lambs, -1) + torch.exp(self.background)  # sum up all events' influence

        ########## Calculate temporal intensity ############
        N = 100
        rand_locs = torch.rand([N, *s_x.shape]).to(self.device) - s_x  # Random locations centered at s_x
        inp = torch.cat((rand_locs, t_diff.unsqueeze(0).unsqueeze(-1).repeat(N, 1, 1, 1)), -1)
        lamb_t = self.f(inp).mean(0).squeeze(-1)
        
        # lamb_t = self.f.lamb_t_stpp(s_x.view(-1, 2), t_diff.view(-1, 1)).view([batch, seq_len])
        lamb_t = torch.sum(lamb_t, -1) + torch.exp(self.background)  # sum up all events' influence

        ######### Calculate integral intensity ##########
        # [batch, seq_len]
        # cumulative intensity of every event
        ta = t_x_cum[:, -1:] - t_x_cum
        tb = t_diff
        rand_t = torch.rand([N, *ta.shape]).to(self.device) * (tb - ta) + ta
        rand_st = torch.cat((rand_locs, rand_t.unsqueeze(-1)), -1)
        lamb_ints = self.f(rand_st).mean(0).squeeze(-1) * (tb - ta)

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
