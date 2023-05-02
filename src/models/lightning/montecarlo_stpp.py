import torch
from models.lightning.autoint_stpp import AutoIntSTPointProcess
from integration.autoint import Exp, act_dict
from loguru import logger
from torch import nn


class MonteCarloSTPointProcess(AutoIntSTPointProcess):
    
    def __init__(
        self, 
        N: int = 100,
        **kwargs
    ) -> None:
        """Monte Carlo Point Process
        
        Parameters
        ----------
        N: int, optional
            Number of random locations to sample to estimate intensity integral
        """ 
        super().__init__(**kwargs)
        self.save_hyperparameters()
        
    def create(self):
        """
        Create the model
        """
        # log background intensity
        self.background = torch.nn.Parameter(torch.ones(1))
        
        # λst
        act = act_dict[self.hparams.activation]
        self.f = [nn.Linear(3, self.hparams.hidden_size, self.hparams.bias), act]
        for _ in range(self.hparams.num_layers - 1):
            self.f += [nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size, 
                                 self.hparams.bias), act]
        self.f += [nn.Linear(self.hparams.hidden_size, 1, self.hparams.bias), Exp()]
        self.f = nn.Sequential(*self.f)
        
    def project(self):
        """
        Don't need non-negative constraint
        """
        pass
        
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
        st_diff = st_diff.view(-1, 3)
        background = self.calc_background(st_diff)
        lambs = self.f.forward(st_diff).view([batch, seq_len])
        lambs_sum = torch.sum(lambs, -1) + background  # sum up all events' influence

        ########## Calculate temporal intensity ############
        N = self.hparams.N
        rand_locs = torch.rand([N, *s_x.shape]).to(self.device) - s_x  # Random locations centered at s_x
        inp = torch.cat((rand_locs, t_diff.unsqueeze(0).unsqueeze(-1).repeat(N, 1, 1, 1)), -1)
        lamb_t = self.f.forward(inp).mean(0).squeeze(-1)
        
        # lamb_t = self.f.lamb_t_stpp(s_x.view(-1, 2), t_diff.view(-1, 1)).view([batch, seq_len])
        lamb_t = torch.sum(lamb_t, -1) + background  # sum up all events' influence

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
        background_int = st_y[..., -1, -1] * background
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

    def calc_f(self, st_diff):
        return self.f.forward(st_diff)
