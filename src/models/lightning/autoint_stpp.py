import torch
import numpy as np
from models.lightning.stpp import BaseSTPointProcess
from integration.autoint import Cuboid, SumNet, ProdNet, act_dict
from loguru import logger
from tqdm import tqdm


class AutoIntSTPointProcess(BaseSTPointProcess):
    
    def __init__(
        self, 
        n_prodnet: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        activation: str = 'tanh',
        bias: bool = True,
        trunc: bool = False,
        **kwargs  # for BaseSTPointProcess
    ) -> None:
        """AutoInt Point Process

        Parameters
        ----------
        n_prodnet : int
            Number of ProdNet layers in the Cuboid L and M
            Increasing number of ProdNet improves model expressiveness at the cost of training time
        num_layers : int
            Number of layers in each ProdNet component
        hidden_size : int
            Hidden size of each ProdNet component
        activation : str
            Activation function of each ProdNet component
        bias : bool
            Whether to use bias in each ProdNet component
        trunc: bool, optional
            Whether to truncate the history for intensity computation
        """ 
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.create()
        
    def create(self):
        """
        Create the model
        """
        act = act_dict[self.hparams.activation]
        L_prod_nets = [ProdNet(out_dim=1, bias=self.hparams.bias, neg=True, activation=act, 
                               num_layers=self.hparams.num_layers, 
                               hidden_size=self.hparams.hidden_size) for _ in range(self.hparams.n_prodnet)]
        M_prod_nets = [ProdNet(out_dim=1, bias=self.hparams.bias, activation=act, 
                               num_layers=self.hparams.num_layers,
                               hidden_size=self.hparams.hidden_size) for _ in range(self.hparams.n_prodnet)]
        cuboid = Cuboid(L=SumNet(*L_prod_nets), M=SumNet(*M_prod_nets))
        
        # log background intensity
        self.background = torch.nn.Parameter(torch.ones(1))
        # ∫_0^t λ
        self.F = cuboid
        
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
        lambs_sum = torch.sum(lambs, -1) + self.calc_background(s_y)
        
        ########## Calculate temporal intensity ############
        # [batch, seq_len]
        lamb_t = self.F.lamb_t_stpp(s_x.view(-1, 2), 
                                    t_diff.view(-1, 1)).view([batch, seq_len])
        lamb_t = torch.sum(lamb_t, -1) + torch.exp(self.background)  # sum up all events' influence

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

        ######### Debugging numerical error ########
        if not torch.all(lambs_sum > 0):
            idx = torch.argmax(torch.isnan(torch.log(lambs_sum)).float())
            logger.debug(idx)
            logger.debug(t_diff[idx])
            logger.debug(lambs[idx])
            logger.debug(torch.sum(lambs[idx]))
            logger.debug(torch.log(torch.sum(lambs[idx])))
            logger.debug(lambs_sum)
            logger.debug(torch.log(lambs_sum))
            logger.debug('-' * 79)
            raise ValueError('Negative intensities.')

        sll = ll - tll
        return -ll, sll, tll
    
    def calc_lamb(self, st_x, st_x_cum, st_y, st_y_cum, scales, biases,
                  x_range, y_range, t_range, device):
        s_grids = torch.stack(torch.meshgrid(x_range, y_range, indexing='ij'), dim=-1).view(-1, 2).to(device)
        backgrounds = self.calc_background(s_grids)
        if type(backgrounds) is torch.Tensor:
            backgrounds = backgrounds.reshape(len(x_range), len(y_range)).cpu().detach().numpy()
        
        ## Convert to history
        his_st_cum = torch.vstack((st_x_cum[0], st_y_cum.squeeze())).numpy()

        lambs = []
        for t in tqdm(t_range):
            i = sum(his_st_cum[:-1, -1] <= t.item()) - 1  # Index of corresponding history events
            
            his_st_cum_ = his_st_cum[:i + 1]   # History events (unscaled)
            his_st_cum_ = torch.tensor(his_st_cum_).float().to(device)
            
            # st_x__ = st_x[:i + 1, 0].to(device)   # History events (scaled)
            st_x_ = his_st_cum_.clone()
            st_x_[1:, -1] = torch.diff(st_x_[:, -1])
            
            ## History events (scaled)
            st_x_ = (st_x_ - torch.tensor(biases).to(device)) / torch.tensor(scales).to(device) 
            
            # Truncate the history events
            if self.hparams.trunc:
                his_st_cum_ = his_st_cum_[-self.hparams.max_history:]
                st_x_ = st_x_[-self.hparams.max_history:]
            
            s_diff = s_grids.T.unsqueeze(-1) - st_x_[:, :-1].T.unsqueeze(-2)  # Spatial difference
            s_diff = s_diff.permute([1, 2, 0])  # [len(s_grids), len(his_st_cum_), 2]
            
            # assert torch.allclose(s_diff[0], s_grids[0] - st_x_[:, :-1])
            # assert torch.allclose(s_diff[1], s_grids[1] - st_x_[:, :-1])
            
            t_diff = t - his_st_cum_[:, -1]  # [len(his_st_cum_)]
            t_diff = torch.stack([t_diff] * len(s_grids), 0).unsqueeze(-1)  # [len(s_grids), len(his_st_cum_), 1]
            t_diff = t_diff / scales[-1]
            
            st_diff = torch.cat((s_diff, t_diff), -1)  # Spatiotemporal difference
            st_diff = st_diff.view(-1, 3)
            
            # Load in batches
            lamb = []
            batch_size = self.hparams.vis_batch_size
            for i in range(0, len(st_diff), batch_size):
                lamb.append(self.calc_f(st_diff[i:i + batch_size]).cpu().detach().numpy())
            lamb = np.concatenate(lamb, 0)
            lamb = lamb.reshape(len(s_grids), -1)
            lamb = lamb.sum(-1).reshape(len(x_range), len(y_range))
            lamb += backgrounds
            
            lambs.append(lamb / np.prod(scales))
            
        return np.array(lambs)

    def calc_f(self, st_diff):
        return self.F.forward(st_diff)

    def calc_background(self, s_grids):
        return torch.exp(self.background).item()
