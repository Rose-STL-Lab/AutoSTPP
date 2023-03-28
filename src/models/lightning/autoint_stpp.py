import torch
from models.lightning.stpp import BaseSTPointProcess
from integration.autoint import Cuboid, SumNet, ProdNet
from loguru import logger


class AutoIntSTPointProcess(BaseSTPointProcess):
    
    def __init__(
        self, 
        n_prodnet: int = 2,
        learning_rate: float = 0.004,
    ) -> None:
        """AutoInt Point Process

        Parameters
        ----------
        n_prodnet : int
            Number of ProdNet layers in the Cuboid L and M
            Increasing number of ProdNet improves model expressiveness at the cost of training time
        learning_rate : float, optional
            Learning rate of Cuboid
        """ 
        super().__init__()
        self.save_hyperparameters()
        
        L_prod_nets = [ProdNet(out_dim=1, bias=True, neg=True) 
                        for _ in range(self.hparams.n_prodnet)]
        M_prod_nets = [ProdNet(out_dim=1, bias=True) for _ in range(self.hparams.n_prodnet)]
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
            logger.debug('-' * 79)
            raise ValueError('Negative intensities.')

        sll = ll - tll
        return -ll, sll, tll

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
        st_x, st_y, _, _, _ = batch
        nll, sll, tll = self(st_x, st_y)

        self.log('test_nll', nll.item())
        self.log('test_sll', sll.item())
        self.log('test_tll', tll.item())
        return nll

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
