import torch
import numpy as np
from models.lightning.autoint_stpp import AutoIntSTPointProcess
from integration.autoint import Cuboid, SumNet, ProdNet, act_dict


class AutoIntSTPointProcessGauss(AutoIntSTPointProcess):
    
    def __init__(
        self, 
        **kwargs  # for AutoIntSTPointProcess
    ) -> None:
        """AutoInt Point Process with Gaussian background intensity
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
        self.background = torch.nn.Parameter(torch.ones(1) * 3.)
        self.background_mu = torch.nn.Parameter(torch.ones(2) * .5)
        self.background_sigma = torch.nn.Parameter(torch.ones(2) * -5.)
        
        # ∫_0^t λ
        self.F = cuboid

    def calc_background(self, s_y):
        lamb_t = torch.exp(self.background)
        s_diff_background = self.background_mu.unsqueeze(0) - s_y
        inv_var = torch.exp(-self.background_sigma).diag()
        exponent = -0.5 * torch.sum(s_diff_background @ inv_var * s_diff_background, -1)
        det_inv_var = 1 / (inv_var[0, 0] * inv_var[1, 1])
        norm = 1 / (2 * np.pi * torch.sqrt(det_inv_var))
        background_pdf = norm * torch.exp(exponent)
        return lamb_t * background_pdf
