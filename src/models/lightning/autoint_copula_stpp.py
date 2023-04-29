import torch
from models.lightning.autoint_stpp import AutoIntSTPointProcess
from integration.autoint import Cuboid, SumNet, MultSequential, CatNet, Prod, act_dict


class AutoIntCopulaSTPointProcess(AutoIntSTPointProcess):
    
    def __init__(
        self, 
        **kwargs  # for BaseSTPointProcess
    ) -> None:
        """AutoInt Point Process with Linear Copula
        """ 
        super().__init__(**kwargs)
        self.save_hyperparameters()
        
        act = act_dict[self.hparams.activation]
        L_prod_nets = [MultSequential(CatNet(bias=True, neg=True, num_layers=self.hparams.num_layers, 
                                             hidden_size=self.hparams.hidden_size, activation=act), 
                                      torch.nn.Linear(3, 3), 
                                      Prod())
                        for _ in range(self.hparams.n_prodnet)]
        M_prod_nets = [MultSequential(CatNet(bias=True, neg=False, num_layers=self.hparams.num_layers, 
                                             hidden_size=self.hparams.hidden_size, activation=act), 
                                      torch.nn.Linear(3, 3), 
                                      Prod())
                        for _ in range(self.hparams.n_prodnet)]
        cuboid = Cuboid(L=SumNet(*L_prod_nets), M=SumNet(*M_prod_nets))
        
        # log background intensity
        self.background = torch.nn.Parameter(torch.ones(1))
        # ∫_0^t λ
        self.F = cuboid
