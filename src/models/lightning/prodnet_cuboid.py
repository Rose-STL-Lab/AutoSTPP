import torch
from models.lightning.cuboid import BaseCuboid
from integration.autoint import Cuboid, SumNet, Prod, MultSequential, CatNet, act_dict


class ProdnetCuboid(BaseCuboid):
    
    def __init__(
        self, 
        n_prodnet: int = 2,
        num_layers: int = 2,
        hidden_size: int = 128,
        activation: str = 'tanh',
        **kwargs  # for BaseCuboid
    ) -> None:
        """AutoInt Prodnet Cuboid

        Parameters
        ----------
        n_prodnet : int
            Number of ProdNet layers in the Cuboid
            Increasing number of ProdNet improves model expressiveness at the cost of training time
        num_layers : int
            Number of layers in each ProdNet component
        hidden_size : int
            Hidden size of each ProdNet component
        activation : str
            Activation function of each ProdNet component
        """ 
        super().__init__(**kwargs)
        self.save_hyperparameters()
        
        act = act_dict[self.hparams.activation]
        L_prod_nets = [MultSequential(CatNet(bias=True, neg=True, num_layers=self.hparams.num_layers, 
                                             hidden_size=self.hparams.hidden_size, activation=act), Prod())
                        for _ in range(self.hparams.n_prodnet)]
        M_prod_nets = [MultSequential(CatNet(bias=True, neg=False, num_layers=self.hparams.num_layers, 
                                             hidden_size=self.hparams.hidden_size, activation=act), Prod())
                        for _ in range(self.hparams.n_prodnet)]
        cuboid = Cuboid(L=SumNet(*L_prod_nets), M=SumNet(*M_prod_nets))
        self.cuboid = cuboid


class ProdnetLinearCuboid(BaseCuboid):
    
    def __init__(
        self, 
        n_prodnet: int = 2,
        num_layers: int = 2,
        hidden_size: int = 128,
        activation: str = 'relu',
        **kwargs  # for BaseCuboid
    ) -> None:
        """AutoInt Cuboid

        Parameters
        ----------
        n_prodnet : int
            Number of ProdNet layers in the Cuboid
            Increasing number of ProdNet improves model expressiveness at the cost of training time
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
        self.cuboid = cuboid
