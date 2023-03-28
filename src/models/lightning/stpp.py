import torch
import pytorch_lightning as pl
from aim import Figure
from loguru import logger
from typing import Union, Optional, Callable, Any
from torch.optim.optimizer import Optimizer
from pytorch_lightning.core.optimizer import LightningOptimizer


class BaseSTPointProcess(pl.LightningModule):
    """Spatiotemporal Point Process Model"""
    
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        
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
