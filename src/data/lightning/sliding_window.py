import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.data import SlidingWindowWrapper


class SlidingWindowDataModule(pl.LightningDataModule):
    
    def __init__(
        self, 
        name: str = 'sthp0', 
        data_dir: str = "./data/spatiotemporal/",
        batch_size: int = 128,
        num_workers: int = 8
    ):
        """Sliding Window Lightning Data Module

        Parameters
        ----------
        name : str, optional
            dataset name, by default spatiotemporal Hawkes, one of the following options:
            - sthp0
            - sthp1
            - sthp2
            - stscp0
            - stscp1
            - stscp2
            - earthquakes_jp
            - covid_nj_cases
            
        data_dir : str, optional
            directory that stores the sequence npz data, by default "./data/spatiotemporal/"
        batch_size : int, optional
            batch size
        num_workers : int, optional
            number of workers for dataloader
        """        
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        """TODO: Download the data"""
        assert os.path.exists(os.path.join(self.hparams.data_dir, f'{self.hparams.name}.npz'))
        assert os.path.exists(os.path.join(self.hparams.data_dir, f'{self.hparams.name}.npz'))

    def setup(self, stage: str):
        """Assign train/val datasets for use in dataloaders""" 
        npz = np.load(os.path.join(self.hparams.data_dir, f'{self.hparams.name}.npz'), allow_pickle=True)
    
        if stage == "fit":
            self.sliding_train = SlidingWindowWrapper(npz['train'], normalized=True, device="cpu")
            self.sliding_val = SlidingWindowWrapper(npz['val'], normalized=True, device="cpu")

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.sliding_test = SlidingWindowWrapper(npz['test'], normalized=True, device="cpu")

        if stage == "predict":
            self.sliding_predict = SlidingWindowWrapper(npz['test'], normalized=True, device="cpu")

    def train_dataloader(self):
        return DataLoader(
            self.sliding_train, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.sliding_val, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.sliding_test, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            shuffle=False
        )

    def predict_dataloader(self):
        return None
