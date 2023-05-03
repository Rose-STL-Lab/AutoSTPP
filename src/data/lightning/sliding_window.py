import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.data import SlidingWindowWrapper
from download_data import download
from loguru import logger


class SlidingWindowDataModule(pl.LightningDataModule):
    
    def __init__(
        self, 
        name: str = 'sthp0', 
        data_dir: str = "./data/spatiotemporal/",
        option: str = "ready", 
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
        option : str, optional
            whether to download the data or use the ready data, by default 'ready'
        batch_size : int, optional
            batch size
        num_workers : int, optional
            number of workers for dataloader
        """        
        super().__init__()
        self.save_hyperparameters()
        
        if self.hparams.option == 'ready':
            pass
        elif self.hparams.option == 'download':
            self.download_data()
        else:
            raise ValueError(f"option {self.hparams.option} not supported")
        try:
            self.validate_data()
        except AssertionError:
            logger.error(f"Data not found at {self.hparams.data_dir}. "
                          "Please run the data module with download or generate option, "
                          "or check if you have specified the correct data directory.")
            raise AssertionError
        
    def validate_data(self):
        assert os.path.exists(os.path.join(self.hparams.data_dir, f'{self.hparams.name}.npz'))
        
    def download_data(self):
        # Hard-coded
        download('data/spatiotemporal/')
        download('data/raw/')

    def setup(self, stage: str):
        """Assign train/val datasets for use in dataloaders""" 
        npz = np.load(os.path.join(self.hparams.data_dir, f'{self.hparams.name}.npz'), allow_pickle=True)
    
        # Load all dataset in any stage (for using the same normalization)
        self.sliding_train = SlidingWindowWrapper(npz['train'], normalized=True, device="cpu")
        self.sliding_val = SlidingWindowWrapper(npz['val'], normalized=True, min=self.sliding_train.min, 
                                                max=self.sliding_train.max, device="cpu")
        self.sliding_test = SlidingWindowWrapper(npz['test'], normalized=True, min=self.sliding_train.min, 
                                                 max=self.sliding_train.max, device="cpu")

    def train_dataloader(self):
        return DataLoader(
            self.sliding_train, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            shuffle=True,
            persistent_workers=True, 
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.sliding_val, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            shuffle=False,
            persistent_workers=True, 
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.sliding_test, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            shuffle=False,
            persistent_workers=True, 
            pin_memory=True
        )

    def predict_dataloader(self):
        return None
