import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Callable, List
import torch
from utils import arange, tqdm, scale
from download_data import download
from scipy.integrate import quad, dblquad, tplquad
from loguru import logger
from tqdm.contrib.concurrent import thread_map


class Toy3D(torch.utils.data.Dataset):
    """
    Wrap X = (x, y, z), f(x, y, z), and
    F1 := ∫ f(x, y, z) dx, 
    F2 := ∫∫ f(x, y, z) dxdy, or
    F3 := ∫∫∫ f(x, y, z) dxdydz
    """
    def __init__(self, X, f, F1, F2, F3):
        self.X = X
        self.f = f
        self.F1 = F1
        self.F2 = F2
        self.F3 = F3

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.f[idx], self.F1[idx], self.F2[idx], self.F3[idx]


class Toy3dDataModule(pl.LightningDataModule):
    
    def __init__(
        self, 
        name: str = 'sine', 
        data_dir: str = "./data/test/test_autoint_3d_positive",
        batch_size: int = 128,
        test_batch_size: int = 8192,
        num_workers: int = 8,
        option: str = 'ready',
        sampling_intensity: int = 1024,
        grid_size: int = 50,
        force: bool = False,
    ):
        """Toy 3D dataset: given integrant f(x) and integral F(x) over space and time,
        fit the integral to find the integrant or
        fit the integrant to find the integral.

        Parameters
        ----------
        name : str, optional
            dataset name, by default sine, one of the following options:
            - sine
            - normal
            
        data_dir : str, optional
            directory that stores the Wrapper class pickle, by default "./data/test/test_autoint_3d_positive"
        batch_size : int, optional
            batch size for fitting
        test_batch_size : int, optional
            batch size for testing
        num_workers : int, optional
            number of workers for dataloader
        option : str, optional
            whether to download the data, generate the data, or use the ready data, by default 'ready'
        sampling_intensity : int, optional
            number of data points in train set, by default 1024, will be ignored when option is not 'generate'
        grid_size : int, optional
            number of grid points in each dimension for visualization, will be ignored when option is not 'generate'
        force : bool, optional
            whether to force regenerate all data, will be ignored when option is not 'generate'
        """        
        super().__init__()
        self.save_hyperparameters()
        if self.hparams.option == 'ready':
            pass
        elif self.hparams.option == 'download':
            self.download_data()
        elif self.hparams.option == 'generate':
            if name == 'sine':
                func_to_fit = lambda x, y, z: np.sin(x) * np.cos(y) * np.sin(z) + 1
                bounds = [[0., 3.], [0., 3.], [0., 3.]]
            elif name == 'normal':
                func_to_fit = lambda x, y, z: np.exp(-5. * x ** 2 - 5. * y ** 2) * np.exp(-z)
                bounds = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]
            self.generate_data(
                sampling_density=self.hparams.sampling_intensity,
                func_to_fit=func_to_fit,
                bounds=bounds,
                N=self.hparams.grid_size,
                force=self.hparams.force
            )
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
        assert os.path.exists(os.path.join(self.hparams.data_dir, f'{self.hparams.name}_train.pkl'))
        assert os.path.exists(os.path.join(self.hparams.data_dir, f'{self.hparams.name}_val.pkl'))
        assert os.path.exists(os.path.join(self.hparams.data_dir, f'{self.hparams.name}_test.pkl'))
    
    def download_data(self):
        # Hard-coded
        download('data/test/test_autoint_3d_positive/')
        
    def generate_data(
        self,
        sampling_density: int = 1024,
        func_to_fit: Callable = lambda x, y, z: np.sin(x) * np.cos(y) * np.sin(z) + 1,
        bounds: List[List] = [[0., 3.], [0., 3.], [0., 3.]],
        N: int = 50,
        force: bool = False
    ) -> None:
        """
        Generate the toy data, using the f(x) `func_to_fit` over the integral range `bounds`
        Number of integrals is `sampling_density`
        
        Parameters
        ----------
        sampling_density : int, optional
            Number of points (x) to sample, by default 1024
        func_to_fit : Callable, optional
            Function to fit, by default lambda x: np.sin(x)
        bounds : List[List], optional
            Integral X/Y/Z lower and upper limits, by default [[0., 3.], [0., 3.], [0., 3.]]
        N : int, optional
            Number of points to sample for the ground truth, by default 50
        force : bool, optional
            Force to regenerate the data, by default False
        """
        Xa = bounds[0][0]
        Ya = bounds[1][0]
        Za = bounds[2][0]
        
        def calc_integral(X):
            if type(X) != torch.Tensor:
                X = torch.tensor(X).float()
            F1 = thread_map(lambda x: quad(lambda x0: func_to_fit(x0, x[1], x[2]), Xa, x[0], epsabs=1e-6)[0], X)
            F2 = thread_map(lambda x: dblquad(lambda x1, x0: func_to_fit(x0, x1, x[2]), 
                                              Xa, x[0], Ya, x[1], epsabs=1e-6)[0], X)
            F3 = thread_map(lambda x: tplquad(lambda x2, x1, x0: func_to_fit(x0, x1, x2), 
                                              Xa, x[0], Ya, x[1], Za, x[2], epsabs=1e-6)[0], X)
            F1 = torch.tensor(F1).float()
            F2 = torch.tensor(F2).float()
            F3 = torch.tensor(F3).float()
            f = func_to_fit(X[:, 0], X[:, 1], X[:, 2]).float()
            return Toy3D(X, f, F1, F2, F3)
        
        train_path = os.path.join(self.hparams.data_dir, f'{self.hparams.name}_train.pkl')
        if not os.path.exists(train_path) or force:
            X = torch.rand(sampling_density, 3)
            X = scale(X, bounds)
            dataset = calc_integral(X)
            torch.save(dataset, train_path)
        else:
            dataset = torch.load(train_path, map_location=torch.device("cpu"))
            
        val_path = os.path.join(self.hparams.data_dir, f'{self.hparams.name}_val.pkl')
        if not os.path.exists(val_path) or force:
            X = torch.rand(sampling_density, 3)
            X = scale(X, bounds)
            dataset = calc_integral(X)
            torch.save(dataset, val_path)
        
        test_path = os.path.join(self.hparams.data_dir, f'{self.hparams.name}_test.pkl')
        if not os.path.exists(test_path) or force:
            X = arange(N, bounds)
            dataset = calc_integral(X)
            torch.save(dataset, test_path)

    def setup(self, stage: str):
        """
        No validation / test set, only setup the train set 
        """ 
        train_set = torch.load(os.path.join(self.hparams.data_dir, f'{self.hparams.name}_train.pkl'), 
                               map_location=torch.device("cpu"))
        val_set = torch.load(os.path.join(self.hparams.data_dir, f'{self.hparams.name}_val.pkl'),
                             map_location=torch.device("cpu"))
        test_set = torch.load(os.path.join(self.hparams.data_dir, f'{self.hparams.name}_test.pkl'),
                              map_location=torch.device("cpu"))
    
        if stage == "fit":
            self.toy_train = train_set
            self.toy_val = val_set

        if stage == "test":
            self.toy_test = test_set

    def train_dataloader(self):
        """
        https://medium.com/@florian-ernst/finding-why-pytorch-lightning-made-my-training-4x-slower-ae64a4720bd1
        """
        return DataLoader(
            self.toy_train, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            shuffle=True,
            persistent_workers=True, 
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.toy_val, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            shuffle=False,
            persistent_workers=True, 
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.toy_test, 
            batch_size=self.hparams.test_batch_size, 
            num_workers=self.hparams.num_workers,
            shuffle=False,
            persistent_workers=True, 
            pin_memory=True
        )

    def predict_dataloader(self):
        return None
