import torch
from pytorch_lightning.cli import ArgsType

from data.lightning.sliding_window import SlidingWindowDataModule
from models.lightning.stpp import BaseSTPointProcess
from cli import MyLightningCLI
from utils import find_ckpt_path


def cli_main(args: ArgsType = None):
    torch.set_float32_matmul_precision('medium')
    cli = MyLightningCLI(
        BaseSTPointProcess, 
        SlidingWindowDataModule, 
        subclass_mode_model=True, 
        subclass_mode_data=True,
        save_config_callback=None,
        run=False,
        args=args,
    )
    return cli


if __name__ == '__main__':
    cli = cli_main()
    torch.multiprocessing.set_sharing_strategy('file_system')
    # cli.model = cli.model.load_from_checkpoint(
    #     find_ckpt_path('cb6465'),    # Copula-STHP0
    #     # find_ckpt_path('5ca538'),    # Auto-STSCP0
    #     # find_ckpt_path('86363e'),    # Auto-Earthquake
    #     # find_ckpt_path('e855a0'),    # Monte-STSCP0
    #     **cli.model.hparams
    # )
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule)
