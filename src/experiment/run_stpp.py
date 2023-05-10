import torch
from pytorch_lightning.cli import ArgsType

from data.lightning.sliding_window import SlidingWindowDataModule
from models.lightning.stpp import BaseSTPointProcess
from cli import MyLightningCLI
from utils import find_ckpt_path, increase_u_limit


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
    increase_u_limit()
    # cli.model = cli.model.load_from_checkpoint(
    #     find_ckpt_path('b7443a', aim_path='.blackhole/.aim'),
    #     **cli.model.hparams
    # )
    # cli.trainer.test(cli.model, cli.datamodule)
    cli.trainer.logger.log_hyperparams({'seed': cli.config['seed_everything']})
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule, ckpt_path='best')
