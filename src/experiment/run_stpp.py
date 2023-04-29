import torch
from pytorch_lightning.cli import ArgsType

from data.lightning.sliding_window import SlidingWindowDataModule
from models.lightning.stpp import BaseSTPointProcess
from cli import MyLightningCLI


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
    # cli.model = cli.model.load_from_checkpoint(
    #     '.aim/autoint_stpp/4ba0f852cb74459fa3dbb0ab/checkpoints/epoch=49-step=950.ckpt',
    #     **cli.model.hparams
    # )
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule)
