import torch
from pytorch_lightning.cli import ArgsType

from data.lightning.toy3d import Toy3dDataModule
from models.lightning.cuboid import BaseCuboid
from cli import MyLightningCLI


def cli_main(args: ArgsType = None):
    torch.set_float32_matmul_precision('medium')
    cli = MyLightningCLI(
        BaseCuboid, 
        Toy3dDataModule, 
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
    #     '.aim/cuboid/a51568a160c946c99c8bea74/checkpoints/epoch=499-step=4000.ckpt',
    # )
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule)
