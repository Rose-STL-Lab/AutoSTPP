from pytorch_lightning.cli import LightningCLI
from lightning_fabric.accelerators import find_usable_cuda_devices
from loguru import logger


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        device = [3]
        # find_usable_cuda_devices(1)
        logger.info(f"Training device: {device}")
        parser.set_defaults(
            {
                "trainer.accelerator": "cuda", 
                "trainer.devices": device
            }
        )
