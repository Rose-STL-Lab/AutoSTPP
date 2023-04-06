from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter, TuneConfig
from ray.air import RunConfig
from loguru import logger
import mock
import os
from pathlib import Path
from pytorch_lightning.cli import LightningCLI
import torch
from data.lightning.toy3d import Toy3dDataModule
from models.lightning.cuboid import BaseCuboid


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.set_defaults(
            {
                "trainer.accelerator": "cuda", 
                "trainer.devices": "[0]"
            }
        )


def cli_main(args=None):
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


def train_model(config):
    with mock.patch("sys.argv", [""]):
        args = ["-c", "../configs/prodnet_cuboid_sine.yaml", 
                "--trainer.callbacks+=ray.tune.integration.pytorch_lightning.TuneReportCallback",
                "--trainer.callbacks.metrics={'loss': 'val_mse'}",
                "--trainer.callbacks.on=validation_end",
                "--trainer.enable_checkpointing=false"]
        os.chdir(str(Path(__file__).parent.parent))
        for k, v in config.items():
            args += [f"--{k}", str(v)]
        cli = cli_main(args)
        cli.trainer.fit(cli.model, cli.datamodule)


def tune_model():
    # hyperparameters to search
    hparams = {
        "model.init_args.learning_rate": tune.loguniform(0.0001, 0.01),
        "model.init_args.num_layers": tune.choice([2, 3]),
        "model.init_args.hidden_size": tune.choice([64, 128, 256])
    }

    # metrics to track: keys are displayed names and
    # values are corresponding labels defined in LightningModule
    metrics = {
        "loss": "val_mse"
    }

    # scheduler
    scheduler = ASHAScheduler(
        max_t=100,
        grace_period=1,
        reduction_factor=2
    )

    # progress reporter
    reporter = CLIReporter(
        parameter_columns={p: p.split('.')[-1] for p in hparams.keys()},
        metric_columns=list(metrics.keys())
    )
    
    # Main analysis
    trainable = tune.with_parameters(
        train_model
    )
    
    # Assign GPU resources
    trainable = tune.with_resources(
        train_model,
        {"cpu": 2, "gpu": 1}
    )
    
    tuner = tune.Tuner(
        trainable,
        run_config=RunConfig(
            name="cuboid",
            progress_reporter=reporter,
            log_to_file=True
        ),
        tune_config=TuneConfig(
            mode="min",
            metric="loss",
            num_samples=1,
            scheduler=scheduler
        ),
        param_space=hparams
    )

    result = tuner.fit()

    best_result = result.get_best_result("loss", "min", "all")
    logger.info("Best hyperparameters found were:")
    logger.info(best_result.config)
    logger.info("Corresponding metrics are:")
    logger.info(best_result.metrics)


if __name__ == "__main__":
    tune_model()
