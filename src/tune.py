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
import importlib
import yaml
import argparse


def load_class(full_class_string):
    class_data = full_class_string.split(".")
    module_path = ".".join(class_data[:-1])
    class_str = class_data[-1]
    module = importlib.import_module(module_path)
    return getattr(module, class_str)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.set_defaults(
            {
                "trainer.accelerator": "cuda", 
                "trainer.devices": "[0]"
            }
        )


def cli_main(args, model_config, data_config):
    torch.set_float32_matmul_precision('medium')
    cli = MyLightningCLI(
        load_class(f"{model_config['module']}.{model_config['class']}"), 
        load_class(f"{data_config['module']}.{data_config['class']}"),
        subclass_mode_model=True, 
        subclass_mode_data=True,
        save_config_callback=None,
        run=False,
        args=args,
    )
    return cli


def train_model(config, tune_config):
    with mock.patch("sys.argv", [""]):
        args = ["-c", tune_config['lightning']['config'],
                "--trainer.callbacks=ray.tune.integration.pytorch_lightning.TuneReportCallback",
                f"--trainer.callbacks.metrics={{'loss': '{tune_config['lightning']['loss']}'}}",
                "--trainer.callbacks.on=validation_end",
                "--trainer.enable_checkpointing=false",
                *tune_config['lightning']['extra_args']]
        print(args)
        os.chdir(str(Path(__file__).parent.parent))
        for k, v in config.items():
            args += [f"--{k}", str(v)]
        cli = cli_main(args, tune_config['model'], tune_config['data'])
        cli.trainer.fit(cli.model, cli.datamodule)


def tune_model(tune_config):
    # hyperparameters to search
    hparams = tune_config['hparams']
    for k, v in hparams.items():
        for func, args in v.items():
            if type(args) == list:
                hparams.update({k: getattr(tune, func)(args)})
            elif type(args) == dict:
                hparams.update({k: getattr(tune, func)(**args)})

    # metrics to track: keys are displayed names and
    # values are corresponding labels defined in LightningModule
    metrics = {
        "loss": tune_config['lightning']['loss']
    }

    # scheduler
    scheduler = ASHAScheduler(**tune_config['scheduler'])

    # progress reporter
    reporter = CLIReporter(
        parameter_columns={p: p.split('.')[-1] for p in hparams.keys()},
        metric_columns=list(metrics.keys())
    )
    
    # Main analysis
    train_with_paramters = lambda config: train_model(config, tune_config=tune_config)
    
    trainable = tune.with_parameters(
        train_with_paramters
    )
    
    # Assign GPU resources
    trainable = tune.with_resources(
        trainable,
        {k: v for k, v in tune_config['resources'].items()}
    )
    
    tuner = tune.Tuner(
        trainable,
        run_config=RunConfig(
            name=tune_config['experiment']['name'],
            progress_reporter=reporter,
            log_to_file=True
        ),
        tune_config=TuneConfig(
            mode="min",
            metric="loss",
            num_samples=tune_config['experiment']['num_samples'],
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        tune_config = yaml.safe_load(f)
        logger.info(tune_config)

    tune_model(tune_config)
