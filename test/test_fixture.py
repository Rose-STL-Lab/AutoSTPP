from autoint_mlp import add_cuboid, cuboid, model, cat_linear_model
from loguru import logger


def test_model(device, model):
    import torch
    logger.debug(model)
    x = torch.rand(1, 3).to(device)
    logger.debug(model(x))


def test_cat_linear_model(device, cat_linear_model):
    import torch
    logger.debug(cat_linear_model)
    x = torch.rand(1, 3).to(device)
    logger.debug(cat_linear_model(x))


def test_cuboid(device, cuboid):
    import torch
    logger.debug(cuboid)
    x = torch.rand(1, 3).to(device)
    logger.debug(cuboid(x))


def test_add_cuboid(device, add_cuboid):
    import torch
    logger.debug(add_cuboid)
    x = torch.rand(1, 3).to(device)
    logger.debug(add_cuboid(x))
