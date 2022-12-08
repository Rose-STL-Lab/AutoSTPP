import pytest

from autoint_mlp import model, cuboid
from conftest import get_params, relpath, update_params, log_config, put_result, plot_training_progress


@pytest.fixture(
    scope="module",
    params=get_params('dataloader')
)
def dataloader(device, request):
    import numpy as np
    from data.data import SlidingWindowWrapper
    from torch.utils.data import DataLoader
    
    name = request.param['name']
    npz = np.load(f'data/spatiotemporal/{name}.npz', allow_pickle=True)
    update_params('dataloader', request)
    
    if name == 'covid_nj_cases':
        batch_size = 128

    train_set = SlidingWindowWrapper(npz['train'], normalized=True)
    val_set = SlidingWindowWrapper(npz['val'], normalized=True, min=train_set.min, max=train_set.max)
    test_set = SlidingWindowWrapper(npz['test'], normalized=True, min=train_set.min, max=train_set.max)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


@pytest.fixture(
    scope="class",
    params=get_params('trained_model')
)
def trained_model(cuboid, dataloader, device, request):
    import torch
    from loguru import logger
    import os
    
    from utils import AverageMeter, tenumerate, eval_loss
    from models.st_model import AutoIntSTPPSameInfluence
    
    update_params('trained_model', request)
    model_fn = relpath('models') + '.pkl'
    log_config()
    train_loader, val_loader, test_loader = dataloader
    
    model_name = request.param['name']
    if model_name == 'auto-stpp':
        model = AutoIntSTPPSameInfluence(cuboid, device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=request.param['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    
    sll_meter = AverageMeter()
    tll_meter = AverageMeter()
    loss_meter = AverageMeter()
    
    train_losses = []
    val_losses = []
    
    model = model.to(device)
    model_fn = relpath('models') + '.pkl'
    if not request.param['retrain']:  # try to use the previous trained model
        if os.path.exists(model_fn):
            model.load_state_dict(torch.load(model_fn)['model_state_dict'])
            train_losses = torch.load(model_fn)['train_losses']
            val_losses = torch.load(model_fn)['val_losses']
            logger.info('Previous model found and loaded.')
            model.eval()
            return model, train_losses, val_losses
        else:
            logger.info('Previous model not found. Retraining...')

    for epoch in range(request.param['n_epoch']):
        train_loss = []
        
        model.train()
        for index, data in tenumerate(train_loader):
            st_x, st_y, _, _, _ = data
            optimizer.zero_grad()
            loss, sll, tll = model(st_x, st_y)

            if torch.isnan(loss):
                logger.error("Numerical error, quiting...")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), request.param['gradient_clip'])
            optimizer.step()

            # Project to feasible set
            model.project()

            loss_meter.update(loss.item())
            sll_meter.update(sll.mean().item())
            tll_meter.update(tll.mean().item())
            
            train_loss.append(loss.item())

        train_losses.append(sum(train_loss) / len(train_loss))
        logger.debug("In Epoch {} | "
                     "total loss: {:5f} | Space: {:5f} | Time: {:5f}".format(
            epoch, loss_meter.avg, sll_meter.avg, tll_meter.avg
        ))
        scheduler.step()

        if (epoch + 1) % request.param['n_eval_epoch'] == 0:
            model.eval()
            val_loss, val_s, val_t = eval_loss(model, val_loader)
            val_losses.append(val_loss)
            
            logger.info("Evaluate   | Val Loss {:5f} | Space: {:5f} | Time: {:5f}".format(val_loss, val_s, val_t))
            if val_loss == min(val_losses):   
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                }, model_fn)
            
    model.load_state_dict(torch.load(model_fn)['model_state_dict'])
    
    logger.info("training done!")
    
    return model, train_losses, val_losses


class TestClass:

    def test_result(self, dataloader, trained_model):
        from loguru import logger
        from utils import eval_loss
        
        train_loader, val_loader, test_loader = dataloader
        model, train_losses, val_losses = trained_model
        model.eval()
        
        plot_training_progress(train_losses, val_losses, f"{relpath('figs', True)}/training")
        test_loss, test_s, test_t = eval_loss(model, test_loader)
        logger.critical("Evaluate   | Val Loss {:5f} | Space: {:5f} | Time: {:5f}".format(test_loss, test_s, test_t))
