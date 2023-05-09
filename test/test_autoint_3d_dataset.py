import pytest

from autoint_mlp import model, cuboid
from conftest import get_params, relpath, update_params, log_config, wandb_init, wandb_discard


# DeepSTPP config
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
           
                
config = Namespace(hid_dim=128, emb_dim=128, out_dim=0, n_layers=1, s_max=None,
                   lr=0.0003, momentum=0.9, epochs=50, batch=128, opt='Adam', generate_type=True,
                   read_model=False, seq_len=20, eval_epoch=5, s_min=1e-4, b_max=20, 
                   lookahead=1, alpha=0.1, z_dim=128, beta=1e-3, dropout=0, num_head=2,
                   nlayers=3, num_points=20, infer_nstep=10000, infer_limit=13, clip=1.0,
                   constrain_b=False, sample=False, decoder_n_layer=3)


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
    elif name == 'earthquakes_jp':
        batch_size = 128
    elif name.startswith('sthp') or name.startswith('stscp'):
        batch_size = 128
    else:
        raise NotImplementedError
    
    train_set = SlidingWindowWrapper(npz['train'], normalized=True, device=device)
    val_set = SlidingWindowWrapper(npz['val'], normalized=True, min=train_set.min, max=train_set.max, device=device)
    test_set = SlidingWindowWrapper(npz['test'], normalized=True, min=train_set.min, max=train_set.max, device=device)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


@pytest.fixture(
    scope="class",
    params=get_params('trained_model')
)
def trained_model(model, cuboid, dataloader, device, request):
    import torch
    from loguru import logger
    import os
    import shutil
    import wandb
    
    from utils import AverageMeter, tenumerate, eval_loss, scale_ll
    from models.spatiotemporal.model import AutoIntSTPPSameInfluence, WeightedAutoIntSTPPSameInfluence
    from models.spatiotemporal.numint import MonteCarloSTPPSameInfluence
    from models.spatiotemporal.deepstpp import DeepSTPP, train
    from integration.autoint import Cuboid, SumNet, ProdNet
    
    update_params('trained_model', request)
    model_fn = relpath('models') + '.pkl'
    log_config()
    wandb_init(__file__)
    train_loader, val_loader, test_loader = dataloader
    
    model_name = request.param['name']
    lr = request.param['lr']
    dataset_name = pytest.fn_params['dataloader']['name']
    if dataset_name == 'covid_nj_cases':
        lr = 0.0002
    elif dataset_name == 'earthquakes_jp':
        lr = 0.004
    
    if model_name == 'auto-stpp':
        L_prod_nets = [ProdNet(out_dim=1, bias=True, neg=True) 
                        for _ in range(request.param['n_prodnet'])]
        M_prod_nets = [ProdNet(out_dim=1, bias=True) for _ in range(request.param['n_prodnet'])]
        cuboid = Cuboid(L=SumNet(*L_prod_nets), M=SumNet(*M_prod_nets)).to(device)
        model = AutoIntSTPPSameInfluence(cuboid, device=device)
        
    elif model_name == 'weight-auto-stpp':
        cuboids = torch.nn.ModuleList([])
        for _ in range(request.param['n_kernel']):
            L_prod_nets = [ProdNet(out_dim=1, bias=True, neg=True) 
                           for _ in range(request.param['n_prodnet'])]
            M_prod_nets = [ProdNet(out_dim=1, bias=True) for _ in range(request.param['n_prodnet'])]
            cuboids.append(Cuboid(L=SumNet(*L_prod_nets), M=SumNet(*M_prod_nets)).to(device))
        model = WeightedAutoIntSTPPSameInfluence(cuboids, device=device)
        
    elif model_name == 'monte-carlo-stpp':
        model.layers.append(torch.nn.ReLU())
        model = MonteCarloSTPPSameInfluence(model, device=device)
        
    elif model_name == 'deep-stpp':
        model = DeepSTPP(config, device)
        best_model = train(model, train_loader, val_loader, config, device)
        torch.save({'model_state_dict': model.state_dict()}, model_fn)
        return best_model, [], []
    else:
        raise NotImplementedError
    logger.info(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    
    train_losses = []
    val_losses = []
    
    model = model.to(device)
    model_fn = relpath('models') + '.pkl'
    if not request.param['retrain']:  # try to use the previous trained model
        if os.path.exists(model_fn):
            model.load_state_dict(torch.load(model_fn, map_location=device)['model_state_dict'])
            if model_name != 'deep-stpp':  
                train_losses = torch.load(model_fn, map_location=device)['train_losses']
                val_losses = torch.load(model_fn, map_location=device)['val_losses']
            else:
                train_losses = []
                val_losses = []
            logger.info('Previous model found and loaded.')
            model.eval()
            return model, train_losses, val_losses
        else:
            logger.info('Previous model not found. Retraining...')

    for epoch in range(request.param['n_epoch']):
        sll_meter = AverageMeter()
        tll_meter = AverageMeter()
        nll_meter = AverageMeter()
        
        model.train()
        for index, data in tenumerate(train_loader):
            st_x, st_y, _, _, _ = data
            optimizer.zero_grad()
            nll, sll, tll = model(st_x, st_y)

            if torch.isnan(nll):
                logger.error("Numerical error, quiting...")

            # sll.backward()
            nll.backward()
            optimizer.step()

            if model_name == 'auto-stpp' or model_name == 'weight-auto-stpp': 
                model.project()

            nll_meter.update(nll.item())
            sll_meter.update(sll.item())
            tll_meter.update(tll.item())

        train_nll, train_sll, train_tll = scale_ll(train_loader, nll_meter.avg, sll_meter.avg, tll_meter.avg)

        train_losses.append(train_nll)
        if train_nll < 10:
            wandb.log({'train_nll': train_nll, 'train_sll': train_sll, 'train_tll': train_tll}, step=epoch)
        if torch.isnan(train_nll):
            id = wandb.run.id
            wandb.finish()
            wandb_discard(id)
            raise ValueError('NaN loss')
        
        logger.debug("In Epoch {} | total loss: {:5f} | Space: {:5f} | Time: {:5f}".format(
            epoch, train_nll, train_sll, train_tll
        ))
        scheduler.step()

        if (epoch + 1) % request.param['n_eval_epoch'] == 0:
            model.eval()
            
            val_nll, val_sll, val_tll = eval_loss(model, val_loader)
            val_nll, val_sll, val_tll = scale_ll(val_loader, val_nll, val_sll, val_tll)
            
            val_losses.append(val_nll)
            wandb.log({'val_nll': val_nll, 'val_sll': val_sll, 'val_tll': val_tll}, step=epoch)
            
            logger.info("Evaluate   | Val Loss {:5f} | Space: {:5f} | Time: {:5f}".format(val_nll, val_sll, val_tll))
            if val_nll == min(val_losses):
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                }, model_fn)
            
    model.load_state_dict(torch.load(model_fn)['model_state_dict'])
    if isinstance(wandb.run, wandb.sdk.wandb_run.Run):
        shutil.copy(model_fn, f'{model_fn[:-4]}-{wandb.run.name}.pkl')
    
    logger.info("training done!")
    
    return model, train_losses, val_losses


class TestClass:

    def test_result(self, dataloader, trained_model):
        from loguru import logger
        from utils import eval_loss, scale_ll
        import wandb
        from models.spatiotemporal.deepstpp import DeepSTPP
        
        train_loader, val_loader, test_loader = dataloader
        model, train_losses, val_losses = trained_model
        model.eval()
        
        if isinstance(model, DeepSTPP):
            from models.spatiotemporal.deepstpp import eval_loss
            test_nll, test_sll, test_tll = eval_loss(model, test_loader)
        else:
            test_nll, test_sll, test_tll = eval_loss(model, test_loader)
        test_nll, test_sll, test_tll = scale_ll(test_loader, test_nll, test_sll, test_tll)
        wandb.log({'test_nll': test_nll, 'test_sll': test_sll, 'test_tll': test_tll})
        
        logger.critical("Evaluate   | Test NLL {:5f} | Space: {:5f} | Time: {:5f}".format(test_nll, test_sll, test_tll))

    def test_lamb(self, dataloader, trained_model, device):
        # Load ground truth lamb
        from data.synthetic import STHPDataset, STSCPDataset
        from visualization.plotter import plot_lambst_interactive, plot_lambst_static
        from loguru import logger
        import numpy as np
        import torch
        import wandb
        
        name: str = pytest.fn_params['dataloader']['name']
        if name == 'sthp0':
            synt = STHPDataset(s_mu=np.array([0, 0]), 
                               g0_cov=np.array([[.2, 0],
                                                [0, .2]]),
                               g2_cov=np.array([[.5, 0],
                                                [0, .5]]),
                               alpha=.5, beta=1, mu=.2,
                               dist_only=False)
            START_IDX = 0
            TIME_RANGE = 200
            # X_NUM = 31
            # Y_NUM = 31
            X_NUM = 101
            Y_NUM = 101
            trunc = False
        elif name == 'sthp1':
            synt = STHPDataset(s_mu=np.array([0, 0]), 
                               g0_cov=np.array([[5, 0],
                                                [0, 5]]),
                               g2_cov=np.array([[.1, 0],
                                                [0, .1]]),
                               alpha=.5, beta=.6, mu=.15,
                               dist_only=False)
            START_IDX = 0
            TIME_RANGE = 200
            X_NUM = 31
            Y_NUM = 31
            trunc = False
        elif name == 'sthp2':
            synt = STHPDataset(s_mu=np.array([0, 0]), 
                               g0_cov=np.array([[1, 0],
                                                [0, 1]]),
                               g2_cov=np.array([[.1, 0],
                                                [0, .1]]),
                               alpha=.3, beta=2, mu=1,
                               dist_only=False)
            START_IDX = 0
            TIME_RANGE = 200
            X_NUM = 31
            Y_NUM = 31
            trunc = False
        elif name == 'stscp0':
            X_NUM = 101
            Y_NUM = 101
            synt = STSCPDataset(g0_cov=np.array([[1, 0],
                                                 [0, 1]]),
                                g2_cov=np.array([[.85, 0],
                                                 [0, .85]]),
                                alpha=.2, beta=.2, mu=1, gamma=0,
                                x_num=X_NUM, y_num=Y_NUM,
                                max_history=100, dist_only=False)
            START_IDX = 5
            TIME_RANGE = 100
            trunc = True
        elif name == 'stscp1':
            X_NUM = 101
            Y_NUM = 101
            synt = STSCPDataset(g0_cov=np.array([[.4, 0],
                                                 [0, .4]]),
                                g2_cov=np.array([[.3, 0],
                                                 [0, .3]]),
                                alpha=.3, beta=.2, mu=1, gamma=0,
                                x_num=X_NUM, y_num=Y_NUM, lamb_max=4, 
                                max_history=100, dist_only=False)
            START_IDX = 2
            TIME_RANGE = 100
            trunc = True
        elif name == 'stscp2':
            X_NUM = 101
            Y_NUM = 101
            synt = STSCPDataset(g0_cov=np.array([[.25, 0],
                                                 [0, .25]]),
                                g2_cov=np.array([[.2, 0],
                                                 [0, .2]]),
                                alpha=.4, beta=.2, mu=1, gamma=0,
                                x_num=X_NUM, y_num=Y_NUM, lamb_max=4, 
                                max_history=100, dist_only=False)
            START_IDX = 0
            TIME_RANGE = 100
            trunc = True
        elif name in ['covid_nj_cases', 'earthquakes_jp']:
            return  # Real world dataset, no GT intensity
        else:
            raise NotImplementedError
        
        synt.load(f'data/raw/spatiotemporal/{name}.data', t_start=0, t_end=10000)
        train_loader, val_loader, test_loader = dataloader
        train_set = train_loader.dataset
        scales = (train_set.max - train_set.min).cpu().numpy()
        biases = train_set.min.cpu().numpy()
        
        his_st = []
        for _, _, st_x, st_y, loc in test_loader:
            if START_IDX not in loc[0]:
                continue
            his_st.append(st_y[np.where(loc[0] == START_IDX)])
            
        his_st = torch.cat(his_st, 0).squeeze(1).detach().cpu().numpy()
        his_st[:, -1] += START_IDX * TIME_RANGE
        
        # Calculate the ground truth intensity
        T_START = his_st[:, -1][0]
        T_END = his_st[:, -1][-1]
        # T_NUM = 101
        T_NUM = int(np.round(T_END) - np.round(T_START)) + 1

        lambs_gt, x_range, y_range, t_range = synt.get_lamb_st(x_num=X_NUM, y_num=Y_NUM, t_num=T_NUM,
                                                               # x_min=-2.5, y_min=-2.5, x_max=2.5, y_max=2.5,
                                                               t_start=np.round(T_START), t_end=np.round(T_END))
        x_min, x_max = x_range[0], x_range[-1]
        y_min, y_max = y_range[0], y_range[-1]
        
        model, train_losses, val_losses = trained_model
        from models.spatiotemporal.numint import MonteCarloSTPPSameInfluence
        from models.spatiotemporal.deepstpp import DeepSTPP
        title = ''
        if isinstance(model, MonteCarloSTPPSameInfluence):
            from models.spatiotemporal.numint import calc_lamb
            title = 'Monte Carlo STPP'
        elif isinstance(model, DeepSTPP):
            from models.spatiotemporal.deepstpp import calc_lamb
            title = 'Deep STPP'
        else:
            from models.spatiotemporal.model import calc_lamb
            title = 'AutoInt STPP'
        
        if isinstance(model, DeepSTPP):
            rtn = calc_lamb(model, test_loader, config, device, scales, biases, xmin=x_min, xmax=x_max, 
                            start_idx=START_IDX, ymin=y_min, ymax=y_max, round_time=False, 
                            x_nstep=X_NUM, y_nstep=Y_NUM, t_nstep=T_NUM, trunc=trunc)
        else:
            rtn = calc_lamb(model, test_loader, device, scales, biases, xmin=x_min, xmax=x_max, start_idx=START_IDX,
                            ymin=y_min, ymax=y_max, round_time=False, 
                            x_nstep=X_NUM, y_nstep=Y_NUM, t_nstep=T_NUM, trunc=trunc)
        lambs_autoint, x_range, y_range, t_range_, his_s, his_t = rtn
        
        fig = plot_lambst_interactive([lambs_gt, lambs_autoint], x_range, y_range, t_range, show=False,
                                      # cmax=0.4, cauto=True,
                                      cauto=True,
                                      master_title=f'{name} Comparison',
                                      subplot_titles=['Ground Truth', title])
        
        # lambs_gt = [lamb.T for lamb in lambs_gt]
        # plot_lambst_static(lambs_gt, x_range, y_range, t_range, cmax=3.5, fps=12, fn='gt2.mp4')
        # lambs_autoint = [lamb.T for lamb in lambs_autoint]
        # plot_lambst_static(lambs_autoint, x_range, y_range, t_range, cmax=3.5, fps=12, fn='auto2.mp4')
                
        file_name = 'intensity_cmp'
        html_fn = f"{relpath('figs', True)}/{file_name}.html"
        fig.write_html(html_fn)
        wandb.log({file_name: wandb.Html(html_fn)})

        # Calculate Hessian
        START_IDX = 0
        hessians = []
        while True:
            his_st = []
            for _, _, st_x, st_y, loc in test_loader:
                if START_IDX not in loc[0]:
                    continue
                his_st.append(st_y[np.where(loc[0] == START_IDX)])
            if len(his_st) == 0:
                break  # No more index
                
            his_st = torch.cat(his_st, 0).squeeze(1).detach().cpu().numpy()
            his_st[:, -1] += START_IDX * TIME_RANGE
            
            T_START = his_st[:, -1][0]
            T_END = his_st[:, -1][-1]

            lambs_gt, x_range, y_range, t_range = synt.get_lamb_st(x_num=X_NUM, y_num=Y_NUM, t_num=101, 
                                                                   t_start=T_START, t_end=T_END)
            x_min, x_max = x_range[0], x_range[-1]
            y_min, y_max = y_range[0], y_range[-1]
            
            model, train_losses, val_losses = trained_model
            if isinstance(model, DeepSTPP):
                rtn = calc_lamb(model, test_loader, config, device, scales, biases, xmin=x_min, xmax=x_max, 
                                start_idx=START_IDX, ymin=y_min, ymax=y_max, round_time=False, 
                                x_nstep=X_NUM, y_nstep=Y_NUM, t_nstep=101, trunc=trunc)
            else:
                rtn = calc_lamb(model, test_loader, device, scales, biases, xmin=x_min, xmax=x_max, start_idx=START_IDX,
                                ymin=y_min, ymax=y_max, round_time=False, 
                                x_nstep=X_NUM, y_nstep=Y_NUM, t_nstep=101, trunc=trunc)
            lambs_autoint, x_range, y_range, t_range_, his_s, his_t = rtn
            
            # Normalize and calculate
            for p, q in zip(lambs_autoint, lambs_gt):
                p = p.flatten() / p.sum()
                q = q.flatten() / q.sum()
                dist = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
                hessians.append(dist)
            
            START_IDX += 1

        hessian = np.mean(hessians)
        wandb.log({'hessian': hessian})
        logger.critical(f'Hessian distance: {hessian}')
