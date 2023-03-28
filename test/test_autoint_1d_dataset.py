import pytest

from conftest import relpath, update_params, get_params, log_config, plot_training_progress

    
@pytest.fixture(
    scope="module",
    params=get_params('dataloader')
)
def dataloader(device, request):
    import numpy as np
    from torch.utils.data import DataLoader
    from data.data import TPPWrapper, pad_collate

    update_params('dataloader', request)

    def lamb_func(t, his_t):  # Intensity spaceholder for real-world dataset
        alpha = .2
        beta = .5
        mu = .2
        delta_t = t - his_t[his_t < t]
        lamb_t = mu + alpha * np.sum((np.cos(delta_t) + 1) * np.exp(-beta * (delta_t)))
        L = 2 / lamb_t
        M = mu + alpha * np.sum(np.exp(-beta * (delta_t)))
        return (lamb_t, L, M)
    
    name = request.param['name']  # Default config for synthetic datasets
    split = [4096, 2048, 2048]
    batch_size = 64
    t_end = 50.0
    if name == 'covidNJ':
        split = [1100, 275, 275]
        batch_size = 32
        t_end = 140.0  # Scaled up range by 20
    elif name == 'earthquakesJP':
        split = [1000, 250, 250]
        batch_size = 32
        t_end = 60.0   # Scaled up range by 2
    elif name == 'shakyHawkes':
        def lamb_func(t, his_t):
            alpha = .2
            beta = .2
            mu = .2
            delta_t = t - his_t[his_t < t]
            lamb_t = mu + alpha * np.sum((np.cos(delta_t) + 1) * np.exp(-beta * (delta_t)))
            L = np.inf
            M = mu + alpha * np.sum(np.exp(-beta * (delta_t)))
            return (lamb_t, L, M)
    elif name == 'delayPeak':
        def lamb_func(t, his_t):
            alpha = .2
            beta = .5
            mu = .3
            delta_t = t - his_t[his_t < t]
            lamb_t = mu + alpha * np.sum(np.maximum(-(beta * delta_t - 1) ** 2 + 1, 0))
            
            L = np.inf
            M = lamb_t + alpha
            return (lamb_t, L, M)
    elif name == 'shiftHawkes':
        def lamb_func(t, his_t):
            alpha = .2
            beta = .2
            mu = .2
            threshold = 2.0
            delta_t = t - his_t[his_t < t]
            delta_t[delta_t < threshold] = np.infty
            lamb_t = mu + alpha * np.sum(np.exp(-beta * (delta_t - threshold)))
            L = np.inf
            M = lamb_t + alpha
            return (lamb_t, L, M)
    else:
        raise NotImplementedError
        
    train_set = TPPWrapper(lamb_func, n_sample=split[0], t_end=t_end, max_lamb=100, fn=f'data/temporal/{name}.db')
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, collate_fn=pad_collate)
    val_set = TPPWrapper(lamb_func, n_sample=split[1], n_start=split[0], t_end=t_end, max_lamb=100, 
                         fn=f'data/temporal/{name}.db')
    val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size, collate_fn=pad_collate)
    test_set = TPPWrapper(lamb_func, n_sample=split[2], n_start=split[0] + split[1], t_end=t_end, max_lamb=100, 
                            fn=f'data/temporal/{name}.db')
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size, collate_fn=pad_collate)

    return lamb_func, t_end, train_set, val_set, test_set, train_loader, val_loader, test_loader


@pytest.fixture(
    scope="class",
    params=get_params('trained_model')
)
def trained_model(dataloader, device, request):
    import torch
    from models.model import AutoIntTPPSameInfluence, AutoIntGRUTPPSameInfluence
    from models.model import FullyTemporalPointProcess
    from models.ctrnn_process import LSTMNeuralHawkesProcess, GRUNeuralHawkesProcess
    from models.rmtpp import RMTPPori
    from models.transformer import TransformerHawkesProcess
    from integration.numint import ChebyshevSameInfluenceProcess, MonteCarloSameInfluenceProcess
    from integration.numint import TaylorSameInfluenceProcess
    from loguru import logger
    import os

    update_params("trained_model", request)
    log_config()
    
    lamb_func, t_end, train_set, val_set, test_set, train_loader, val_loader, test_loader = dataloader
    
    model_name = request.param['name']
    hidden_size = request.param['hidden_size']
    if model_name == 'autoint':
        model = AutoIntTPPSameInfluence(hidden_size, t_end=t_end, device=device)
    elif model_name == 'autoint-gru':
        model = AutoIntGRUTPPSameInfluence(hidden_size, t_end=t_end, device=device)
    elif model_name == 'neural-hawkes':
        model = LSTMNeuralHawkesProcess(hidden_size, t_end=t_end, device=device)
    elif model_name == 'rmtpp':
        model = RMTPPori(hidden_size, t_end=t_end, device=device)
    elif model_name == 'transformer-hawkes':
        model = TransformerHawkesProcess(hidden_size, t_end=t_end, device=device)
    elif model_name == 'ct-gru':
        model = GRUNeuralHawkesProcess(hidden_size, t_end=t_end, device=device)
    elif model_name == 'chebyshev':
        model = ChebyshevSameInfluenceProcess(hidden_size, t_end=t_end, device=device)
    elif model_name == 'monte-carlo':
        model = MonteCarloSameInfluenceProcess(hidden_size, t_end=t_end, device=device)
    elif model_name == 'taylor':
        model = TaylorSameInfluenceProcess(hidden_size, t_end=t_end, device=device)
    elif model_name == 'fully':
        model = FullyTemporalPointProcess(hidden_size, t_end=t_end, device=device)
    else:
        raise NotImplementedError
        
    model = model.to(device)
    model_fn = relpath('models') + '.pkl'
    if not request.param['retrain']:  # try to use the previous trained model
        if os.path.exists(model_fn):
            model.load_state_dict(torch.load(model_fn, map_location=device)['model_state_dict'])
            train_losses = torch.load(model_fn, map_location=device)['train_losses']
            val_losses = torch.load(model_fn, map_location=device)['val_losses']
            logger.info('Previous model found and loaded.')
            model.eval()
            return model, train_losses, val_losses
        else:
            logger.info('Previous model not found. Retraining...')
  
    optimizer = torch.optim.Adam(model.parameters(), lr=request.param['lr'])
    
    if pytest.fn_params['dataloader']['name'] in ['covidNJ', 'earthquakesJP']:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
        
    train_losses = []
    val_losses = []

    for epoch in range(request.param['n_epoch']):
        train_loss = []
        train_like = []

        for i, (seq_pads, seq_lens, _) in enumerate(train_loader):
            optimizer.zero_grad()
            
            seq_pads = seq_pads.float().to(device)
            
            # Compute loss
            if model_name == 'neural-hawkes':
                befores, afters, delta_t_pads, decay_infos = model(seq_pads, seq_lens)
                nll = model.loss(befores, afters, delta_t_pads, seq_lens, *decay_infos)
            elif model_name == 'ct-gru':
                befores, afters, delta_t_pads = model(seq_pads, seq_lens)
                nll = model.loss(befores, afters, delta_t_pads, seq_lens)
            elif model_name == 'rmtpp' or model_name == 'transformer-hawkes' or model_name == "fully":
                hidden, delta_t_pads = model(seq_pads, seq_lens)
                nll = model.loss(hidden, delta_t_pads, seq_lens)
            else:
                nll = model(seq_pads, seq_lens)
            loss = nll
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()

            # Project to feasible set
            if "autoint" in model_name:
                model.project()
            
            # Print statistics
            train_loss.append(loss.item())
            train_like.append(nll.item())
            
        train_losses.append(sum(train_loss) / len(train_loss))
        
        logger.debug(f'Epoch {epoch} \t trainloss: {train_losses[-1]}')
        
        scheduler.step()
        
        # Validate the model
        if (epoch + 1) % request.param['n_eval_epoch'] == 0:
            model.eval()
            
            val_loss = []
            val_like = []
            
            for (seq_pads, seq_lens, _) in val_loader:
                seq_pads = seq_pads.float().to(device)
                
                if model_name == 'neural-hawkes':
                    befores, afters, delta_t_pads, decay_infos = model(seq_pads, seq_lens)
                    nll = model.loss(befores, afters, delta_t_pads, seq_lens, *decay_infos)
                elif model_name == 'ct-gru':
                    befores, afters, delta_t_pads = model(seq_pads, seq_lens)
                    nll = model.loss(befores, afters, delta_t_pads, seq_lens)
                elif model_name == 'rmtpp' or model_name == 'transformer-hawkes' or model_name == "fully":
                    hidden, delta_t_pads = model(seq_pads, seq_lens)
                    nll = model.loss(hidden, delta_t_pads, seq_lens)
                else:
                    nll = model(seq_pads, seq_lens)
                loss = nll

                val_loss.append(loss.item())
                val_like.append(nll.item())
                
            val_losses.append(sum(val_loss) / len(val_loss))
            
            logger.info(f'Epoch {epoch} \t valloss: {val_losses[-1]}')
            
            if val_losses[-1] == min(val_losses):   
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                }, model_fn)
            
            model.train()
    
    model.load_state_dict(torch.load(model_fn, map_location=device)['model_state_dict'])
    
    return model, train_losses, val_losses


class TestClass:

    def test_result(self, dataloader, trained_model):
        from utils import evaluate, plot_predict_intensity
        from loguru import logger
        from predict import get_predict
        
        model, train_losses, val_losses = trained_model
        model.eval()
        lamb_func, t_end, train_set, val_set, test_set, train_loader, val_loader, test_loader = dataloader
        predict = get_predict(model)
        
        plot_training_progress(train_losses, val_losses, f"{relpath('figs', True)}/training")
        
        fig = plot_predict_intensity(lamb_func, predict, model, train_set.seqs[0].numpy(), t_end=t_end)
        fig.savefig(f"{relpath('figs', True)}/intensity.png")
        logger.critical(evaluate(lamb_func, predict, model, test_set.seqs))
