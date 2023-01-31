import torch
import torch.nn as nn
import numpy as np
from models.model import AutoIntTPPSameInfluence, AutoIntGRUTPPSameInfluence, FullyTemporalPointProcess
from models.ctrnn_process import LSTMNeuralHawkesProcess, GRUNeuralHawkesProcess
from models.rmtpp import RMTPPori
from models.transformer import TransformerHawkesProcess
from integration.numint import ChebyshevSameInfluenceProcess, MonteCarloSameInfluenceProcess
from integration.numint import TaylorSameInfluenceProcess
from utils import eps, adapt


def get_predict(model):
    
    device = next(model.parameters()).device

    if type(model) == AutoIntTPPSameInfluence:
        """
        Compute model intensities at different time

        :param his_t: np array [seqlen,], the event time history
        :param x: np array [N,], a batch of times
        :return:  np array [N, 1], the intensities at different time
        """
        def predict(model, his_t, x):
            def lamb_func(t, his_t):
                delta_t = t - his_t[his_t < t]
                if len(delta_t) == 0:
                    return model.background.item()
                lamb_t = model.F.dnforward(torch.tensor(delta_t).float().unsqueeze(-1).to(device), 
                                           [0]).sum() + model.background
                return lamb_t.item()
            
            y = np.array([lamb_func(t, his_t) for t in x])
            return y
    
    elif type(model) == AutoIntGRUTPPSameInfluence:
        def predict(model, his_t, x):
            seq = torch.tensor(his_t).to(device)
            seq[1:] = seq[1:] - seq[:-1]
            seq = seq.unsqueeze(-1).unsqueeze(0)
            hidden, _ = model.RNN(seq.float())
            alphas = model.alpha(hidden).squeeze() + 1.0
            
            def lamb_func(t, his_t):
                delta_t = t - his_t[his_t < t]
                if len(delta_t) == 0:
                    return model.background.item()
                term = model.F.dforward(torch.tensor(delta_t).float().unsqueeze(-1).to(device), 0).squeeze()
                lamb_t = sum(term * alphas[:len(delta_t)]) + model.background
                return lamb_t.item()
            
            y = np.array([lamb_func(t, his_t) for t in x])
            return y
        
    elif type(model) == LSTMNeuralHawkesProcess:
        def predict(model, his_t, x):
            seq_len = [len(his_t)]
            seq = adapt(his_t, device)
            befores, afters, _, decay_infos = model(seq, seq_len)
            times = adapt(x, device)
            lamb_input = model.h(times, seq, afters, *decay_infos)
            y = model.f(lamb_input).squeeze().detach().cpu().numpy() + eps
            return y
    
    elif type(model) == RMTPPori:
        def predict(model, his_t, x):    
            seq_len = [len(his_t)]
            seq = adapt(his_t, device)
            hidden, delta_t_pads = model(seq, seq_len, torch.tensor([his_t[-1]]).to(device))
            times = adapt(x, device)
            h, t = model.h(times, seq, hidden)
            a = model.param(h)
            
            a = nn.functional.softplus(a).squeeze(-1).squeeze(0)
            b = nn.functional.softplus(model.b)
            t = (times - t).squeeze()
            y = a * torch.exp(-b * t)
            
            return y.detach().cpu().numpy()
        
    elif type(model) == TransformerHawkesProcess:
        def predict(model, his_t, x):
            his_t = his_t
            seq_len = [len(his_t)]
            seq = adapt(his_t, device)
            
            hidden, _ = model(seq, seq_len)
            times = adapt(x, device)
            
            h_select, t_select = model.h(times, seq, hidden)
            y = nn.functional.softplus(model.f(h_select) + model.beta * (times - t_select))
            
            return y.squeeze().detach().cpu().numpy() + eps

    elif type(model) == GRUNeuralHawkesProcess:
        def predict(model, his_t, x):
            seq_len = [len(his_t)]
            seq = adapt(his_t, device)
            befores, afters, _ = model(seq, seq_len)
            times = adapt(x, device)
            lamb_input = model.h(times, seq, afters)
            y = model.f(lamb_input).squeeze().detach().cpu().numpy() + eps
            return y
        
    elif type(model) == ChebyshevSameInfluenceProcess or \
         type(model) == MonteCarloSameInfluenceProcess or \
         type(model) == TaylorSameInfluenceProcess:
        def predict(model, his_t, x):    
            def lamb_func(t, his_t):
                delta_t = t - his_t[his_t < t]
                if len(delta_t) == 0:
                    return model.background.item()
                lamb_t = model.f(torch.tensor(delta_t).float().unsqueeze(-1).to(device)).sum() + model.background
                return lamb_t.item()
            
            y = np.array([lamb_func(t, his_t) for t in x])
            return y
        
    elif type(model) == FullyTemporalPointProcess:
        def predict(model, his_t, x):
            seq_len = [len(his_t)]
            seq = adapt(his_t, device)
            hidden, _ = model(seq, seq_len, None)
            times = adapt(x, device)
            lamb_input = model.th(times, seq, hidden)
            y = model.F.dforward(lamb_input, 0).squeeze().detach().cpu().numpy() + eps
            return y
        
    else:
        raise NotImplementedError

    return predict
