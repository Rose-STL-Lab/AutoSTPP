import numpy as np

import torch
from torch import nn

from ctrnn import CTGRU, CTLSTM

eps = 1e-10

class GRUNeuralHawkesProcess(nn.Module):
    
    '''
    hidden_size: the dimension of hidden representation and linear hidden layer
    t_end: the time when observation terminates
    '''
    def __init__(self, hidden_size, t_end, device, deepf=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.t_end = t_end
        self.device = device
        
        # Encoder
        self.RNN = CTGRU(1, hidden_size, device, 1.0, t_end)
        
        # Intensity decoder
        if not deepf:
            self.f = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Softplus()
            )
        else:
            self.f = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
                nn.Softplus()
            )
        
                
    '''
    Encode a batch of sequences to hidden states, including h0 (all zeros)
    
    ARGS
    seq_pads: [batch, maxlen, 1], the padded event timings
    seq_lens: [batch], the sequence length before padding
    
    RETURN
    befores: [num_hidden, batch, maxlen+1, hiddim], the representation before each event
    afters:  [num_hidden, batch, maxlen+1, hiddim], the representation after each event
    delta_t_pads: [batch, maxlen(+1), 1]] the padded event timing differences, 
                  include before the first event and after last event
    '''
    def forward(self, seq_pads, seq_lens):
        # Compute the last event timings
        t_last = torch.gather(seq_pads, 1, torch.tensor(seq_lens).to(self.device).view(-1, 1, 1) - 1).squeeze(-1)
        
        # Convert absolute timing to timing difference
        delta_t_pads = - torch.ones_like(seq_pads)
        for i, seq_len in enumerate(seq_lens):
            delta_t_pads[i, 0] = seq_pads[i, 0] # retain first event timing
            delta_t_pads[i, 1:seq_len] = seq_pads[i, 1:seq_len] - seq_pads[i, :seq_len-1]
        
        # Extend delta_t_pads to include after last event
        delta_t_pads = torch.cat([delta_t_pads, - torch.ones(len(seq_lens), 1, 1).to(self.device)], 1)
        for i, seq_len in enumerate(seq_lens):
            delta_t_pads[i, seq_len] = self.t_end - t_last[i]
        
        # Use CTGRU to generate hidden representation before and after each event
        befores, afters = self.RNN(delta_t_pads[:, :-1], 
                                   delta_t_pads[:, 1:], seq_lens)
        
        return befores, afters, delta_t_pads
    
    
    '''
    Utility for preparing the input to λ at certain time points
    
    ARGS
    times: [batch, N, 1], the time points to evaluate intensity
    seq_pads: [batch, maxlen, 1], the padded event timings
    afters: [num_hidden, batch, maxlen+1, hiddim], the representation after each event
    
    RETURN
    inp: [batch, N, hiddim], the [t, h] input at the time points 
    '''
    def h(self, times, seq_pads, afters):
        temp = torch.sum(times.transpose(2, 1) <= seq_pads, 1) # find reverse index
        idx_max, _ = torch.max(temp, 1)
        idxs = idx_max.unsqueeze(-1) - temp # number of event before each time
        afters = afters.permute(1, 0, 2, 3)
        h_select = torch.stack([torch.index_select(h, 1, idx) for h, idx in zip(afters, idxs)]).permute(1, 0, 2, 3)
        temp = torch.cat([torch.zeros(len(seq_pads), 1, 1).to(seq_pads.device), seq_pads], 1)
        t_select = torch.stack([torch.index_select(t, 0, idx) for t, idx in zip(temp, idxs)])
        decay_rate = torch.exp(-(times - t_select).unsqueeze(-1) / self.RNN.scales).permute(3, 0, 1, 2)

        return torch.sum(h_select * decay_rate, 0)
    
    
    '''
    Calculate NLL for a batch of sequences
    
    ARGS
    befores: [num_hidden, batch, maxlen+1, hiddim], the representation before each event
    afters:  [num_hidden, batch, maxlen+1, hiddim], the representation after each event
    delta_t_pads: [batch, maxlen+1, 1]] the padded event timing differences, 
                  include before the first event and after last event
    seq_lens: [batch], the sequence length before padding
    
    RETURN
    nll: scalar, the average negative log likelihood
    '''
    def loss(self, befores, afters, delta_t_pads, seq_lens):
        # Calculate intensity before every event
        hidden_sum = torch.sum(befores, dim=0)
        lambs = self.f(hidden_sum[:, :-1]) + eps
            
        # Calculate lambs integral using Monte Carlo
        lamb_int_samples = []
        diff = np.infty
        
        for _ in range(30):
        # while diff > 2.0: # Estimate with precision 2.0
            rand_time = torch.rand_like(delta_t_pads) * delta_t_pads # random time between [0, delta_t]
            decay_rate = (rand_time / self.RNN.scales).permute(2, 0, 1).unsqueeze(-1)
            hidden_sample = torch.sum(afters * torch.exp(-decay_rate), dim=0) # perform decay
            lamb_sample = self.f(hidden_sample) + eps
            lamb_int_sample = lamb_sample * delta_t_pads # f(x_bar) * V
            lamb_int_samples.append(lamb_int_sample)
            
            '''
            diff = 0.0
            
            lamb_int_now = sum(lamb_int_samples) / len(lamb_int_samples)
            if len(lamb_int_samples) > 1:
                lamb_int_pre = sum(lamb_int_samples[:-1]) / (len(lamb_int_samples) - 1)
            else:
                lamb_int_pre = - torch.ones_like(lamb_int_now)
            
            for now, pre, seq_len in zip(lamb_int_now, lamb_int_pre, seq_lens):
                diff += torch.nansum(abs(now - pre)[:seq_len]).item()
                
            print(diff)
            '''
                
        lamb_ints = sum(lamb_int_samples) / len(lamb_int_samples)
        loglike = []
         
        for lamb_int, lamb, seq_len in zip(lamb_ints, lambs, seq_lens):
            loglike.append(torch.log(lamb[:seq_len]).sum() - lamb_int[:seq_len+1].sum()) # NLL
            
        return - sum(loglike) / len(loglike)


# model = GRUNeuralHawkesProcess(hidden_size=32, t_end=50.0, device=device).to(device)
# hidden, delta_t_pads = model(seq_pads, seq_lens, t_last)
# model.loss(hidden, delta_t_pads, seq_lens)


class LSTMNeuralHawkesProcess(nn.Module):
    
    '''
    hidden_size: the dimension of hidden representation and linear hidden layer
    t_end: the time when observation terminates
    '''
    def __init__(self, hidden_size, t_end, device, deepf=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.t_end = t_end
        self.device = device
        
        # Encoder
        self.RNN = CTLSTM(1, hidden_size, device)
        
        # Intensity decoder
        if not deepf:
            self.f = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Softplus()
            )
        else:
            self.f = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
                nn.Softplus()
            )
        
                
    '''
    Encode a batch of sequences to hidden states, including h0 (all zeros)
    
    ARGS
    seq_pads: [batch, maxlen, 1], the padded event timings
    seq_lens: [batch], the sequence length before padding
    
    RETURN
    befores: [batch, maxlen+1, hiddim], the representation before each event
    afters:  [batch, maxlen+1, hiddim], the representation after each event
    delta_t_pads: [batch, maxlen(+1), 1]] the padded event timing differences, 
                  include before the first event and after last event
    [cs, cbars, os, decays]: [batch, maxlen+1, hiddim], information for interpolation
    '''
    def forward(self, seq_pads, seq_lens):
        # Compute the last event timings
        t_last = torch.gather(seq_pads, 1, torch.tensor(seq_lens).to(self.device).view(-1, 1, 1) - 1).squeeze(-1)
        
        # Convert absolute timing to timing difference
        delta_t_pads = - torch.ones_like(seq_pads)
        for i, seq_len in enumerate(seq_lens):
            delta_t_pads[i, 0] = seq_pads[i, 0] # retain first event timing
            delta_t_pads[i, 1:seq_len] = seq_pads[i, 1:seq_len] - seq_pads[i, :seq_len-1]
        
        # Extend delta_t_pads to include after last event
        delta_t_pads = torch.cat([delta_t_pads, - torch.ones(len(seq_lens), 1, 1).to(self.device)], 1)
        for i, seq_len in enumerate(seq_lens):
            delta_t_pads[i, seq_len] = self.t_end - t_last[i]
        
        # Use CTGRU to generate hidden representation before and after each event
        befores, [afters, cs, cbars, os, decays] = self.RNN(delta_t_pads[:, :-1], delta_t_pads[:, 1:], seq_lens)
        
        return befores, afters, delta_t_pads, [cs, cbars, os, decays]
    
    
    '''
    Utility for preparing the input to λ at certain time points
    
    ARGS
    times: [batch, N, 1], the time points to evaluate intensity
    seq_pads: [batch, maxlen, 1], the padded event timings
    afters: [batch, maxlen+1, hidden_size], the representation after each event
    cs, cbars, os, decays: [batch, maxlen+1, hidden_size], info for calculating decay
    
    RETURN
    inp: [batch, N, hiddim], the h input at the time points 
    '''
    def h(self, times, seq_pads, afters, cs, cbars, os, decays):
        temp = torch.sum(times.transpose(2, 1) <= seq_pads, 1) # find reverse index
        idx_max, _ = torch.max(temp, 1)
        idxs = idx_max.unsqueeze(-1) - temp # number of event before each time
        h_select = torch.stack([torch.index_select(h, 0, idx) for h, idx in zip(afters, idxs)])
        temp = torch.cat([torch.zeros(len(seq_pads), 1, 1).to(seq_pads.device), seq_pads], 1)
        t_select = torch.stack([torch.index_select(t, 0, idx) for t, idx in zip(temp, idxs)])
        cs_select = torch.stack([torch.index_select(c, 0, idx) for c, idx in zip(cs, idxs)])
        cs_bar_select = torch.stack([torch.index_select(cbar, 0, idx) for cbar, idx in zip(cbars, idxs)])
        os_select = torch.stack([torch.index_select(o, 0, idx) for o, idx in zip(os, idxs)])
        decays_select = torch.stack([torch.index_select(decay, 0, idx) for decay, idx in zip(decays, idxs)])

        return os_select * torch.tanh(cs_bar_select + (cs_select - cs_bar_select) 
                                      * torch.exp(-(times - t_select) * decays_select))
    
    
    '''
    Calculate NLL for a batch of sequences
    
    ARGS
    befores: [batch, maxlen+1, hiddim], the representation before each event
    afters:  [batch, maxlen+1, hiddim], the representation after each event
    delta_t_pads: [batch, maxlen+1, 1]] the padded event timing differences, 
                  include before the first event and after last event
    seq_lens: [batch], the sequence length before padding
    cs, cbars, os, decays: [batch, maxlen+1, hidden_size], info for calculating decay
    
    RETURN
    nll: scalar, the average negative log likelihood
    '''
    def loss(self, befores, afters, delta_t_pads, seq_lens, cs, cbars, os, decays):
        # Calculate intensity before every event
        lambs = self.f(befores) + eps
            
        # Calculate lambs integral using Monte Carlo
        lamb_int_samples = []
        diff = np.infty
        
        for _ in range(30): # Estimate with precision 2.0
            rand_time = torch.rand_like(delta_t_pads) * delta_t_pads # random time between [0, delta_t]
            hidden_sample = os * torch.tanh(cbars + (cs - cbars) * torch.exp(-rand_time * decays)) # decay
            lamb_sample = self.f(hidden_sample) + eps
            lamb_int_sample = lamb_sample * delta_t_pads # f(x_bar) * V
            lamb_int_samples.append(lamb_int_sample)
            '''
            diff = 0.0
            
            lamb_int_now = sum(lamb_int_samples) / len(lamb_int_samples)
            if len(lamb_int_samples) > 1:
                lamb_int_pre = sum(lamb_int_samples[:-1]) / (len(lamb_int_samples) - 1)
            else:
                lamb_int_pre = - torch.ones_like(lamb_int_now)
            
            for now, pre, seq_len in zip(lamb_int_now, lamb_int_pre, seq_lens):
                diff += torch.nansum(abs(now - pre)[:seq_len]).item()
                
            print(diff)
            '''
                
        lamb_ints = sum(lamb_int_samples) / len(lamb_int_samples)
        loglike = []
        
        for lamb_int, lamb, seq_len in zip(lamb_ints, lambs, seq_lens):
            loglike.append(torch.log(lamb[:seq_len]).sum() - lamb_int[:seq_len+1].sum()) # NLL
            
        return - sum(loglike) / len(loglike)


# model = LSTMNeuralHawkesProcess(hidden_size=32, t_end=50.0, device=device).to(device)
# befores, afters, delta_t_pads, [cs, cbars, os, decays] = model(seq_pads, seq_lens)
# hts = os * nn.functional.tanh(cs) # shall equal afters
# cts = cbars + (cs - cbars) * torch.exp(-delta_t_pads * decays)
# hts = os * nn.functional.tanh(cts) # shall equal before