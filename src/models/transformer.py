import numpy as np
import math

import torch
from torch import nn


"""
Return a square attention mask to only allow self-attention layers to attend the earlier positions
"""
def subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


"""
Return a mask hidding the padded entries

seqlens: [batch], list
"""
def padding_mask(seq_lens):
    max_len = max(seq_lens)
    mask = torch.zeros((len(seq_lens), max_len), dtype=torch.bool)
    for i, seq_len in enumerate(seq_lens):
        mask[i, seq_len:] = True
    return mask


"""
Injects some information about the relative or absolute position of the tokens in the sequence
ref: https://github.com/harvardnlp/annotated-transformer/
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, device):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model # encoded's size
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

    '''
    x: [seqlen, batch, hidden_size]
    t: [seqlen, batch, 1]
    '''
    def forward(self, x, t):
        pe = torch.zeros(len(x), self.d_model).to(self.device)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(self.device)
        
        pe = torch.zeros(*t.shape[:2], self.d_model).to(self.device)
        pe[..., 0::2] = torch.sin(t * div_term)
        pe[..., 1::2] = torch.cos(t * div_term)
        
        x = x + pe[:x.size(0)]
        return self.dropout(x)
    

"""
Encode time/space record to variational posterior for location latent
"""
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_heads, n_layers, dropout, device):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(hidden_size, dropout, device)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, n_heads,
                                                    hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.Linear(input_size, hidden_size, bias=False)
        self.init_weights()
        self.device = device
        self.to(device)
        
        
    def init_weights(self, initrange=0.1):
        self.encoder.weight.data.uniform_(-initrange, initrange)

        
    '''
    x: [batch, seqlen, input_size]
    t: [batch, seqlen, 1]
    mask: [seqlen, seqlen] allow events only to see events before
    seq_lens: [batch]
    ''' 
    def forward(self, x, t, mask=None, seq_lens=None):
        x = x.transpose(1,0) # Convert to seq-len-first
        t = t.transpose(1,0)
        if mask is None:
            mask = subsequent_mask(len(x)).to(self.device)
        x = self.encoder(x) * math.sqrt(self.hidden_size)
        x = self.pos_encoder(x, t)
        
        if seq_lens is None:
            src_key_padding_mask = None
        else:
            src_key_padding_mask = padding_mask(seq_lens).to(self.device)
        
        output = self.transformer_encoder(x, mask, src_key_padding_mask)
        return output.transpose(1,0)



class TransformerHawkesProcess(nn.Module):        
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
        self.transformer = Encoder(1, hidden_size, 2, 1, 0.5, device)
        
        # Intensity decoder
        self.f = nn.Linear(hidden_size, 1)

        if not deepf:
            self.current = nn.Sequential(
                nn.Softplus()
            )
        else:
            self.current = nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
                nn.Softplus()
            )

        self.beta = torch.nn.Parameter(torch.ones(1) * 0.15) # intensity decay rate
        
                
    '''
    Encode a batch of sequences to hidden states, including h0 (all zeros)
    
    ARGS
    seq_pads: [batch, maxlen, 1], the padded event timings
    seq_lens: [batch], the sequence length before padding
    
    RETURN
    hidden: [batch, maxlen+1, hiddim], the representation after each event
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
        
        # Use Transformer to generate hidden representation after each event
        hidden = self.transformer(delta_t_pads[:, :-1], seq_pads, seq_lens=seq_lens)
        
        # Add the hidden vector (all 0) before the first event
        hidden = torch.cat([torch.zeros(len(hidden), 1, self.hidden_size).to(self.device), hidden], 1)
        
        return hidden, delta_t_pads
    
    
    '''
    Utility for preparing the input to λ at certain time points
    
    ARGS
    times: [batch, N, 1], the time points to evaluate intensity
    seq_pads: [batch, maxlen, 1], the padded event timings
    afters: [num_hidden, batch, maxlen+1, hidden_size], the representation after each event
    
    RETURN
    h_select, t_select: [batch, N, hidden_size], [batch, N, 1] hidden and time necessary for computing lamb
    '''
    @staticmethod
    def h(times, seq_pads, afters):
        temp = torch.sum(times.transpose(2, 1) <= seq_pads, 1) # find reverse index
        idx_max, _ = torch.max(temp, 1)
        idxs = idx_max.unsqueeze(-1) - temp # number of event before each time
        h_select = torch.stack([torch.index_select(h, 0, idx) for h, idx in zip(afters, idxs)])
        temp = torch.cat([torch.zeros(len(seq_pads), 1, 1).to(seq_pads.device), seq_pads], 1)
        t_select = torch.stack([torch.index_select(t, 0, idx) for t, idx in zip(temp, idxs)])

        return h_select, t_select
    
    
    '''
    Calculate NLL for a batch of sequences
    
    ARGS
    hidden: [batch, maxlen+1, hiddim], the representation after each event
    delta_t_pads: [batch, maxlen+1, 1]] the padded event timing differences, 
                  include before the first event and after last event
    seq_lens: [batch], the sequence length before padding
    
    RETURN
    nll: scalar, the average negative log likelihood
    '''
    def loss(self, hidden, delta_t_pads, seq_lens):
        # Calculate intensity before every event
        befores = self.f(hidden)
        lambs = self.current(befores + self.beta * delta_t_pads)[:, :-1]
            
        # Calculate lambs integral using Monte Carlo
        lamb_int_samples = []
        diff = np.infty
        
        # for _ in range(30):
        while diff > 2.0: # Estimate with precision 2.0
            rand_time = torch.rand_like(delta_t_pads) * delta_t_pads # random time between [0, delta_t]
            lamb_sample = self.current(befores + self.beta * rand_time)
            lamb_int_sample = lamb_sample * delta_t_pads # f(x_bar) * V
            lamb_int_samples.append(lamb_int_sample)
            
            diff = 0.0
            
            lamb_int_now = sum(lamb_int_samples) / len(lamb_int_samples)
            if len(lamb_int_samples) > 1:
                lamb_int_pre = sum(lamb_int_samples[:-1]) / (len(lamb_int_samples) - 1)
            else:
                lamb_int_pre = - torch.ones_like(lamb_int_now)
            
            for now, pre, seq_len in zip(lamb_int_now, lamb_int_pre, seq_lens):
                diff += torch.nansum(abs(now - pre)[:seq_len]).item()
                
            #print(diff)
                
        lamb_ints = sum(lamb_int_samples) / len(lamb_int_samples)
        loglike = []
        
        for lamb_int, lamb, seq_len in zip(lamb_ints, lambs, seq_lens):
            loglike.append(torch.log(lamb[:seq_len]).sum() - lamb_int[:seq_len+1].sum()) # NLL
            
        return - sum(loglike) / len(loglike)


# model = TransformerHawkesProcess(hidden_size=32, t_end=50.0, device=device).to(device)
# hidden, delta_t_pads = model(seq_pads, seq_lens)
# model.loss(hidden, delta_t_pads, seq_lens)