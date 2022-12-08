import torch
from torch import nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class RMTPP(nn.Module):
    
    def __init__(self, hidden_size, t_end, device):
        """
        :param hidden_size: the dimension of hidden representation and linear hidden layer
        :param t_end: the time when observation terminates
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.t_end = t_end
        self.device = device
        
        # Encoder
        self.RNN = nn.GRU(1, hidden_size, batch_first=True, num_layers=1)
        
        # a * exp(-b) 
        self.param = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )
        
    def forward(self, seq_pads, seq_lens, t_last):    
        """
        Encode a batch of sequences to hidden states, including h0 (all zeros)
        
        :param seq_pads: [batch, maxlen, 1], the padded event timings
        :param seq_lens: [batch], the sequence length before padding
        :param t_last: [batch], the last event timings; 
                       if is None, then time after last event is not considered
        :return hidden: [batch, maxlen+1, hiddim], the representation of history from h0 to hn
        :return delta_t_pads: [batch, maxlen(+1), 1]] the padded event timing differences, 
                              include before the first event and after last event
        """
        # Convert absolute timing to timing difference
        delta_t_pads = - torch.ones_like(seq_pads)
        for i, seq_len in enumerate(seq_lens):
            delta_t_pads[i, 0] = seq_pads[i, 0]  # retain first event timing
            delta_t_pads[i, 1:seq_len] = seq_pads[i, 1:seq_len] - seq_pads[i, :seq_len - 1]
            
        delta_t_input = pack_padded_sequence(delta_t_pads, seq_lens, batch_first=True, enforce_sorted=False)
        
        # Use RNN to generate hidden representation after each event
        hidden, _ = self.RNN(delta_t_input)
        hidden, _ = pad_packed_sequence(hidden, padding_value=0, batch_first=True)  # [batch, seq_len, 128]
        
        # Add the hidden vector (all 0) before the first event
        hidden = torch.cat([torch.zeros(len(hidden), 1, self.hidden_size).to(self.device), hidden], 1)
        
        # Extend delta_t_pads to include after last event
        if t_last is not None:
            delta_t_pads = torch.cat([delta_t_pads, - torch.ones(len(seq_lens), 1, 1).to(self.device)], 1)
            for i, seq_len in enumerate(seq_lens):
                delta_t_pads[i, seq_len] = self.t_end - t_last[i]
        
        return hidden, delta_t_pads
    
    @staticmethod
    def h(times, seq_pads, hidden):   
        """
        Utility for preparing the input to λ/F at certain time points
        
        :param times: [batch, N, 1], the time points to evaluate intensity
        :param seq_pads: [batch, maxlen, 1], the padded event timings
        :param hidden: [batch, maxlen+1, hiddim], the representation of history from h0 to hn
        :return h_select: [batch, N, hiddim], the hidden at the time points 
        :return t_select: [batch, N, 1], the last event time before the time points 
        """
        temp = torch.sum(times.transpose(2, 1) <= seq_pads, 1)  # Find reverse index
        idx_max, _ = torch.max(temp, 1)
        idxs = idx_max.unsqueeze(-1) - temp  # Number of event before each time
        h_select = torch.stack([torch.index_select(h, 0, idx) for h, idx in zip(hidden, idxs)])
        temp = torch.cat([torch.zeros(len(seq_pads), 1, 1).to(seq_pads.device), seq_pads], 1)
        t_select = torch.stack([torch.index_select(t, 0, idx) for t, idx in zip(temp, idxs)])
        return h_select, t_select  # Last event time and hidden at the timepoints
    
    def loss(self, hidden, delta_t_pads, seq_lens):
        """
        Calculate NLL for a batch of sequences
        
        :param hidden: [batch, maxlen+1, hiddim], the representation of history from h0 to hn
        :param delta_t_pads: [batch, maxlen(+1), 1]] the padded event timing differences, 
                             include before the first event (and after last event)
        :param seq_lens: [batch], the sequence length before padding
        :return nll: scalar, the average negative log likelihood
        """
        include_after = hidden.shape[1] == delta_t_pads.shape[1]
        
        ab = self.param(hidden)
        a = nn.functional.softplus(ab[..., 0])  # Intensity is non-negative
        b = nn.functional.softplus(ab[..., 1])
        t = delta_t_pads.squeeze()
        
        lambs = a * torch.exp(-b * t)
        lamb_ints = (a - a * torch.exp(-b * t)) / b
        
        loglike = []
        for lamb_int, lamb, seq_len in zip(lamb_ints, lambs, seq_lens):
            length = seq_len + 1 if include_after else seq_len
            loglike.append(torch.log(lamb[:seq_len]).sum() - lamb_int[:length].sum())  # NLL
            
        return - sum(loglike) / len(loglike)


class RMTPPori(nn.Module):
    
    def __init__(self, hidden_size, t_end, device):
        """
        :param hidden_size: the dimension of hidden representation and linear hidden layer
        :param t_end: the time when observation terminates
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.t_end = t_end
        self.device = device
        
        # Encoder
        self.RNN = nn.GRU(1, hidden_size, batch_first=True, num_layers=1)
        
        # a * exp(-b) 
        self.param = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.b = torch.nn.Parameter(torch.ones(1))
        
    def forward(self, seq_pads, seq_lens, t_last=None):          
        """
        Encode a batch of sequences to hidden states, including h0 (all zeros)
        
        :param seq_pads: [batch, maxlen, 1], the padded event timings
        :param seq_lens: [batch], the sequence length before padding
        :param t_last: [batch], the last event timings; 
                       if is None, then time after last event is not considered
        :return hidden: [batch, maxlen+1, hiddim], the representation of history from h0 to hn
        :return delta_t_pads: [batch, maxlen(+1), 1]] the padded event timing differences, 
                              include before the first event and after last event
        """
        # Convert absolute timing to timing difference
        delta_t_pads = - torch.ones_like(seq_pads)
        for i, seq_len in enumerate(seq_lens):
            delta_t_pads[i, 0] = seq_pads[i, 0]  # Retain first event timing
            delta_t_pads[i, 1:seq_len] = seq_pads[i, 1:seq_len] - seq_pads[i, :seq_len - 1]
            
        delta_t_input = pack_padded_sequence(delta_t_pads, seq_lens, batch_first=True, enforce_sorted=False)
        
        # Use RNN to generate hidden representation after each event
        hidden, _ = self.RNN(delta_t_input)
        hidden, _ = pad_packed_sequence(hidden, padding_value=0, batch_first=True)  # [batch, seq_len, 128]
        
        # Add the hidden vector (all 0) before the first event
        hidden = torch.cat([torch.zeros(len(hidden), 1, self.hidden_size).to(self.device), hidden], 1)
        
        # Extend delta_t_pads to include after last event
        if t_last is not None:
            delta_t_pads = torch.cat([delta_t_pads, - torch.ones(len(seq_lens), 1, 1).to(self.device)], 1)
            for i, seq_len in enumerate(seq_lens):
                delta_t_pads[i, seq_len] = self.t_end - t_last[i]
        
        if t_last is not None:
            return hidden, delta_t_pads
        else:
            return hidden[:, :-1], delta_t_pads
    
    @staticmethod
    def h(times, seq_pads, hidden): 
        """
        Utility for preparing the input to λ/F at certain time points
        
        :param times: [batch, N, 1], the time points to evaluate intensity
        :param seq_pads: [batch, maxlen, 1], the padded event timings
        :param hidden: [batch, maxlen+1, hiddim], the representation of history from h0 to hn
        
        :return h_select: [batch, N, hiddim], the hidden at the time points 
        :return t_select: [batch, N, 1], the last event time before the time points 
        """
        temp = torch.sum(times.transpose(2, 1) <= seq_pads, 1)  # Find reverse index
        idx_max, _ = torch.max(temp, 1)
        idxs = idx_max.unsqueeze(-1) - temp  # Number of event before each time
        h_select = torch.stack([torch.index_select(h, 0, idx) for h, idx in zip(hidden, idxs)])
        temp = torch.cat([torch.zeros(len(seq_pads), 1, 1).to(seq_pads.device), seq_pads], 1)
        t_select = torch.stack([torch.index_select(t, 0, idx) for t, idx in zip(temp, idxs)])
        return h_select, t_select  # Last event time and hidden at the timepoints
    
    def loss(self, hidden, delta_t_pads, seq_lens):
        """
        Calculate NLL for a batch of sequences
        
        :param hidden: [batch, maxlen+1, hiddim], the representation of history from h0 to hn
        :param delta_t_pads: [batch, maxlen(+1), 1]] the padded event timing differences, 
                             include before the first event (and after last event)
        :param seq_lens: [batch], the sequence length before padding
        :return nll: scalar, the average negative log likelihood
        """
        include_after = hidden.shape[1] == delta_t_pads.shape[1]
        
        t = delta_t_pads.squeeze()

        a = self.param(hidden).squeeze(-1)
        a = nn.functional.softplus(a)  # Intensity is non-negative
        b = nn.functional.softplus(self.b)
        t = delta_t_pads.squeeze()
        
        lambs = a * torch.exp(-b * t)
        lamb_ints = (a - a * torch.exp(-b * t)) / b
        
        loglike = []
        for lamb_int, lamb, seq_len in zip(lamb_ints, lambs, seq_lens):
            length = seq_len + 1 if include_after else seq_len
            loglike.append(torch.log(lamb[:seq_len]).sum() - lamb_int[:length].sum())  # NLL
            
        return - sum(loglike) / len(loglike)
