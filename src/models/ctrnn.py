import numpy as np

import torch
from torch import nn

import math


"""
Continuous-time GRU s.t. interevent cell state decays exponentially to 0
-------------
Mozer, Michael C., Denis Kazakov, and Robert V. Lindsey. 
"Discrete event, continuous time rnns." 
arXiv preprint arXiv:1710.04110 (2017).
"""
class CTGRUCell(nn.Module):

    '''
    input_size: number of input dimension of GRU
    hidden_size: number of hidden dimension of GRU
    scales: a range of time scales that spans over possible delta_t
            exponential curve with an arbitrary decay rate is modeled as 
            a mixture of exponentials with these predefined decay rates
    bias: if false, then the layer does not use bias weights b_ih and b_hh
    '''
    def __init__(self, input_size, hidden_size, scales, bias=True):
        super(CTGRUCell, self).__init__()
        self.scales      = scales           # Preset tilde Tau Scales, 1D
        self.num_hidden  = scales.size()[0] # Number of hidden tensors
                                            # Each accounts for a timescale
                                            # = number of preset timescales
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.bias        = bias
        self.xmap        = nn.Linear(input_size,  3 * hidden_size, bias=bias)
        self.hmap        = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()


    '''
    Xavier initialize the parameters
    '''
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


    '''
    Decay with time δt
    
    hidden: [num_hidden, batch, hidden_size]
    delta_t: [batch]
    
    return new hidden: [num_hidden, batch, hidden_size]
    '''
    def decay(self, hidden, delta_t):
        # 4. Update multiscale state [2]
        if len(delta_t.shape) == 1:
            delta_t = delta_t.unsqueeze(-1)
        hidden = hidden * torch.exp(-delta_t / self.scales).T.unsqueeze(-1)
        return hidden


    '''
    h_{i+1} = GRU(x_{i+1}, h_{i})
    
    x: [batch, input_size]
    hidden: [num_hidden, batch, hidden_size]
    return new hidden: [num_hidden, batch, hidden_size]
    '''
    def forward(self, x, hidden):
        x = x.view(-1, self.input_size)
        i_r, i_i, i_n = self.xmap(x).chunk(3, dim=-1)

        # 5. Combine time scales
        hidden_sum = torch.sum(hidden, dim=0)
        h_r, h_i, h_n = self.hmap(hidden_sum).chunk(3, dim=-1)

        ln_scales  = torch.log(self.scales).view(-1, 1, 1) # [num_hidden, 1, 1]

        # 1. Determine retrieval scale and weighting
        ln_tau_r   = i_r + h_r # [batch, hidden_size]
        ln_tau_r   = ln_tau_r.unsqueeze(dim=0).repeat(self.num_hidden, 1, 1)
        resetgate  = nn.functional.softmax(-torch.pow(ln_tau_r - ln_scales, 2), dim=0)

        # 2. Detect relevant event signals 
        newgate    = torch.tanh(i_n + torch.sum(resetgate * h_n, dim=0))

        # 3. Determine storage scale and weighting
        ln_tau_i   = i_i + h_i
        ln_tau_i   = ln_tau_i.unsqueeze(dim=0).repeat(self.num_hidden, 1, 1)
        inputgate  = nn.functional.softmax(-torch.pow(ln_tau_i - ln_scales, 2), dim=0)

        # 4. Update multiscale state
        hy = hidden - inputgate * (hidden - newgate)

        return hy



class CTGRU(nn.Module):

    '''
    input_size: number of input dimension of GRU
    hidden_size: number of hidden dimension of GRU
    scales: timescales for CTGRU cell
    
    delta_t_min: minimum delta_t (>0), for precomputing scales
    delta_t_max: maximum delta_t (>0), for precomputing scales
    '''
    def __init__(self, input_size, hidden_size, device,
                 delta_t_min, delta_t_max):
        super(CTGRU, self).__init__()

        # Hidden dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.scales = self.tilde_Tau(delta_t_min, delta_t_max, r=np.sqrt(10)).to(device)
        self.ctgru_cell = CTGRUCell(input_size, hidden_size, self.scales.to(device))

        # Number of hidden tensors in one layer
        self.num_hidden = self.scales.size()[0]

        self.device = device
        self.to(device)

    '''
    tau_{i+1} = 10^0.5 tau_{i} is empiricially a high-fidely half-life match
    '''
    @staticmethod
    def tilde_Tau(delta_t_min, delta_t_max, r):
        M = math.ceil(math.log(delta_t_max / delta_t_min, r))
        scale  = delta_t_min
        scales = torch.zeros(M + 1)
        for i in range(M + 1):
            scales[i] = scale
            scale = scale * r
        return scales


    '''
    x: [batch, max_len, input_size], may be padded
    delta_t: [batch, max_len(-1), 1], may include after last event or not
    seq_lens: [batch], if entered, then accelerate by ignoring padding entries
    
    return before: [num_hidden, batch, max_len+1, hidden_size] (hidden before each event)
           after:  [num_hidden, batch, max_len+1, hidden_size] (hidden after each event)
           
    after include hidden at time 0
    before include hidden at time end
    '''
    # Time without padding acceleration: 3.2021427154541016
    # Time without padding acceleration: 1.6057541370391846

    def forward(self, x, delta_t, seq_lens=None): # delta_t: 1D, [seq]
        if delta_t.shape[1] < x.shape[1]:
            delta_t = torch.cat([delta_t, torch.zeros(len(delta_t), 1, 1).to(self.device)], 1)

        # Initialize hidden state with zeros
        h0   = torch.zeros(self.num_hidden, x.size(0), self.hidden_size).to(self.device)
        hn   = h0
        befores = [h0,]
        afters = [h0,] # include hidden at time 0 (no event)

        if type(seq_lens) is list:
            seq_lens = torch.tensor(seq_lens).to(self.device)

        for seq in range(x.size(1)):
            if seq_lens is not None:
                mask = seq_lens > seq
                indice = torch.nonzero(mask).squeeze(-1)
                hn = self.ctgru_cell(x[indice, seq], hn[:, indice]) # read a event
                temp = h0.index_copy(1, indice, hn)
                afters.append(temp)
                hn = self.ctgru_cell.decay(hn, delta_t[indice, seq]) # decay over time
                hn = h0.index_copy(1, indice, hn)
                befores.append(hn)
            else:
                hn = self.ctgru_cell(x[:, seq], hn) # read a event
                afters.append(hn)
                hn = self.ctgru_cell.decay(hn, delta_t[:, seq]) # decay over time
                befores.append(hn)

        return torch.stack(befores, dim=2), torch.stack(afters, dim=2)

# model = CTGRU(1, 128, device, 1.0, 50)
# model(seq_pads, seq_pads[:, 1:], seq_lens).shape


"""
Continuous-time LSTM s.t. interevent cell state decays from c to c_bar
-------------
Mei, Hongyuan, and Jason Eisner. 
"The neural hawkes process: A neurally self-modulating multivariate point process." 
arXiv preprint arXiv:1612.09328 (2016).
"""
class CTLSTMCell(nn.Module):

    '''
    input_size: number of input dimension of LSTM
    hidden_size: number of hidden dimension of LSTM
    '''
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.xmap = nn.Linear(input_size,  7 * hidden_size, bias=bias)
        self.hmap = nn.Linear(hidden_size, 7 * hidden_size, bias=bias)
        self.init_weights()


    '''
    Decay with time δt
    
    c:          [batch, hidden_size]
    c_bar:      [batch, hidden_size]
    o:          [batch, hidden_size]
    decay_rate: [batch, hidden_size]
    delta_t:    [batch]
    
    return new hidden: [num_hidden, batch, hidden_size]
    '''
    def decay(self, c, c_bar, o, decay_rate, delta_t):
        if len(delta_t.shape) == 1:
            delta_t = delta_t.unsqueeze(-1)

        ct = c_bar + (c - c_bar) * torch.exp(-decay_rate * delta_t)
        ht = o * torch.tanh(ct)
        return ht, ct


    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    '''
    x:  [batch, input_size]
    ht: [batch, hidden_size] (decayed hidden)
    ct: [batch, hidden_size] (decayed cell)
    
    return
    c:          [batch, input_size] (undecayed cell)
    c_bar:      [batch, input_size]
    o:          [batch, input_size]
    decay_rate: [batch, input_size]
    h:          [batch, input_size] (undecayed hidden)
    '''
    def forward(self, x, ht, ct, init_states=None):
        x = x.view(-1, self.input_size)

        wxi, wxf, wxz, wxo, wxd, wxi_bar, wxf_bar = self.xmap(x).chunk(7, dim=-1)
        uhi, uhf, uhz, uho, uhd, uhi_bar, uhf_bar = self.hmap(ht).chunk(7, dim=-1)

        i = torch.sigmoid(wxi + uhi)
        f = torch.sigmoid(wxf + uhf)
        z = torch.tanh(wxz + uhz)
        o = torch.sigmoid(wxo + uho)

        i_bar = torch.sigmoid(wxi_bar + uhi_bar)
        f_bar = torch.sigmoid(wxf_bar + uhf_bar)

        c = f * ct + i * z
        h = o * torch.tanh(c)
        c_bar = f_bar * ct + i_bar * z

        decay_rate = nn.functional.softplus(wxd + uhd)

        return h, c, c_bar, o, decay_rate


class CTLSTM(nn.Module):


    '''
    input_size: number of input dimension of LSTM
    hidden_size: number of hidden dimension of LSTM
    '''
    def __init__(self, input_size, hidden_size, device):
        super(CTLSTM, self).__init__()

        # Hidden dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.ctlstm_cell = CTLSTMCell(input_size, hidden_size)

        self.device = device
        self.to(device)


    '''
    x: [batch, max_len, input_size], may be padded
    delta_t: [batch, max_len(-1), 1], may include after last event or not
    seq_lens: [batch], if entered, then accelerate by ignoring padding entries
    
    return before: [batch, max_len+1, hidden_size] (hidden before each event)
           for h only
           after:  [batch, max_len+1, hidden_size] (hidden after each event)
           for h, c, c_bar, o, decay_rate
           
    after include hidden at time 0
    before include hidden at time end
    '''
    def forward(self, x, delta_t, seq_lens=None):
        if delta_t.shape[1] < x.shape[1]:
            delta_t = torch.cat([delta_t, torch.zeros(len(delta_t), 1, 1).to(self.device)], 1)

        # Initialize hidden state with zeros
        batch = x.size(0)
        h0 = torch.zeros(batch, self.hidden_size).to(self.device)
        c0 = torch.zeros(batch, self.hidden_size).to(self.device)
        hn = h0
        cn = c0
        befores = [h0,]
        afters = [(h0, c0, c0, c0, c0),] # include hidden at time 0 (no event)

        if type(seq_lens) is list:
            seq_lens = torch.tensor(seq_lens).to(self.device)

        for seq in range(x.size(1)):
            if seq_lens is not None:
                mask = seq_lens > seq
                indice = torch.nonzero(mask).squeeze(-1)
                hn, cn, c_bar, o, decay_rate = self.ctlstm_cell(x[indice, seq], hn[indice], cn[indice]) # jump
                afters.append((h0.index_copy(0, indice, hn),
                               c0.index_copy(0, indice, cn),
                               c0.index_copy(0, indice, c_bar),
                               c0.index_copy(0, indice, o),
                               c0.index_copy(0, indice, decay_rate)))
                hn, cn = self.ctlstm_cell.decay(cn, c_bar, o, decay_rate, delta_t[indice, seq]) # decay over time
                hn = h0.index_copy(0, indice, hn)
                cn = c0.index_copy(0, indice, cn)
                befores.append(hn)
            else:
                hn, cn, c_bar, o, decay_rate = self.ctlstm_cell(x[:, seq], hn, cn)
                afters.append((hn, cn, c_bar, o, decay_rate))
                hn, cn = self.ctlstm_cell.decay(cn, c_bar, o, decay_rate, delta_t[:, seq]) # decay over time
                befores.append(hn)

        afters = list(zip(*afters))

        return torch.stack(befores, dim=1), [torch.stack(afters[i], dim=1) for i in range(5)]


#model = CTLSTM(1, 128, device)
#befores, [afters, cs, cbars, os, decays] = model(seq_pads, seq_pads[:, 1:], seq_lens)
#print(befores.shape)
#print(cs.shape)
#print(cbars.shape)
#print(os.shape)
#print(decays.shape)
#print(afters.shape)
