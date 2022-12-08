import torch
from torch import nn

from integration.cheb import chebyCoef, chebyInt
from integration.taylor import TaylorInt

eps = 1e-10


class MonteCarloSameInfluenceProcess(nn.Module):
    
    def __init__(self, hidden_size, t_end, device):
        """
        :param hidden_size: the dimension of linear hidden layer
        :param t_end: the time when observation terminates
                      if is None, then time after last event is not considered
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.t_end = t_end
        self.device = device
        
        # log background intensity 
        self.background = torch.nn.Parameter(torch.ones(1))
        
        # λ
        self.f = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softplus()
        )
                    
    def forward(self, seq_pads, seq_lens):         
        """
        Calculate NLL for a batch of sequence
        
        :param seq_pads: [batch, maxlen, 1], the padded event timings
        :param seq_lens: [batch], the sequence length before padding
        :return nll: scalar, the average negative log likelihood
        """
        batch, maxlen, _ = seq_pads.shape
        t_last = torch.gather(seq_pads, 1, torch.tensor(seq_lens).to(self.device).view(-1, 1, 1) - 1).squeeze()
        
        # Perform outer subtraction
        seq_pads_roll = torch.cat((torch.zeros(batch, 1, 1).to(self.device), 
                                   seq_pads.transpose(1, 2)[..., :-1]), -1)
        diff_pads = seq_pads - seq_pads_roll
        tril_idx = torch.tril_indices(maxlen - 1, maxlen - 1)
        diff_pads = diff_pads[:, tril_idx[0] + 1, tril_idx[1] + 1].unsqueeze(-1)
        
        ########## Calculate intensity ############
        # [batch, seq_len]
        # intensity before every event
        lambs = self.f(diff_pads).squeeze(-1)
        lambs = lambs.split(list(range(1, maxlen)), dim=-1)
        lambs = torch.stack([lamb.sum(dim=-1) for lamb in lambs], -1)
        lambs = torch.cat((torch.zeros(batch, 1).to(self.device), lambs), -1)  # First event zero influence
        lambs += self.background  # Add background intensity
        
        ######### Calculate integral intensity ##########
        # [batch, seq_len]
        # cumulative influence of every event
        diff_pads_with_last = []
        for seq_pad, seq_len in zip(seq_pads, seq_lens):
            if self.t_end is None:
                t_end = seq_pad[seq_len - 1]
            else:
                t_end = self.t_end
            temp = t_end - seq_pad[:seq_len]
            temp = torch.cat((temp, torch.zeros(maxlen - seq_len, 1).to(self.device)))
            diff_pads_with_last.append(temp)
        diff_pads_with_last = torch.stack(diff_pads_with_last)
        
        lamb_int_samples = []
        # diff = np.infty
        
        # while diff > 2.0: # Estimate with precision 2.0
        for _ in range(15):
            rand_time = torch.rand_like(diff_pads_with_last) * diff_pads_with_last  # Random time
            lamb_sample = self.f(rand_time) + eps
            lamb_int_sample = lamb_sample * diff_pads_with_last  # f(x_bar) * V
            lamb_int_samples.append(lamb_int_sample)
            '''
            diff = 0.0
            
            lamb_int_now = sum(lamb_int_samples) / len(lamb_int_samples)
            if len(lamb_int_samples) > 1:
                lamb_int_pre = sum(lamb_int_samples[:-1]) / (len(lamb_int_samples) - 1)
            else:
                lamb_int_pre = - torch.ones_like(lamb_int_now)
            for now, pre, seq_len in zip(lamb_int_now, lamb_int_pre, seq_lens):
                diff += torch.sum(abs(now - pre)[:seq_len]).item()
            print(diff)
            '''
            
        lamb_ints = sum(lamb_int_samples) / len(lamb_int_samples)
        lamb_ints = lamb_ints.squeeze(-1)
        
        ######### Calculate loss ########
        sum_log_lambs = []
        for lamb, seq_len in zip(lambs, seq_lens):
            sum_log_lambs.append(torch.log(lamb[:seq_len]).sum())
            
        lamb_ints = torch.sum(lamb_ints, -1)
        if self.t_end is None:
            background_int = t_last * self.background
        else:
            background_int = self.t_end * self.background
        lamb_ints += background_int  # Add background integral
        
        nll = - (sum(sum_log_lambs) - sum(lamb_ints)) / batch
        
        return nll


class ChebyshevSameInfluenceProcess(nn.Module):
    
    def __init__(self, hidden_size, t_end, device):
        """
        :param hidden_size: the dimension of linear hidden layer
        :param t_end: the time when observation terminates
                      if is None, then time after last event is not considered
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.t_end = t_end
        self.device = device
        
        # log background intensity 
        self.background = torch.nn.Parameter(torch.ones(1))
        
        # λ
        self.f = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softplus()
        )
        
    def forward(self, seq_pads, seq_lens): 
        """
        Calculate NLL for a batch of sequence
        
        :param seq_pads: [batch, maxlen, 1], the padded event timings
        :param seq_lens: [batch], the sequence length before padding
        :return nll: scalar, the average negative log likelihood
        """
        batch, maxlen, _ = seq_pads.shape
        t_last = torch.gather(seq_pads, 1, torch.tensor(seq_lens).to(self.device).view(-1, 1, 1) - 1).squeeze()
        
        # Perform outer subtraction
        seq_pads_roll = torch.cat((torch.zeros(batch, 1, 1).to(self.device), 
                                   seq_pads.transpose(1, 2)[..., :-1]), -1)
        diff_pads = seq_pads - seq_pads_roll
        tril_idx = torch.tril_indices(maxlen - 1, maxlen - 1)
        diff_pads = diff_pads[:, tril_idx[0] + 1, tril_idx[1] + 1].unsqueeze(-1)
        
        ########## Calculate intensity ############
        # [batch, seq_len]
        # intensity before every event
        lambs = self.f(diff_pads).squeeze(-1)
        lambs = lambs.split(list(range(1, maxlen)), dim=-1)
        lambs = torch.stack([lamb.sum(dim=-1) for lamb in lambs], -1)
        lambs = torch.cat((torch.zeros(batch, 1).to(self.device), lambs), -1)  # First event zero influence
        lambs += self.background  # Add background intensity
        
        ######### Calculate integral range ##########
        # [batch, seq_len]
        # cumulative influence of every event
        diff_pads_with_last = []
        for seq_pad, seq_len in zip(seq_pads, seq_lens):
            if self.t_end is None:
                t_end = seq_pad[seq_len - 1]
            else:
                t_end = self.t_end
            temp = t_end - seq_pad[:seq_len]
            temp = torch.cat((temp, torch.zeros(maxlen - seq_len, 1).to(self.device)))
            diff_pads_with_last.append(temp)
        diff_pads_with_last = torch.stack(diff_pads_with_last)
        
        ######### Apply chebyshev integration ##########
        def f(x):
            if x.shape[-1] != 1:
                x = x.unsqueeze(-1)  # Add one dimension
            return self.f(x).squeeze()
            
        coefs = chebyCoef(f, 1000, self.t_end, 0.0)
        x = diff_pads_with_last.view(-1)
        lamb_ints = chebyInt(x, coefs, self.t_end, 0.0)
        lamb_ints = lamb_ints.view(batch, maxlen)
        
        ######### Calculate loss ########
        sum_log_lambs = []
        for lamb, seq_len in zip(lambs, seq_lens):
            sum_log_lambs.append(torch.log(lamb[:seq_len]).sum())
            
        lamb_ints = torch.sum(lamb_ints, -1)
        if self.t_end is None:
            background_int = t_last * self.background
        else:
            background_int = self.t_end * self.background
        lamb_ints += background_int  # Add background integral
        
        nll = - (sum(sum_log_lambs) - sum(lamb_ints)) / batch
        
        return nll


class TaylorSameInfluenceProcess(nn.Module):
    
    '''
    :param hidden_size: the dimension of linear hidden layer
    :param t_end: the time when observation terminates
                  if is None, then time after last event is not considered
    '''
    def __init__(self, hidden_size, t_end, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.t_end = t_end
        self.device = device
        
        # log background intensity 
        self.background = torch.nn.Parameter(torch.ones(1))
        
        # λ
        self.f = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softplus()
        )
        
        # Taylor integral utility  
        self.tint = TaylorInt(0.0, t_end, 0.05, device).to(device)
             
    def forward(self, seq_pads, seq_lens):               
        """
        Calculate NLL for a batch of sequence
        
        :param seq_pads: [batch, maxlen, 1], the padded event timings
        :param seq_lens: [batch], the sequence length before padding
        :return nll: scalar, the average negative log likelihood
        """
        batch, maxlen, _ = seq_pads.shape
        t_last = torch.gather(seq_pads, 1, torch.tensor(seq_lens).to(self.device).view(-1, 1, 1) - 1).squeeze()
        
        # Perform outer subtraction
        seq_pads_roll = torch.cat((torch.zeros(batch, 1, 1).to(self.device), 
                                   seq_pads.transpose(1, 2)[..., :-1]), -1)
        diff_pads = seq_pads - seq_pads_roll
        tril_idx = torch.tril_indices(maxlen - 1, maxlen - 1)
        diff_pads = diff_pads[:, tril_idx[0] + 1, tril_idx[1] + 1].unsqueeze(-1)
        
        ########## Calculate intensity ############
        # [batch, seq_len]
        # intensity before every event
        lambs = self.f(diff_pads).squeeze(-1)
        lambs = lambs.split(list(range(1, maxlen)), dim=-1)
        lambs = torch.stack([lamb.sum(dim=-1) for lamb in lambs], -1)
        lambs = torch.cat((torch.zeros(batch, 1).to(self.device), lambs), -1)  # First event zero influence
        lambs += self.background  # Add background intensity
         
        ######### Calculate integral range ##########
        # [batch, seq_len]
        # cumulative influence of every event
        diff_pads_with_last = []
        for seq_pad, seq_len in zip(seq_pads, seq_lens):
            if self.t_end is None:
                t_end = seq_pad[seq_len - 1]
            else:
                t_end = self.t_end
            temp = t_end - seq_pad[:seq_len]
            temp = torch.cat((temp, torch.zeros(maxlen - seq_len, 1).to(self.device)))
            diff_pads_with_last.append(temp)
        diff_pads_with_last = torch.stack(diff_pads_with_last)
          
        ######### Apply taylor integration ##########
        def f(x):
            if x.shape[-1] != 1:
                x = x.unsqueeze(-1)  # Add one dimension
            return self.f(x).squeeze()
        
        x = diff_pads_with_last.view(-1)
        lamb_ints = self.tint(f, x)
        lamb_ints = lamb_ints.view(batch, maxlen)
        
        ######### Calculate loss ########
        sum_log_lambs = []
        for lamb, seq_len in zip(lambs, seq_lens):
            sum_log_lambs.append(torch.log(lamb[:seq_len]).sum())
            
        lamb_ints = torch.sum(lamb_ints, -1)
        if self.t_end is None:
            background_int = t_last * self.background
        else:
            background_int = self.t_end * self.background
        lamb_ints += background_int  # Add background integral
        
        nll = - (sum(sum_log_lambs) - sum(lamb_ints)) / batch
        
        return nll
