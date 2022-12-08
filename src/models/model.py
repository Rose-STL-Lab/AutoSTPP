import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from integration.autoint import MultSequential, PadLinear, NonNegLinear

eps = 1e-10  # A negligible positive number
np.random.seed(0)


class AutoIntTPPSameInfluence(nn.Module):

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

        # ∫_0^t λ
        self.F = MultSequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.project()

    def project(self):
        """Employ non-negative constraint"""
        with torch.no_grad():
            self.F[0].weight.clamp_(0.0)
            self.F[2].weight.clamp_(0.0)
            self.F[4].weight.clamp_(0.0)
            self.F[6].weight.clamp_(0.0)

    def forward(self, seq_pads, seq_lens):
        """
        Calculate NLL for a batch of sequence
        
        :param seq_pads: [batch, maxlen, 1], the padded event timings
        :param seq_lens: [batch], the sequence length before padding
        :return: scalar, the average negative log likelihood
        """
        batch, maxlen, _ = seq_pads.shape
        t_last = torch.gather(seq_pads, 1, torch.tensor(seq_lens).to(self.device).view(-1, 1, 1) - 1).squeeze()

        # Perform outer subtraction
        seq_pads_roll = torch.cat((torch.zeros(batch, 1, 1).to(self.device), seq_pads.transpose(1, 2)[..., :-1]), -1)
        diff_pads = seq_pads - seq_pads_roll
        tril_idx = torch.tril_indices(maxlen - 1, maxlen - 1)
        diff_pads = diff_pads[:, tril_idx[0] + 1, tril_idx[1] + 1].unsqueeze(-1)

        ########## Calculate intensity ############
        # [batch, seq_len]
        # intensity before every event
        lambs = self.F.dforward(diff_pads, 0).squeeze(-1)
        lambs = lambs.split(list(range(1, maxlen)), dim=-1)
        lambs = torch.stack([lamb.sum(dim=-1) for lamb in lambs], -1)
        lambs = torch.cat((torch.zeros(batch, 1).to(self.device), lambs), -1)  # first event zero influence
        lambs += self.background  # add background intensity

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
        # print(diff_pads_with_last[0, ..., 0])

        lamb_ints = self.F(diff_pads_with_last).squeeze(-1)
        lamb_ints = lamb_ints - self.F(torch.zeros(1).to(self.device))  # remove F(0)

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

        nll = -(sum(sum_log_lambs) - sum(lamb_ints)) / batch

        return nll


class AutoIntGRUTPPSameInfluence(nn.Module):

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

        # intensity magnitude (alpha) encoder
        self.RNN = nn.GRU(1, hidden_size, batch_first=True, num_layers=1)

        # ... decoder
        self.alpha = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1), nn.Sigmoid()
        )

        # ∫_0^t λ
        self.F = MultSequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.project()

    def project(self):
        """Employ non-negative constraint"""
        with torch.no_grad():
            self.F[0].weight.clamp_(0.0)
            self.F[2].weight.clamp_(0.0)
            self.F[4].weight.clamp_(0.0)
            self.F[6].weight.clamp_(0.0)

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
        seq_pads_roll = torch.cat((torch.zeros(batch, 1, 1).to(self.device), seq_pads.transpose(1, 2)[..., :-1]), -1)
        diff_pads = seq_pads - seq_pads_roll
        tril_idx = torch.tril_indices(maxlen - 1, maxlen - 1)
        diff_pads = diff_pads[:, tril_idx[0] + 1, tril_idx[1] + 1].unsqueeze(-1)

        # Convert absolute timing to timing difference
        delta_t_pads = -torch.ones_like(seq_pads)
        for i, seq_len in enumerate(seq_lens):
            delta_t_pads[i, 0] = seq_pads[i, 0]  # retain first event timing
            delta_t_pads[i, 1:seq_len] = seq_pads[i, 1:seq_len] - seq_pads[i, : seq_len - 1]

        ########## Calculate magnitude ############
        delta_t_input = pack_padded_sequence(delta_t_pads, seq_lens, batch_first=True, enforce_sorted=False)
        hidden, _ = self.RNN(delta_t_input)
        hidden, _ = pad_packed_sequence(hidden, padding_value=0, batch_first=True)
        alphas = self.alpha(hidden).squeeze(-1) + 1.0

        ########## Calculate intensity ############
        # [batch, seq_len]
        # intensity before every event
        lambs = self.F.dforward(diff_pads, 0).squeeze(-1)
        lambs = lambs.split(list(range(1, maxlen)), dim=-1)
        lambs = torch.stack([lamb.sum(dim=-1) for lamb in lambs], -1)
        lambs = torch.cat((torch.zeros(batch, 1).to(self.device), lambs), -1)  # first event zero influence
        lambs *= alphas
        lambs += self.background  # add background intensity

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
        # print(diff_pads_with_last[0, ..., 0])

        lamb_ints = self.F(diff_pads_with_last).squeeze(-1)
        lamb_ints = lamb_ints - self.F(torch.zeros(1).to(self.device))  # remove F(0)

        ######### Calculate loss ########
        sum_log_lambs = []
        for lamb, seq_len in zip(lambs, seq_lens):
            sum_log_lambs.append(torch.log(lamb[:seq_len]).sum())

        lamb_ints = torch.sum(lamb_ints * alphas, -1)
        if self.t_end is None:
            background_int = t_last * self.background
        else:
            background_int = self.t_end * self.background
        lamb_ints += background_int  # Add background integral

        nll = -(sum(sum_log_lambs) - sum(lamb_ints)) / batch

        return nll


class FullyTemporalPointProcess(nn.Module):

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

        # ∫_0^t λ
        self.F = MultSequential(
            PadLinear(hidden_size, hidden_size),
            nn.Tanh(),
            PadLinear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size + 1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.project()

    def project(self):
        """Employ non-negative constraint"""
        with torch.no_grad():
            self.F[-3].weight.clamp_(0.0)
            self.F[-1].weight.clamp_(0.0)

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
        delta_t_pads = -torch.ones_like(seq_pads)
        for i, seq_len in enumerate(seq_lens):
            delta_t_pads[i, 0] = seq_pads[i, 0]  # retain first event timing
            delta_t_pads[i, 1:seq_len] = seq_pads[i, 1:seq_len] - seq_pads[i, : seq_len - 1]

        delta_t_input = pack_padded_sequence(delta_t_pads, seq_lens, batch_first=True, enforce_sorted=False)

        # Use RNN to generate hidden representation after each event
        hidden, _ = self.RNN(delta_t_input)
        hidden, _ = pad_packed_sequence(hidden, padding_value=0, batch_first=True)  # [batch, seq_len, 128]

        # Add the hidden vector (all 0) before the first event
        hidden = torch.cat([torch.zeros(len(hidden), 1, self.hidden_size).to(self.device), hidden], 1)

        # Extend delta_t_pads to include after last event
        if t_last is not None:
            delta_t_pads = torch.cat([delta_t_pads, -torch.ones(len(seq_lens), 1, 1).to(self.device)], 1)
            for i, seq_len in enumerate(seq_lens):
                delta_t_pads[i, seq_len] = self.t_end - t_last[i]

        return hidden, delta_t_pads

    @staticmethod
    def th(times, seq_pads, hidden):
        """
        Utility for preparing the input to λ/F at certain time points
        
        :param times: [batch, N, 1], the time points to evaluate intensity
        :param seq_pads: [batch, maxlen, 1], the padded event timings
        :param hidden: [batch, maxlen+1, hiddim], the representation of history from h0 to hn
        :return inp: [batch, N, hiddim+1], the [t, h] input at the time points 
        """
        temp = torch.sum(times.transpose(2, 1) <= seq_pads, 1)  # find reverse index
        idx_max, _ = torch.max(temp, 1)
        idxs = idx_max.unsqueeze(-1) - temp  # number of event before each time
        h_select = torch.stack([torch.index_select(h, 0, idx) for h, idx in zip(hidden, idxs)])
        temp = torch.cat([torch.zeros(len(seq_pads), 1, 1).to(seq_pads.device), seq_pads], 1)
        t_select = torch.stack([torch.index_select(t, 0, idx) for t, idx in zip(temp, idxs)])
        return torch.cat([times - t_select, h_select], -1)  # concatenate time and hidden

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

        if include_after:
            lamb_input = torch.cat([delta_t_pads, hidden], -1)
            lambs = self.F.dforward(lamb_input, 0) + eps
            lambs = lambs[:, :-1]
        else:
            lamb_input = torch.cat([delta_t_pads, hidden[:, :-1]], -1)
            lambs = self.F.dforward(lamb_input, 0) + eps

        Ft = self.F(lamb_input)  # Use buffered result
        lamb_input_t0 = torch.cat([torch.zeros_like(lamb_input[..., :1]), lamb_input[..., 1:]], -1)
        F0 = self.F(lamb_input_t0)
        lamb_ints = Ft - F0

        loglike = []
        for lamb_int, lamb, seq_len in zip(lamb_ints, lambs, seq_lens):
            length = seq_len + 1 if include_after else seq_len
            loglike.append(torch.log(lamb[:seq_len]).sum() - lamb_int[:length].sum())  # NLL

        return -sum(loglike) / len(loglike)
