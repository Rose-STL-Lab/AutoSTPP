import torch
from torch import nn

from src.integration.autoint import Cuboid


class AutoIntSTPPSameInfluence(nn.Module):

    def __init__(self, hidden_size, t_end, device):
        """
        :param: hidden_size: the dimension of linear hidden layer
        :param: t_end: the time when observation terminates
                if is None, then time after last event is not considered
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.t_end = t_end
        self.device = device

        # log background intensity
        self.background = torch.nn.Parameter(torch.ones(1))

        # ∫_0^t λ
        self.F = Cuboid().to(device)

        self.project()

    def project(self):
        """
        Employ non-negative constraint
        """
        self.F.project()

    def forward(self, seq_pads, seq_lens):
        """
        Calculate NLL for a batch of sequence

        :param seq_pads: [batch, maxlen, 3], the padded event timings
        :param seq_lens: [batch], the sequence length before padding
        :return: nll: scalar, the average negative log likelihood
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

        nll = - (sum(sum_log_lambs) - sum(lamb_ints)) / batch

        return nll
