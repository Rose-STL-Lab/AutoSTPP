from tqdm.auto import trange

import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import inspect

from copy import deepcopy


class TPPWrapper(Dataset):
    """
    Wrap data of a temporal point process
    """

    def __init__(self, lamb_func, n_sample, t_end, max_lamb,
                 fn=None, n_start=0, seed=0, verbose=False):
        self.lamb_func = lamb_func
        self.n_sample = n_sample
        self.t_end = t_end
        self.max_lamb = max_lamb
        self.seqs = []
        np.random.seed(seed)

        if fn is not None:
            self.seqs = [seq[seq < t_end] for seq in torch.load(fn)[n_start:n_start + n_sample]]
        else:
            for _ in trange(n_sample):
                self.seqs.append(torch.tensor(self.generate(verbose)))

    def save(self, name):
        torch.save(self.seqs, f'{name}.db')
        with open(f'{name}.info', 'w') as file:
            file.write(inspect.getsource(self.lamb_func) + '\n')
            file.write(f'n_sample = {self.n_sample}\n')
            file.write(f't_end = {self.t_end}\n')
            file.write(f'max_lamb = {self.max_lamb}\n')
            file.close()

    def generate(self, verbose=False):
        """
        Generate event timing sequence governed by temporal point process
        """
        if verbose:
            print(f'Generating events from t=0 to t={self.t_end}')

        t = 0.0
        his_t = np.array([])

        while True:
            # Calculate the maximum intensity
            lamb_t, L, M = self.lamb_func(t, his_t)
            delta_t = np.random.exponential(scale=1 / M)
            if lamb_t > self.max_lamb:  # Discarding the sequence
                return self.generate(verbose)
            if t + delta_t > self.t_end:
                break
            if delta_t > L:
                t += L
                continue
            else:
                t += delta_t
                new_lamb_t, _, _ = self.lamb_func(t, his_t)

                if new_lamb_t / M >= np.random.uniform():  # Accept the sample
                    if verbose:
                        print("----")
                        print(f"t:  {t}")
                        print(f"λt: {new_lamb_t}")
                    # Draw a location
                    his_t = np.append(his_t, t)

        return his_t

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def pad_collate(batch):
    seq_lens = [len(seq) for seq in batch]
    t_last = torch.tensor([seq[-1] for seq in batch])
    seq_pads = pad_sequence(batch, batch_first=True, padding_value=-1)
    return seq_pads.unsqueeze(-1), seq_lens, t_last


class SlidingWindowWrapper(torch.utils.data.Dataset):
    """
    Wrap data of a spatiotemporal point process
    """

    def __init__(self, seqs, lookback=20, lookahead=1, normalized=False, roll=True, min=None, max=None, 
                 device=torch.device("cuda:0")):
        """
        Take a batch of sequences, applying sliding window to each of it to create a
        fixed length dataset.
        
        st_X: torch.tensor, [N, lookback]
        st_Y: torch.tensor, [N, lookahead]

        :param seqs: a list of sequences of np shape [N, 3], time is the first dimension
        :param roll: whether to roll the time to the last dimension
        """
        self.seqs_cum = seqs
        self.seqs = deepcopy(self.seqs_cum)
        for i, seq in enumerate(self.seqs):
            self.seqs[i][:, 0] = np.diff(seq[:, 0], axis=0, prepend=0)

        if roll:
            self.seqs_cum = [np.roll(seq, -1, -1) for seq in self.seqs_cum]
            self.seqs = [np.roll(seq, -1, -1) for seq in self.seqs]

        self.st_X = []
        self.st_Y = []
        self.st_X_cum = []
        self.st_Y_cum = []
        self.indices = []

        # Create normalizer
        if normalized and (min is None or max is None):
            temp = np.vstack(self.seqs)
            self.min = torch.tensor(np.min(temp, 0)).float().to(device)
            self.max = torch.tensor(np.max(temp, 0)).float().to(device)
        elif normalized:
            self.min = min
            self.max = max

        for seq_i, (seq, seq_cum) in enumerate(zip(self.seqs, self.seqs_cum)):
            for i in range(lookback, len(seq) + 1 - lookahead):
                self.st_X_cum.append(seq_cum[i - lookback: i])
                self.st_Y_cum.append(seq_cum[i: i + lookahead])

                self.st_X.append(seq[i - lookback: i])
                self.st_Y.append(seq[i: i + lookahead])

                self.indices.append((seq_i, i))  # Get the location in original sequence

        self.st_X = torch.tensor(np.stack(self.st_X)).float().to(device)
        self.st_Y = torch.tensor(np.stack(self.st_Y)).float().to(device)

        self.st_X_cum = torch.tensor(np.stack(self.st_X_cum)).float().to(device)
        self.st_Y_cum = torch.tensor(np.stack(self.st_Y_cum)).float().to(device)

        if normalized:
            def scale(st):
                return (st - self.min) / (self.max - self.min)

            self.st_X = scale(self.st_X)
            self.st_Y = scale(self.st_Y)

    def __len__(self):
        return len(self.st_X)

    def __getitem__(self, idx):
        """
        :return:
            - normalized sequence diff
            - un-normalized original sequence
        """
        return self.st_X[idx], self.st_Y[idx], self.st_X_cum[idx], self.st_Y_cum[idx], self.indices[idx]
