from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

import abc

from scipy.integrate import quad, quad_vec
from utils import arange

eps = 1e-10  # A negligible positive number


def spatiotemporal_events_collate_fn(data):
    """
    Padded the spatiotemporal sequences with different lengths to the same length from the front.

    :param data: a list of tensors with shape [T, 1 + D], where T may be different for each tensor.
    :return:
        - data: padded data with shape [batch, max_len, 1 + D]
        - mask: non-padding indices (except last) with shape [batch, max_len]
    """
    dim = data[0].shape[1]
    lengths = [seq.shape[0] for seq in data]
    max_len = max(lengths)
    padded_seqs = [torch.cat([torch.zeros(max_len - s.shape[0], dim).to(s), s], 0)  # From the front
                   if s.shape[0] != max_len else s for s in data]
    data = torch.stack(padded_seqs, dim=0)
    mask = torch.stack([torch.cat([torch.zeros(max_len - seq_len), torch.ones(seq_len - 1),
                                   torch.zeros(1)], dim=0)
                        for seq_len in lengths])
    return data, mask


class ListDataset(torch.utils.data.Dataset):
    """
    Dataset loading list of tensors, use with `spatiotemporal_events_collate_fn`
    """
    def __init__(self, data):
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return torch.tensor(self.dataset[index], dtype=torch.float)


class SyntheticDataset(abc.ABC):
    """
    Abstract class, parent class of all synthetic dataset

    :ivar his_s, his_t, t_start, t_end: to be initialized by `self.generate`
    :ivar train, val, test: to be initialized by `self.dataset`
    """
    def __init__(self, dist_only=False):
        self.his_s, self.his_t, self.t_start, self.t_end = None, None, None, None
        self.train, self.val, self.test = None, None, None
        self.st_scaler = None
        self.dist_only = dist_only

    @abc.abstractmethod
    def generate(self, t_start, t_end):
        """
        Generate long event sequence
        """
        pass

    @staticmethod
    def g0(s, s_mu, s_sqrt_inv_det_cov, s_inv_cov):
        """
        g0(s) = 1 / 2π * |Σ|^(-0.5) * exp{ - 0.5 (s-mean_s) Σ^(-1) (s-mean_s)' }
        
        :param s: shape [2], or shape [N, 2]
        :param s_mu: mean of the spatial distribution, shape [2], no broadcasting
        :param s_sqrt_inv_det_cov: float, square root of the inverse of the determinant of the covariance matrix
        :param s_inv_cov: shape [2, 2], inverse of the covariance matrix
        :return: shape [1], or shape [N]
        """
        assert s_mu.shape == (2,), "Mean shape should be [2,]"
        result = SyntheticDataset.g2(s, s_mu, s_sqrt_inv_det_cov, s_inv_cov)
        if len(s.shape) == 2:
            result = result.squeeze(-1)
        return result

    @staticmethod
    def g1(t, his_t, alpha, beta):
        """
        g1(δt) = α * exp{ - β δt }
        
        :param t: float, or shape [N]
        :param his_t: [his_len,], or [N, his_len,]
        :return: shape [his_len,] or shape [N, his_len]
        """
        flag = False
        if type(t) is np.ndarray and np.prod(t.shape) != 1:
            t = np.reshape(t, (-1, 1))
            flag = True
        if flag and len(his_t) == 1:  # Handle multiple t and single his_t
            his_t = np.reshape(his_t, (1, -1))
        delta_t = t - his_t
        return alpha * np.exp(-beta * delta_t)

    @staticmethod
    def g2(s, his_s, s_sqrt_inv_det_cov, s_inv_cov):
        """
        g2(δs) = 1 / 2π * |Σ|^(-0.5) * exp{ - 0.5 δs Σ^(-1) δs' }
        
        :param s: shape [2], or shape [N, 2]
        :param his_s: [his_len, 2], or [N, his_len, 2], or [2] (for s_mu)
        :param s_sqrt_inv_det_cov: float, square root of the inverse of the determinant of the covariance matrix
        :param s_inv_cov: shape [2, 2], inverse of the covariance matrix
        :return: shape [his_len], or shape [N, his_len]
        """
        if flag := len(s.shape) == 1:
            s = np.reshape(s, (1, 1, 2))
        else:
            s = np.reshape(s, (-1, 1, 2))
        if len(his_s.shape) == 2:
            his_s = np.reshape(his_s, (1, *his_s.shape))
            
        delta_s = s - his_s
        result = 1 / 2 / np.pi * s_sqrt_inv_det_cov * \
                 np.exp(-np.einsum('kij,kij->ki', delta_s.dot(s_inv_cov), delta_s) / 2)
        if flag and len(result) == 1:
            result = result[0]
        return result

    def save(self, text_path):
        """
        Export generated sequence as text file

        :param text_path: path to save the his_s and his_t in txt format.
        """
        np.savetxt(text_path, np.hstack([self.his_s, np.expand_dims(self.his_t, 1)]), delimiter=',', fmt='%f')

    def load(self, text_path, t_start, t_end):
        """
        Load saved sequence. The class's his_s and his_t is filled by the stored data.

        :param text_path: a txt file or npz file containing the data,
                          every line is "s0,s1,t", t is monotonically increasing
        :param t_start: lines with t < t_start is omitted.
        :param t_end: lines with t > t_end is omitted.
        """
        self.t_start = t_start
        self.t_end = t_end

        his_st = np.loadtxt(text_path, delimiter=',')
        self.his_s = his_st[:, :2]
        self.his_t = his_st[:, 2]

        idx = np.logical_and(self.his_t >= t_start, self.his_t < t_end)
        self.his_t = self.his_t[idx]
        self.his_s = self.his_s[idx]

        if isinstance(self, DEBMDataset):
            self.his_t[0] -= eps

    def dataset(self, lookback=10, lookahead=1, split=None):
        """
        Create train, val, test for model training & testing
        """
        if self.dist_only:
            # Use Euclidean distance
            temp = np.sum(np.square(np.diff(self.his_s, axis=0)), axis=1)
            dist = np.expand_dims(np.append(0, np.sqrt(temp)), 1)
            st_data = np.hstack((dist, np.expand_dims((self.his_t), 1)))
        else:
            st_data = np.hstack((self.his_s, np.expand_dims((self.his_t), 1)))

        # Time -> delta_t
        st_data[:, -1][1:] = np.diff(st_data[:, -1])
        st_data[:, -1][0] = 0

        # Default split is train:val:test = 8:1:1
        if split is None:
            split = [8, 1, 1]
        split = split / np.sum(split)

        length = len(st_data) - lookback - lookahead

        # Min-max scale spatiotemporal data
        self.st_scaler = MinMaxScaler()
        self.st_scaler.fit(st_data)
        st_data = self.st_scaler.transform(st_data)

        # Breaking sequence to training data: [1-1303] -> [1 ~ lookback][lookback+1],
        # [2 ~ lookback+1][lookback+2]...
        if self.dist_only:
            num_features = 2
        else:
            num_features = 3
        st_input = np.zeros((length, lookback, num_features))
        st_label = np.zeros((length, lookahead, num_features))

        for i in range(length):
            st_input[i] = st_data[i:i + lookback]
            st_label[i] = st_data[i + lookback:i + lookback + lookahead]

        train_size = int(split[0] * length)
        test_size = int(split[2] * length)

        self.train = TensorDataset(torch.Tensor(st_input[:train_size]),
                                   torch.Tensor(st_label[:train_size]))

        self.val = TensorDataset(torch.Tensor(st_input[train_size:-test_size]),
                                 torch.Tensor(st_label[train_size:-test_size]))

        self.test = TensorDataset(torch.Tensor(st_input[-test_size:]),
                                  torch.Tensor(st_label[-test_size:]))

        print("Finished.")

    def dataset_fixedtime(self, time_interval=30, split=None):
        if self.dist_only:
            # Use Euclidean distance
            temp = np.sum(np.square(np.diff(self.his_s, axis=0)), axis=1)
            dist = np.expand_dims(np.append(0, np.sqrt(temp)), 1)
            st_data = np.hstack((dist, np.expand_dims(self.his_t, 1)))
        else:
            st_data = np.hstack((self.his_s, np.expand_dims(self.his_t, 1)))

        # Time -> delta_t
        t_absolute = np.array(st_data[:, -1])
        st_data[:, -1][1:] = np.diff(st_data[:, -1])
        st_data[:, -1][0] = 0

        # Default split is train:val:test = 8:1:1
        if split is None:
            split = [8, 1, 1]
        split = split / np.sum(split)

        # Min-max scale spatiotemporal data
        self.st_scaler = MinMaxScaler()
        self.st_scaler.fit(st_data)
        st_data = self.st_scaler.transform(st_data)

        st_input = []
        length = len(range(0, self.t_end, time_interval))
        for start_t in range(0, self.t_end, time_interval):
            index = np.logical_and(t_absolute > start_t, t_absolute < start_t + time_interval)
            seq = st_data[index]
            if len(seq) > 1:
                st_input.append(seq)

        train_size = int(split[0] * length)
        test_size = int(split[2] * length)

        print('Max len: ', max([len(seq) for seq in st_input]))

        self.train = ListDataset(st_input[:train_size])
        self.val = ListDataset(st_input[train_size:-test_size])
        self.test = ListDataset(st_input[-test_size:])

        print("Finished.")
        
    def get_lamb_st(self, x_num, y_num, t_num, t_start, t_end):
        """
        return lamb_st for a given time range
        
        :param x_num: int, resolution of x
        :param y_num: int, resolution of y
        :param t_num: int, resolution of t
        :return lambs: List of len (t_num+1), element np array [x_num+1, y_num+1]
        :return x_range: np array [x_num+1]
        :return y_range: np array [y_num+1]
        :return t_range: np array [t_num+1]
        """
        idx = np.logical_and(self.his_t >= t_start, self.his_t < t_end)

        his_s = self.his_s[idx]

        x_max, y_max = np.max(his_s, 0)
        x_min, y_min = np.min(his_s, 0)

        x_range = np.linspace(x_min, x_max, x_num)
        y_range = np.linspace(y_min, y_max, y_num)
        t_range = np.linspace(t_start, t_end, t_num)

        lambs = []
        for t in tqdm(t_range):
            lamb_st = np.zeros((x_num, y_num))
            for i, x in enumerate(x_range):
                for j, y in enumerate(y_range):
                    lamb_st[i, j] = self.lamb_st(np.array([[x, y]]), t)
            lambs.append(lamb_st)
        return lambs, x_range, y_range, t_range


class STSCPDataset(SyntheticDataset):
    """
    Simulate Spatio-Temporal Self-Correcting Process
    """
    def __init__(self, g0_cov, g2_cov, alpha, beta, mu, gamma, lamb_max=10,
                 max_history=100, x_num=51, y_num=51, dist_only=False):
        """
        :param g0_cov: np array with shape (2, 2)
        :param g2_cov: np array with shape (2, 2)
        """
        SyntheticDataset.__init__(self, dist_only)
        self.g0_cov = g0_cov
        self.g2_cov = g2_cov
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.gamma = gamma
        self.lamb_max = lamb_max
        self.max_history = max_history

        # Discretization
        self.x_num, self.y_num = x_num, y_num
        x_range = np.linspace(0, 1, x_num)
        y_range = np.linspace(0, 1, y_num)
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        num_point = x_num * y_num
        s_grids = np.c_[grid_x.ravel(), grid_y.ravel()]

        self.x_range, self.y_range, self.s_grids = x_range, y_range, s_grids
        self.t_range, self.lambs = None, []

        # Pre-calculate g0 and g2
        g0_mat = self.g0(s_grids, np.array([0.5, 0.5]), 1 / np.sqrt(np.linalg.det(g0_cov)), np.linalg.inv(g0_cov))
        g0_mat = g0_mat / np.sum(g0_mat) * x_num * y_num * mu

        g2_mats = np.zeros((num_point, num_point))
        for i, s in tqdm(enumerate(s_grids)):
            g2_cov = self.g2_cov / (gamma * np.sum(np.square(s - np.array([0.5, 0.5]))) + 1)
            g2_mats[i] = self.g2(s_grids, s, 1 / np.sqrt(np.linalg.det(g2_cov)), np.linalg.inv(g2_cov)).squeeze(-1)
            g2_mats[i] = g2_mats[i] / np.sum(g2_mats[i]) * x_num * y_num

        self.g0_mat, self.g2_mats = g0_mat, g2_mats

    def lamb_st(self, mu, his_s, s, t):
        """
        mu is a [x_num * y_num] array that stores the initial intensity
        λ(s,t|H) = g0(s) μ exp(βt - α ∑g2(δs))
        """
        i = self.stoi(s)
        lamb_st = mu[i] * np.exp(self.g0_mat[i] * self.beta * t - self.alpha *
                                 np.sum(self.g2_mats[self.stoi(his_s), i]))
        return np.minimum(self.lamb_max * np.ones(lamb_st.shape), lamb_st)

    def lamb_St(self, mu, his_s, t):
        """
        Calculate intensity all over the space
        """
        lamb_St = mu * np.exp(self.g0_mat * self.beta * t - self.alpha *
                              np.sum(self.g2_mats[self.stoi(his_s)].reshape(-1, self.x_num * self.y_num), 0))
        return np.minimum(self.lamb_max * np.ones(lamb_St.shape), lamb_St)

    def lamb_t(self, mu, his_s, t):
        """
        λ(t|H) = ∫ λ(s,t|H)
        """
        return np.sum(self.lamb_St(mu, his_s, t)) / self.x_num / self.y_num

    def stoi(self, s):
        """
        spatial point -> discretized index
        """
        return np.round(s * np.array([[self.x_num - 1, self.y_num - 1]])) \
            .dot(np.array([[1], [self.y_num]])).astype(int).squeeze()

    def generate(self, t_start, t_end, t_num=None, verbose=False):
        """
        Generate long event sequence

        t_num: number of steps per max_history
        """
        self.t_start = t_start
        self.t_end = t_end
        if t_num is None:
            t_num = self.max_history * 2

        self.t_range = np.array([t_start, ])
        self.lambs = [self.mu * np.ones(self.x_num * self.y_num), ]  # Start at an uniform distribution

        self.his_s = np.zeros((0, 2))
        self.his_t = np.array([])

        while self.t_range[-1] < self.t_end:
            t_range, lambs, his_s, his_t = self.generate_batch(self.t_range[-1], t_num, self.lambs[-1], verbose)
            self.lambs += lambs
            self.t_range = np.append(self.t_range, t_range)
            self.his_s = np.vstack((self.his_s, his_s))
            self.his_t = np.append(self.his_t, his_t)

        self.lambs = [lamb_st.reshape(self.x_num, self.y_num).transpose() for lamb_st in self.lambs]

    def generate_batch(self, t_start, t_num, mu, verbose=False):
        """
        mu is a [x_num * y_num] array that stores the initial intensity
        Generate event sequence from t=0 to t=self.max_history
        """
        if verbose:
            print(f"Generating events from t={t_start} to t={t_start + self.max_history}")

        t_end = self.max_history
        t = 0
        his_s = np.zeros((0, 2))
        his_t = np.array([])

        t_range = np.linspace(0, t_end, t_num)[1:]
        lambs = []

        while True:
            # Calculate the max intensity
            lamb_t = self.lamb_t(mu, his_s, t)
            l = 2 / lamb_t
            m = lamb_t * np.exp(self.beta * l)
            delta_t = np.random.exponential(scale=1 / m)

            if t + delta_t > t_end:
                break
            if delta_t > l:
                t += l
                continue
            else:
                t += delta_t
                new_lamb_t = self.lamb_t(mu, his_s, t)
                if new_lamb_t / m >= np.random.uniform():  # Accept the sample
                    lamb_st = self.lamb_St(mu, his_s, t)

                    if verbose:
                        print("----")
                        print(f"t:  {t + t_start}")
                        print(f"λt: {new_lamb_t}")

                    p = lamb_st / np.sum(lamb_st)
                    # Draw a location
                    i = np.argmax(np.random.multinomial(1, p))
                    his_s = np.vstack((his_s, self.s_grids[i]))
                    his_t = np.append(his_t, t)

        if verbose:
            print("===")

        # Calculate lamb_st
        for t in t_range:
            lambs.append(self.lamb_St(mu, his_s[his_t < t], t))
        return t_range + t_start, lambs, his_s, his_t + t_start

    def get_lamb_st(self, x_num, y_num, t_num, t_start, t_end):
        """
        return lamb_st for a given time range
        """
        idx = np.logical_and(self.his_t >= t_start, self.his_t < t_end)
        # his_t = self.his_t[idx]
        his_s = self.his_s[idx]

        x_max, y_max = np.max(his_s, 0)
        x_min, y_min = np.min(his_s, 0)

        x_range = np.linspace(x_min, x_max, x_num)
        y_range = np.linspace(y_min, y_max, y_num)
        t_range = np.linspace(t_start, t_end, t_num)

        mu = self.mu * np.ones(self.x_num * self.y_num)
        lambs = [mu, ]
        for t in tqdm(t_range[1:]):
            lamb_st = self.lamb_St(mu, self.his_s[self.his_t < t], t)
            lambs.append(lamb_st)

        lambs = [lamb_st.reshape(self.x_num, self.y_num).transpose() for lamb_st in lambs]
        return lambs, x_range, y_range, t_range

    def get_lamb_st_(self, t_start, t_end):
        idx = np.argwhere(np.logical_and(self.t_range >= t_start, self.t_range < t_end)).squeeze()
        return [self.lambs[i] for i in idx], self.x_range, self.y_range, self.t_range[idx]


class STHPDataset(SyntheticDataset):
    """
    Simulate Spatio-Temporal Hawkes Process
    """
    def __init__(self, s_mu, g0_cov, g2_cov, alpha, beta, mu, max_history=100, dist_only=False):
        """
        :param s_mu:   np array with shape (2,  )
        :param g0_cov: np array with shape (2, 2)
        :param g2_cov: np array with shape (2, 2)
        """
        SyntheticDataset.__init__(self, dist_only)
        self.s_mu = s_mu
        self.g0_cov = g0_cov
        self.g2_cov = g2_cov
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.max_history = max_history

        self.g0_ic = np.linalg.inv(g0_cov)  # inverse covariance
        self.g0_sidc = 1 / np.sqrt(np.linalg.det(g0_cov))  # sqrt of inverse of covariance determinant
        self.g2_ic = np.linalg.inv(g2_cov)
        self.g2_sidc = 1 / np.sqrt(np.linalg.det(g2_cov))

    def trunc(self, his):
        """
        Since very early history's influence is negligible,
        Only calculate the last MAX_HISTORY influences to speed-up
        """
        if len(his) > self.max_history:
            return his[-self.max_history:]
        else:
            return his

    def lamb_t(self, t, comp=False):
        """
        λ(t|H) = μ + ∑g1(δt)
        """
        if comp:
            temp = np.append(self.mu, self.g1(t, self.trunc(self.his_t[self.his_t < t]), self.alpha, self.beta))
            return temp / np.sum(temp)
        else:
            return self.mu + np.sum(self.g1(t, self.trunc(self.his_t[self.his_t < t]), self.alpha, self.beta))

    def lamb_st(self, s, t):
        """
        λ(s,t|H) = μ g0(s) + ∑g1(δt)g2(δs)
        
        :param s: shape [2], or shape [N, 2]
        :param t: float, or shape [N]
        :return: shape [1], or shape [N]
        """
        mask = 1.
        if type(t) == np.ndarray and np.prod(t.shape) != 1:
            idx = np.searchsorted(self.his_t, t)  # Number of events before each t
            his_t_ = []
            his_s_ = []
            seq_lens = []
            for i in idx:
                idx_start = i - self.max_history if i > self.max_history else 0
                idx_end = i
                seq_len = idx_end - idx_start
                his_t_.append(self.his_t[idx_start:idx_end])
                his_s_.append(self.his_s[idx_start:idx_end])
                seq_lens.append(seq_len)
            his_t = np.zeros((len(his_t_), max(seq_lens)))  # Create padded array of history
            his_s = np.zeros((len(his_s_), max(seq_lens), 2))
            mask = np.zeros_like(his_t).astype(bool)
            for i, his in enumerate(his_t_):
                his_t[i, :len(his)] = his
                his_s[i, :len(his)] = his_s_[i]
                mask[i, :len(his)] = True
        else:
            his_t = self.trunc(self.his_t[self.his_t < t])
            his_s = self.trunc(self.his_s[self.his_t < t])
        
        return self.mu * self.g0(s, self.s_mu, self.g0_sidc, self.g0_ic) + \
               np.sum(self.g1(t, his_t, self.alpha, self.beta) *
                      self.g2(s, his_s, self.g2_sidc, self.g2_ic) * mask, -1)

    def predict_next(self, i):
        """
        Predict time and location of the (i+1)th event

        :param i: consider the ith event as the last event
        """
        ti = self.his_t[i]
        c = self.alpha * np.sum(np.exp(-self.beta * (ti - self.trunc(self.his_t[:(i + 1)]))))

        int_lamb = lambda t: np.exp(c / self.beta * (np.exp(-self.beta * (t - ti)) - 1) - 
                                    self.mu * (t - ti))

        time_pdf = lambda t: t * (self.mu + c * np.exp(-self.beta * (t - ti))) * int_lamb(t)

        space_pdf = lambda t: (self.mu * self.s_mu + self.alpha * 
                               np.sum(np.exp(-self.beta * (t - self.trunc(self.his_t[:(i + 1)])))[:, np.newaxis] * 
                                      self.trunc(self.his_s[:(i + 1)]), axis=0)) * int_lamb(t)

        return quad_vec(space_pdf, ti, np.inf)[0], quad(time_pdf, ti, np.inf)[0]

    def generate_offsprings(self, t_i, s_i, verbose=False):
        """
        Generate the offsprings for event_i through Ogata thinning

        :param t_i: from non-homogeneous Poisson process with λ = α * exp(-β(t-t_i))
        :param s_i: from normal distribution centered at s_i
        """
        t = t_i
        count = 0
        while True:
            # Calculate the max intensity
            m = self.alpha * np.exp(-self.beta * (t - t_i))
            t += np.random.exponential(scale=1 / m)
            if t > self.t_end:
                break
            # Calculate the new intensity
            lamb = self.alpha * np.exp(-self.beta * (t - t_i))

            if lamb / m >= np.random.uniform():  # Accept the sample
                s = np.random.multivariate_normal(s_i.squeeze(), self.g2_cov)
                s = np.expand_dims(s.astype("float64"), 0)
                count += 1
                # Insert at the correct place
                n = len(self.his_t[self.his_t < t])
                self.his_s = np.insert(self.his_s, n, s, axis=0)
                self.his_t = np.insert(self.his_t, n, t)
        if verbose:
            print(f"{count} offsprings generated for event at {t_i}")

    def generate(self, t_start, t_end, verbose=False):
        """
        Generate events in [0, T] with the cluster structure of the self-exciting process
        """
        # First generate 0-generation events from the background process using μ
        self.t_start = t_start
        self.t_end = t_end
        t = t_start
        self.his_s = np.zeros((0, 2))
        self.his_t = np.array([])

        count = 0
        while True:
            count += 1
            t += np.random.exponential(scale=1 / self.mu)
            if t > t_end:
                break
            s = np.random.multivariate_normal(self.s_mu, self.g0_cov)
            s = np.expand_dims(s.astype("float64"), 0)
            self.his_s = np.vstack((self.his_s, s))
            self.his_t = np.append(self.his_t, t)

        if verbose:
            print(f"{count} 0-generation events generated")

        # Generate next generations events
        t = t_start
        n = 0
        while True:
            self.generate_offsprings(self.his_t[n], self.his_s[n], verbose)
            try:
                # Find next event index after t
                n = next(x[0] for x in enumerate(self.his_t) if x[1] > t)
                t = self.his_t[n]
            except StopIteration:
                # When there is no such event, terminate
                break

    def nll(self, alpha, beta, mu, g0_cov, g2_cov):
        """
        Calculate the negative log likelihood

        numerical: boolean, whether evaluate the second term using numerical integration
        """
        # Pre-calculate Σ^(-1) and |Σ|^(-0.5)
        g0_ic = np.linalg.inv(g0_cov)
        g0_sidc = 1 / np.sqrt(np.linalg.det(g0_cov))
        g2_ic = np.linalg.inv(g2_cov)
        g2_sidc = 1 / np.sqrt(np.linalg.det(g2_cov))

        # It is intuitive that optimal s_mu is the mean of all events...
        s_mu = np.mean(self.his_s, axis=0)

        # - ∑ log λ(s_i, t_i)
        term_1 = 0
        for i in range(1, len(self.his_s)):
            lamb = np.sum(self.g1(self.his_t[i], self.trunc(self.his_t[:i]), alpha, beta) *
                          self.g2(self.his_s[i], self.trunc(self.his_s[:i]), g2_sidc, g2_ic))
            lamb += mu * self.g0(self.his_s[i], s_mu, g0_sidc, g0_ic)
            term_1 -= np.log(lamb)

        # + ∫∫ λ(s,t) dsdt = μT - α/β ∑ [exp(-β(T-t_i)) - 1]
        term_2 = mu * (self.t_end - self.t_start)
        term_2 -= alpha / beta * np.sum((np.exp(-beta * (self.t_end - self.his_t)) - 1))

        return term_1 + term_2

    def plot_intensity(self, s=None, t_start=None, t_end=None, color='blue'):
        """
        Plot λst vs. time at a specified location
        """
        if s is None:
            s = self.s_mu[np.newaxis, :]
        if t_start is None:
            t_start = self.t_start
        if t_end is None:
            t_end = self.t_end

        width, _ = plt.figaspect(.1)
        _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(width, width / 2))

        # Plot the intensity
        x = np.arange(1001) / 1000
        x = eps + t_start + x * (t_end - t_start)
        y = [self.lamb_st(s, t) for t in x]
        ax1.plot(x, y, color, label=f'Intensity at ({s[0, 0]}, {s[0, 1]})')
        ax1.set_xlim([t_start, t_end])
        ax1.legend()

        # Plot the events
        idx = np.logical_and(self.his_t >= t_start, self.his_t < t_end)
        ax2.stem(self.his_t[idx], np.sqrt(np.sum(np.square(self.his_s[idx] - s), axis=1)),
                 use_line_collection=True, label=f'Events(height = dist to ({s[0, 0]}, {s[0, 1]}))')
        ax2.set_xlim([t_start, t_end])
        ax2.invert_yaxis()
        ax2.legend()
        
    def get_lamb_st(self, x_num, y_num, t_num, t_start, t_end, 
                    x_min=None, x_max=None, y_min=None, y_max=None):
        """
        return STHP lamb_st for a given time range
        
        :param x_num: int, resolution of x
        :param y_num: int, resolution of y
        :param t_num: int, resolution of t
        :return lambs: List of len (t_num+1), element np array [x_num+1, y_num+1]
        :return x_range: np array [x_num+1]
        :return y_range: np array [y_num+1]
        :return t_range: np array [t_num+1]
        """
        idx = np.logical_and(self.his_t >= t_start, self.his_t < t_end)

        his_s = self.his_s[idx]

        if x_max is None or y_max is None:
            x_max, y_max = np.max(his_s, 0)
        if x_min is None or y_min is None:
            x_min, y_min = np.min(his_s, 0)

        x_range = arange(x_num - 1, [[x_min, x_max]]).squeeze(-1)
        y_range = arange(y_num - 1, [[y_min, y_max]]).squeeze(-1)
        xy_range = arange([x_num - 1, y_num - 1], [[x_min, x_max], [y_min, y_max]])
        t_range = np.linspace(t_start, t_end, t_num)
        
        lambs = []
        for t in tqdm(t_range):
            lamb_st = self.lamb_st(xy_range, t).reshape(x_num, y_num, order='F')
            lambs.append(lamb_st)
        return lambs, x_range, y_range, t_range


class DEBMDataset(SyntheticDataset):
    """
    Simulate Discrete-Event Brownian Motion
    """
    def __init__(self, s_mu, g2_cov, alpha, beta, mu, max_history=100, dist_only=False):
        """
        s_mu: shape (2,)
        g2_cov: shape (2, 2)
        """
        SyntheticDataset.__init__(self, dist_only)
        self.s_mu = s_mu
        self.g2_cov = g2_cov
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.max_history = max_history

        self.g2_ic = np.linalg.inv(g2_cov)
        self.g2_sidc = 1 / np.sqrt(np.linalg.det(g2_cov))

    def lamb_t(self, t):
        """
        λ(t|H) = μ + β(t-t_j)
        """
        return self.mu + self.alpha * np.exp(-self.beta * (t - self.his_t[self.his_t < t][-1]))

    def lamb_st(self, s, t):
        """
        λ(s,t|H) = g0(s) μ exp(βt - α ∑g2(δs))
        """
        return self.g2(self.his_s[self.his_t < t][-1:], s, self.g2_sidc, self.g2_ic) * self.lamb_t(t)

    def generate(self, t_start, t_end, verbose=False):
        """
        Generate long event sequence
        """
        t = 0
        self.his_s = self.s_mu.reshape(1, 2)
        self.his_t = np.array([-eps, ])

        while True:
            # Calculate the max intensity
            lamb_t = self.lamb_t(t)
            if self.beta >= 0:  # Decreasing or stable intensity
                l = np.inf
                m = self.lamb_t(t + eps)
            else:  # Increasing intensity
                l = 2 / lamb_t
                m = self.lamb_t(t + l)
            delta_t = np.random.exponential(scale=1 / m)

            if t + delta_t > t_end:
                break
            if delta_t > l:
                t += l
                continue
            else:
                t += delta_t
                new_lamb_t = self.lamb_t(t)
                if new_lamb_t / m >= np.random.uniform():  # Accept the sample
                    if verbose:
                        print("----")
                        print(f"t:  {t}")
                        print(f"λt: {new_lamb_t}")

                    # Sample a new location based on last location
                    s = np.random.multivariate_normal(self.his_s[-1], self.g2_cov)
                    s = np.expand_dims(s.astype("float64"), 0)
                    self.his_s = np.vstack((self.his_s, s))
                    self.his_t = np.append(self.his_t, t)


def main():
    data = [torch.zeros(30, 3), torch.zeros(40, 3), torch.zeros(31, 3), torch.zeros(32, 3)]
    dataset = ListDataset(data)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=spatiotemporal_events_collate_fn)
    padded, mask = next(iter(dataloader))
    print(padded.shape, mask.shape)
    print(padded, mask)


if __name__ == '__main__':
    main()
