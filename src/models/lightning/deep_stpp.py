import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

from tqdm import tqdm
from models.lightning.stpp import BaseSTPointProcess

from typing import Dict, Union
from utils import load_class


def subsequent_mask(sz):
    """
    Return a square attention mask to only allow self-attention layers to attend the earlier positions
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout, max_len):
        """
        Injects some information about the relative or absolute position of the tokens in the sequence
        ref: https://github.com/harvardnlp/annotated-transformer/
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device("cpu")

    def forward(self, x, t):
        pe = torch.zeros(self.max_len, self.d_model).to(self.device)
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() *
                             (-math.log(10000.0) / self.d_model)).to(self.device)
        
        t = t.unsqueeze(-1)
        pe = torch.zeros(*t.shape[:2], self.d_model).to(self.device)
        pe[..., 0::2] = torch.sin(t * div_term)
        pe[..., 1::2] = torch.cos(t * div_term)
        
        x = x + pe[:x.size(0)]
        return self.dropout(x)
    

class TransformerEncoder(pl.LightningModule):    

    def __init__(
        self,
        emb_dim: int = 128,
        z_dim: int = 128,
        dropout: float = 0,
        num_head: int = 2,
        enc_n_layers: int = 3,
        seq_len: int = 20,
        enc_hid_dim: int = 128,
        **kwargs
    ):
        """Transformer encoder module
        Encode time/space record to variational posterior for location latent.

        Parameters
        ----------
        emb_dim : int, optional
            The dimension of the input embeddings
        z_dim : int, optional
            The dimension of the latent representation
        dropout : float, optional
            The dropout rate for the encoder
        num_head : int, optional
            The number of attention heads to use
        enc_n_layers : int, optional
            The number of encoder layers
        seq_len : int, optional
            The length of the input sequence
        enc_hid_dim : int, optional
            The number of hidden units in the feedforward layers
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(self.hparams.emb_dim, self.hparams.dropout, 
                                              self.hparams.seq_len)
        encoder_layers = nn.TransformerEncoderLayer(self.hparams.emb_dim, self.hparams.num_head,
                                                    self.hparams.enc_hid_dim, self.hparams.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.hparams.enc_n_layers)
        self.seq_len = self.hparams.seq_len
        self.ninp = self.hparams.emb_dim
        self.encoder = nn.Linear(3, self.hparams.emb_dim, bias=False)
        self.decoder = nn.Linear(self.hparams.emb_dim, self.hparams.z_dim * 2)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def encode(self, x, x_mask=None):
        x = x.transpose(1, 0)  # Convert to seq-len-first
        if x_mask is None:
            x_mask = subsequent_mask(len(x)).to(self.device)
        t = torch.cumsum(x[..., -1], 0)
        x = self.encoder(x) * math.sqrt(self.ninp)
        x = self.pos_encoder(x, t)
        
        output = self.transformer_encoder(x, x_mask)
        output = self.decoder(output)
        
        output = output[-1]  # get last output only
        m, v_ = torch.split(output, output.size(-1) // 2, dim=-1)
        v = F.softplus(v_) + 1e-5
        return m, v


class MLPDecoder(pl.LightningModule):
    
    def __init__(
        self, 
        out_dim: int, 
        z_dim: int = 128, 
        dec_hid_dim: int = 128, 
        dec_n_layers: int = 3, 
        softplus: bool = False, 
        **kwargs
    ):
        """
        An MLP decoder that takes a latent representation 
        and decodes it into spatiotemporal kernel coefficients

        Parameters
        ----------
        out_dim : int
            The dimension of the output sequence.
        z_dim : int
            The dimension of the input latent representation.
        dec_hid_dim : int
            The number of hidden units in each fully-connected layer.
        dec_n_layers : int
            The number of fully-connected layers in the decoder.
        softplus : bool, optional
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.net = nn.Sequential(
            nn.Linear(self.hparams.z_dim, self.hparams.dec_hid_dim),
            nn.ELU(),
            *[nn.Linear(self.hparams.dec_hid_dim, self.hparams.dec_hid_dim),
              nn.ELU()] * (self.hparams.dec_n_layers - 1),
            nn.Linear(self.hparams.dec_hid_dim, self.hparams.out_dim),
        )

    def decode(self, z):
        output = self.net(z)
        if self.hparams.softplus:
            output = F.softplus(output) + 1e-5
        return output
    
    
def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension

    :param qm: tensor: (batch, dim): q mean
    :param qv: tensor: (batch, dim): q variance
    :param pm: tensor: (batch, dim): p mean
    :param pv: tensor: (batch, dim): p variance
    :return kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl


def sample_gaussian(m, v):
    """
    Element-wise application reparameterization trick to sample from Gaussian

    :param m: tensor: (batch, ...): Mean
    :param v: tensor: (batch, ...): Variance
    :return z: tensor: (batch, ...): Samples
    """
    z = torch.randn_like(m)
    z = z * torch.sqrt(v) + m
    return z


def ll_no_events(w_i, b_i, tn_ti, t_ti):
    """
    Log likelihood of no events happening from t_n to t
    - ∫_{t_n}^t λ(t') dt' 

    :param tn_ti: [batch, seq_len]
    :param t_ti: [batch, seq_len]
    :param w_i: [batch, seq_len]
    :param b_i: [batch, seq_len]

    :return: scalar
    """
    return torch.sum(w_i / b_i * (torch.exp(-b_i * t_ti) - torch.exp(-b_i * tn_ti)), -1)


def log_ft(t_ti, tn_ti, w_i, b_i):
    return ll_no_events(w_i, b_i, tn_ti, t_ti) + torch.log(t_intensity(w_i, b_i, t_ti))


def t_intensity(w_i, b_i, t_ti):
    """
    Compute spatial/temporal/spatiotemporal intensities

    :param tn_ti: [batch, seq_len]
    :param s_diff: [batch, seq_len, 2]
    :param inv_var: [batch, seq_len, 2]
    :param w_i: [batch, seq_len]
    :param b_i: [batch, seq_len]

    :return: λ(t) [batch] <br>
             f(s|t) [batch] <br>
             λ(s,t) [batch]
    """
    v_i = w_i * torch.exp(-b_i * t_ti)
    lamb_t = torch.sum(v_i, -1)
    return lamb_t


def s_intensity(w_i, b_i, t_ti, s_diff, inv_var):
    v_i = w_i * torch.exp(-b_i * t_ti)
    v_i = v_i / torch.sum(v_i, -1).unsqueeze(-1)  # Normalize
    g2 = torch.sum(s_diff * inv_var * s_diff, -1)
    g2 = torch.sqrt(torch.prod(inv_var, -1)) * torch.exp(-0.5 * g2) / (2 * np.pi)
    f_s_cond_t = torch.sum(g2 * v_i, -1)
    return f_s_cond_t


def intensity(w_i, b_i, t_ti, s_diff, inv_var):
    return t_intensity(w_i, b_i, t_ti) * s_intensity(w_i, b_i, t_ti, s_diff, inv_var)


class DeepSTPointProcess(BaseSTPointProcess):
    
    def __init__(
        self, 
        enc: Dict,
        w_dec: Dict,
        b_dec: Dict,
        s_dec: Dict,
        seq_len: int = 20,
        s_min: float = 1e-4,
        s_max: float = None,
        b_max: float = 20,
        lookahead: int = 1,
        beta: float = 1e-3,
        num_points: int = 20,
        clip: float = 1.0,
        constrain_b: Union[bool, str] = False,
        sample: bool = False,
        **kwargs  # for BaseSTPointProcess
    ):  
        """
        STPP model with VAE: directly modeling λ(s,t)
        """
        super(DeepSTPointProcess, self).__init__(**kwargs)
        out_dim = seq_len + num_points
        w_dec['init_args'].update({'out_dim': out_dim})
        b_dec['init_args'].update({'out_dim': out_dim})
        s_dec['init_args'].update({'out_dim': out_dim * 2})
        
        self.save_hyperparameters()
        
        # VAE for predicting spatial intensity
        self.enc = load_class(self.hparams.enc)
        self.w_dec = load_class(self.hparams.w_dec)
        self.b_dec = load_class(self.hparams.b_dec)
        self.s_dec = load_class(self.hparams.s_dec)
        
        # Set prior as fixed parameter attached to Module
        self.z_prior_m = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)
        
        # Background 
        self.num_points = self.hparams.num_points
        self.background = nn.Parameter(torch.rand((self.num_points, 2)), requires_grad=True)
        
    def project(self):
        pass

    def forward(self, st_x, st_y):
        """
        :param st_x: [batch, seq_len, 3] (lat, lon, time)
        :param st_y: [batch, 1, 3]
        """
        batch = st_x.shape[0]
        background = self.background.unsqueeze(0).repeat(batch, 1, 1)
        
        s_diff = st_y[..., :2] - torch.cat((st_x[..., :2], background), 1)  # s - s_i
        t_cum = torch.cumsum(st_x[..., 2], -1)
        
        tn_ti = t_cum[..., -1:] - t_cum  # t_n - t_i
        tn_ti = torch.cat((tn_ti, torch.zeros(batch, self.num_points).to(self.device)), -1)
        t_ti = tn_ti + st_y[..., 2]  # t - t_i

        [qm, qv], w_i, b_i, inv_var = self.encode(st_x)
            
        # Calculate likelihood
        sll = torch.log(s_intensity(w_i, b_i, t_ti, s_diff, inv_var))
        tll = log_ft(t_ti, tn_ti, w_i, b_i)
        
        # KL Divergence
        if self.hparams.sample:
            kl = kl_normal(qm, qv, *self.z_prior).mean()
            nelbo = kl - self.hparams.beta * (sll.mean() + tll.mean())
        else:
            nelbo = - (sll.mean() + tll.mean())

        return nelbo, sll.mean(), tll.mean()
   
    def encode(self, st_x):
        # Encode history locations and times
        if self.hparams.sample:
            qm, qv = self.enc.encode(st_x)  # Variational posterior
            # Monte Carlo
            z = sample_gaussian(qm, qv)
        else:
            qm, qv = None, None
            z, _ = self.enc.encode(st_x)
        
        w_i = self.w_dec.decode(z)
        if self.hparams.constrain_b == 'tanh':
            b_i = torch.tanh(self.b_dec.decode(z)) * self.hparams.b_max
        elif self.hparams.constrain_b == 'sigmoid':
            b_i = torch.sigmoid(self.b_dec.decode(z)) * self.hparams.b_max
        elif self.hparams.constrain_b == 'neg-sigmoid':
            b_i = - torch.sigmoid(self.b_dec.decode(z)) * self.hparams.b_max
        elif self.hparams.constrain_b == 'softplus':
            b_i = torch.nn.functional.softplus(self.b_dec.decode(z))
        elif self.hparams.constrain_b == 'clamp':
            b_i = torch.clamp(self.b_dec.decode(z), -self.hparams.b_max, self.hparams.b_max)
        else:
            b_i = self.b_dec.decode(z)
                    
        s_i = self.s_dec.decode(z) + self.hparams.s_min
        if self.hparams.s_max is not None:
            s_i = torch.sigmoid(s_i) * self.hparams.s_max
        
        s_x, s_y = torch.split(s_i, s_i.size(-1) // 2, dim=-1)
        inv_var = torch.stack((1 / s_x, 1 / s_y), -1)

        return [qm, qv], w_i, b_i, inv_var
    
    def on_before_optimizer_step(self, optimizer):
        """Clip gradients
        """
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams.clip)
    
    def calc_lamb(self, st_x, st_x_cum, st_y, st_y_cum, scales, biases,
                  x_range, y_range, t_range, device):
        s_grids = torch.stack(torch.meshgrid(x_range, y_range, indexing='ij'), dim=-1).view(-1, 2).to(device)
        st_x = st_x.to(device)
        st_x_cum = st_x_cum.to(device)
        x_range = x_range.to(device)
        y_range = y_range.to(device)
        t_range = torch.Tensor(t_range).to(device)

        # Calculate intensity
        background = self.background.unsqueeze(0)

        # Sample model parameters
        _, w_i, b_i, inv_var = self.encode(st_x.to(device))
        
        lambs = []
        for t in tqdm(t_range):
            i = sum(st_x_cum[:, -1, -1] <= t) - 1  # Index of corresponding history events

            if i >= 0:
                st_x_ = st_x[i:i + 1]
                w_i_ = w_i[i:i + 1]
                b_i_ = b_i[i:i + 1]
                inv_var_ = inv_var[i:i + 1]

                t_ = t - st_x_cum[i:i + 1, -1, -1]  # Time since lastest event
                t_ = (t_ - biases[-1]) / scales[-1]
            else:
                ## To accommadate time range before first event
                i_ = torch.searchsorted(st_x_cum[0, :, -1].contiguous(), t)
                st_x_ = torch.cat((torch.zeros_like(st_x[:1, i_:]), st_x[:1, :i_]), 1)
                _, w_i_, b_i_, inv_var_ = self.encode(st_x_.to(device))
                
                t_ = t - st_x_cum[0, i_ - 1, -1]  # Time since lastest event
                t_ = (t_ - biases[-1]) / scales[-1]

            # Calculate temporal intensity
            t_cum = torch.cumsum(st_x_[..., -1], -1)
            tn_ti = t_cum[..., -1:] - t_cum  # t_n - t_i
            tn_ti = torch.cat((tn_ti, torch.zeros(1, self.hparams.num_points).to(device)), -1).to(device)
            t_ti = tn_ti + t_

            lamb_t = t_intensity(w_i_, b_i_, t_ti) / np.prod(scales)

            # Calculate spatial intensity
            N = len(s_grids)  # Number of grid points

            s_x_ = torch.cat((st_x_[..., :-1], background), 1).repeat(N, 1, 1).to(device)
            s_diff = s_grids.unsqueeze(1) - s_x_
            lamb_s = s_intensity(w_i_.repeat(N, 1), b_i_.repeat(N, 1), t_ti.repeat(N, 1), 
                                 s_diff, inv_var_.repeat(N, 1, 1))

            lamb = (lamb_s * lamb_t).view(len(x_range), len(y_range))
            lambs.append(lamb.cpu().detach().numpy())
            
        return np.array(lambs)
