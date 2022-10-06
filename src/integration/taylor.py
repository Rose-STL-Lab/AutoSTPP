import torch
from torch import nn
from torch.autograd import grad


beta = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1], dtype=torch.float64) # params


def nth_derivatives(f, wrt, n):
    grads = []
    for i in range(n):
        g = grad(f.sum(), wrt, create_graph=True)[0] # Compute gradient elementwise
        f = g.sum()
        grads.append(g)
    return torch.stack(grads, 0)


def y(f, ds):
    return torch.stack([beta[0]*ds[0], 
        beta[1]*ds[0] + beta[0]**2*ds[1]/2.0,
        beta[2]*ds[0] + beta[0]*beta[1]*ds[1] + beta[0]**3*ds[2]/6.0,
        beta[3]*ds[0] + (beta[0]*beta[2] + beta[1]**2/2.0)*ds[1] \
            + beta[0]**2*beta[1]*ds[2]/2.0 + beta[0]**4*ds[3]/24.0,
        beta[4]*ds[0] + (beta[0]*beta[3] + beta[1]*beta[2])*ds[1] \
            + (beta[0]**2*beta[2] + beta[0]*beta[1]**2)*ds[2]/2.0 \
            + beta[0]**3*beta[1]*ds[3]/6.0 + beta[0]**5*ds[4]/120.0]).squeeze().float()


def A(c, a, b):
    A5 = ((b-c)**6 - (a-c)**6) / (beta[0]**5*6)
    A4 = ((b-c)**5 - (a-c)**5) / (beta[0]**4*5) \
        - 4*beta[1]*A5 / beta[0]
    A3 = ((b-c)**4 - (a-c)**4) / (beta[0]**3*4) \
        - 3*beta[1]*A4 / beta[0] \
        - 3 * (beta[2]/beta[0] + beta[1]**2 / beta[0]**2) * A5
    A2 = ((b-c)**3 - (a-c)**3) / (beta[0]**2*3) \
        - 2*beta[1]*A3 / beta[0] \
        - (2*beta[2]/beta[0] + beta[1]**2 / beta[0]**2) * A4 \
        - 2*(beta[3]/beta[0] + beta[1]*beta[2]/beta[0]**2) * A5
    A1 = ((b-c)**2 - (a-c)**2) / (beta[0]*2) - beta[1]*A2 / beta[0] \
        - beta[2]*A3 / beta[0] - beta[3]*A4 / beta[0] \
        - beta[4]*A5 / beta[0]
    return torch.stack([A1, A2, A3, A4, A5]).squeeze().float()


def int_over_interval(f, c, a, b):
    Y_val = y(f, nth_derivatives(f, c, n=5))
    A_val = A(c, a, b)
    try:
        prod = torch.einsum('ji,ji->i', Y_val, A_val)
    except:
        prod = Y_val @ A_val
    return prod + f * (b-a)


"""
Taylor Integration
"""
class TaylorInt(nn.Module):
    
    # Compute ∫ MLP(x) from 0 to x
    def __init__(self, x_min, x_max, width, device):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.device = device
        self.step(width)

    # Reset integration step width
    def step(self, new_width):
        self.width = new_width
        self.intervals = torch.arange(self.x_min, self.x_max + self.width / 2.0, 
                                      self.width, dtype=torch.float32).to(self.device)
        self.center = (self.intervals[1:] + self.intervals[:-1]) / 2.0
        self.center.requires_grad = True
        
    '''Forward pass'''
    def forward(self, model, x, discrete=False):
        # Calculate integral over all subintervals
        f = model(self.center)
        int_vals_s = int_over_interval(f, self.center, self.intervals[:-1], self.intervals[1:])
        int_vals_s = torch.cat([torch.zeros(1).to(self.device), torch.cumsum(int_vals_s, 0)])
        a = x - x % self.width
        idx = (a / self.width).long()
        
        if discrete:
            int_vals_s = (int_vals_s[:-1] + int_vals_s[1:]) / 2.0
            return int_vals_s[idx]
        
        # Calculate interval for x in their subintervals
        c = (a + x) / 2.0
        c.requires_grad = True
        fc = model(c)
        int_vals_x = int_over_interval(fc, c, a, x)
        
        # Summing [0, a] and [a, x]
        idx = (a / self.width).long()
        return int_vals_s[idx] + int_vals_x