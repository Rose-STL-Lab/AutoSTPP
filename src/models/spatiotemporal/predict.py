from models.spatiotemporal.model import AutoIntSTPPSameInfluence
import torch


def get_predict(model):
    
    device = next(model.parameters()).device
    
    if type(model) == AutoIntSTPPSameInfluence:
        """
        Compute model intensities at different time

        :param his_st: np array [seqlen, 3], the event spatiotemporal history
        :param x: np array [N, 3], a batch of space and time
        :return:  np array [N, 1], the intensities at different space and time
        """
        def predict(model, his_st, x):
            def lamb_func(t, his_st):
                delta_t = t - his_st[his_st < t]
                if len(delta_t) == 0:
                    return model.background.item()
                lamb_t = model.F.dnforward(torch.tensor(delta_t).float().unsqueeze(-1).to(device), 
                                           [0]).sum() + model.background
                return lamb_t.item()
            
            y = np.array([lamb_func(t, his_st) for t in x])
            return y
        