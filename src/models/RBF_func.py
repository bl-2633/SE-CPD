import torch
import torch.nn.functional as F

class RBF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_rbf = 16
        self.D_min = 0
        self.D_max = 20

    def forward(self, D):
        D_mu = torch.linspace(self.D_min, self.D_max, self.num_rbf).to(D.get_device())
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (self.D_max - self.D_min) / self.num_rbf
        D_rbf = torch.exp(-((D - D_mu) / D_sigma)**2)
        return D_rbf