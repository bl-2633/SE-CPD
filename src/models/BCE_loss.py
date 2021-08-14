import torch
import torch.nn.functional as F

class BCE_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.EPS = 1e-5

    def forward(self, out, target, mask):
        assert out.size() == target.size(), "pred and target should have the same size"
        out = F.softmax(out, dim = 2)

        pred = torch.clamp(out, self.EPS, 1-self.EPS)
        gt = target
        loss = torch.multiply(gt,torch.log(pred)) + torch.multiply((1 - gt), torch.log(1 - pred))
        loss = loss.sum(-1)
        loss = torch.divide((loss * mask).sum(-1), mask.sum(-1)).mean()
        
        
        return -loss
