import torch
import torch.nn.functional as F


class NLL_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.EPS = 1e-9

    def forward(self, out, target, mask, alpha = 0.1):
        assert out.size() == target.size(), "pred and target should have the same size"
        out = torch.clamp(out, self.EPS, 1-self.EPS)
        out = F.log_softmax(out, dim = 2)
        #target = (1 - alpha) * target + alpha / float(target.size(1))
        
        loss = (target * out).sum(-1)
        loss = torch.divide((loss * mask).sum(-1), mask.sum(-1)).mean()

        return -loss