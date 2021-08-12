import torch
import torch.nn.functional as F


class NLL_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.EPS = 1e-9

    def forward(self, out, target, alpha = 0.1):
        assert out.size() == target.size(), "pred and target should have the same size"
        out = torch.clamp(out, self.EPS, 1-self.EPS)
        out = F.log_softmax(out, dim = 2)
        pred = out.squeeze(0)
        gt = target.squeeze(0)
        gt = (1 - alpha) * gt + alpha / float(gt.size(1))
        loss = torch.sum(torch.multiply(gt, pred)) / pred.size(0)
        return -loss