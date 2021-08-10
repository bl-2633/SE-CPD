import torch

class NLL_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, target, alpha = 0.1):
        assert out.size() == target.size(), "pred and target should have the same size"
        pred = out.squeeze(0)
        gt = target.squeeze(0)
        #gt = (1 - alpha) * gt + alpha / float(gt.size(1))
        loss = torch.sum(torch.multiply(gt, pred)) / pred.size(0)
        return -loss