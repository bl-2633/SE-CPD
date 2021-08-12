import torch
import torch.nn.functional as F

class BCE_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.EPS = 1e-9

    def forward(self, out, target):
        assert out.size() == target.size(), "pred and target should have the same size"
        out = F.softmax(out, dim = 2)
        out = out.squeeze(0)
        pred = torch.clamp(out, self.EPS, 1-self.EPS)
        gt = target.squeeze(0)
        loss = torch.multiply(gt,torch.log(pred)) + torch.multiply((1 - gt), torch.log(1 - pred))
        loss = -1 * loss.mean()
        return loss
