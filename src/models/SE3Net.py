import torch
from se3_transformer_pytorch import SE3Transformer
from torch import nn
class SE3Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.SE3_encoder  = SE3Transformer(
            dim = 32,
            heads = 1,
            depth = 1,
            dim_head = 32,
            num_degrees = 2,
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 20),
            nn.ReLU(inplace = True)
        )

    def forward(self,feats, coors, mask):
        out = self.SE3_encoder(feats, coors, mask)
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = SE3Net().to(device)
    with torch.cuda.amp.autocast():
        for i in range(5):
            feats = torch.randn(1, 500, 32, dtype = torch.half).to(device)
            coors = torch.randn(1, 500, 3, dtype = torch.half).to(device)
            mask  = torch.ones(1, 500, dtype = torch.half).bool().to(device)
            out = model(feats, coors, mask)
            print(out.size())
        max_mem = torch.cuda.max_memory_allocated(device)
        max_cache = torch.cuda.max_memory_cached(device)
        print(bytesto(max_cache, 'g', bsize=1024))


    