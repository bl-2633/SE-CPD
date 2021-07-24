import torch
from se3_transformer_pytorch import SE3Transformer
from torch import nn
class SE3Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_dim = 64
        self.SE3_encoder  = SE3Transformer(
            dim = self.feat_dim,
            heads = 2,
            depth = 1,
            dim_head = 64,
            num_degrees = 1,
            edge_dim = 1
        )

        self.feat_enc = nn.Sequential(
            nn.Linear(6, self.feat_dim),
            nn.ReLU(inplace = True)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 20),
            nn.Softmax(dim = 2)
        )

    def forward(self,feats, coors, mask, edges):
        feats = torch.cat([torch.sin(feats), torch.cos(feats)], axis = -1)
        feats = self.feat_enc(feats)
        out = self.SE3_encoder(feats, coors, mask, edges = edges)
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


    