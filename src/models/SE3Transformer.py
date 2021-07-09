import torch
from se3_transformer_pytorch import SE3Transformer

class SE3Tfmer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.SE3_encoder  = SE3Transformer(
            dim = 32,
            heads = 2,
            depth = 2,
            dim_head = 28,
            num_degrees = 1,
        )

    def forward(self,feats, coors, mask):
        out = self.SE3_encoder(feats, coors, mask)
        return out


if __name__ == '__main__':
    def bytesto(bytes, to, bsize=1024): 
        a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
        r = float(bytes)
        return bytes / (bsize ** a[to])
    
    device = torch.device('cuda:0')
    model = SE3Tfmer().to(device)
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


    