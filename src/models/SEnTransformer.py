import torch
from en_transformer import EnTransformer

class SEnTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.SEn_encoder = EnTransformer(
            dim = 32,
            depth = 4,
            dim_head = 64,
            heads = 8,
            edge_dim = 1,
            neighbors = 30,
            use_cross_product = True
        )

    def forward(self, feats, coors, edges, mask):
        out = self.SEn_encoder(feats, coors, edges, mask)
        return out

if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = SEnTransformer().to(device)
    with torch.cuda.amp.autocast():
        for i in range(100):
            feats = torch.randn(1, 500, 32).to(device)
            coors = torch.randn(1, 500, 3).to(device)
            edges = torch.randn(1, 500, 500, 1).to(device)
            mask = torch.ones(1, 500).bool().to(device)
            out_feats, out_coors = model(feats, coors, edges, mask)
            print(out_feats.size())