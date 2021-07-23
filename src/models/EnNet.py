import torch
from en_transformer import EnTransformer
from torch import nn
import numpy as np


class EnNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.feat_dim = 64

        self.SEn_encoder_1 = EnTransformer(
            dim = self.feat_dim,
            depth = 8,
            dim_head = 128,
            coors_hidden_dim = 64,
            heads = 8,
            edge_dim = 1,
            neighbors = 30,
        )

        self.feat_enc = nn.Sequential(
            nn.Linear(6, self.feat_dim),
            nn.ReLU(inplace = True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 20),
            nn.Softmax(dim = 2)
        )

    def forward(self, feats, coors, edges, mask):
        feats =torch.cat([torch.sin(feats), torch.cos(feats)], axis = -1)
        feats = self.feat_enc(feats)
        out_feat = self.SEn_encoder_1(feats, coors, edges, mask)[0]
        out = self.classifier(out_feat)
        return out

if __name__ == '__main__':
    device = torch.device('cuda:1')
    model = EnNet().to(device)
    with torch.cuda.amp.autocast():
        for i in range(100):
            feats = torch.randn(1, 500, 32).to(device)
            coors = torch.randn(1, 500, 3).to(device)
            edges = torch.randn(1, 500, 500, 1).to(device)
            mask = torch.ones(1, 500).bool().to(device)
            out_feats, out_coors = model(feats, coors, edges, mask)
            print(out_feats.size())