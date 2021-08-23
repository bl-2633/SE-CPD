import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
import torch
from se3_transformer_pytorch import SE3Transformer
from en_transformer import EnTransformer
from torch import nn
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append(dir_path)
import PE_module, RBF_func
import utils

class EnNet(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.feat_dim = 128
        self.device = device

        #self.En_encoder = SE3Transformer(
        #    dim = self.feat_dim,
        #    depth = 2,
        #    heads = 4,
        #    num_degrees = 1,
        #    edge_dim = 17,
        #)


        #self.En_decoder = SE3Transformer(
        #    dim = self.feat_dim,
        #    depth = 2,
        #    heads = 4,
        #    num_degrees = 1,
        #    edge_dim = 17,
        #)

        self.En_encoder = EnTransformer(
            dim = self.feat_dim,
            depth = 6,
            heads = 8,
            edge_dim = 16,
            dim_head = self.feat_dim,
            neighbors = 30
        )


        self.En_decoder = EnTransformer(
            dim = self.feat_dim,
            depth = 6,
            heads = 8,
            edge_dim = 16,
            dim_head = self.feat_dim,
            neighbors = 30
        )
        self.RBF = RBF_func.RBF(Device = device)
        self.decoder = nn.TransformerDecoderLayer(self.feat_dim, 4)
        self.PE = PE_module.PositionalEncoding(d_model = self.feat_dim)

        # Input node feature encoder
        self.feat_enc = nn.Sequential(
            nn.Linear(6, self.feat_dim, bias = True),
            nn.ReLU(inplace = True)
        )
        self.tgt_embed = nn.Sequential(
            nn.Linear(20, self.feat_dim, bias = True),
            nn.LeakyReLU(inplace = True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 20, bias = True),
        )


    def forward(self, feats, coors, edges, mask, seq_shifted):
        seq_len = seq_shifted.size(1)
        # encoder 
        in_feats =torch.cat([torch.sin(feats), torch.cos(feats)], axis = -1)
        in_feats = self.feat_enc(in_feats)
        in_feats = self.PE(in_feats)
        edges = self.RBF(edges)
        enc_out, enc_coors = self.En_encoder(in_feats, coors, edges, mask)

        # decoder
        # masked attention encoder for teacher forcing
        #ar_mask = utils.generate_square_subsequent_mask(seq_len).to(self.device)
        #tgt_embed = self.PE(self.tgt_embed(seq_shifted)).permute(1,0,2)
        #decoder_out = self.decoder(tgt = tgt_embed, memory = enc_out, tgt_mask = ar_mask, 
        #                            tgt_key_padding_mask = ~mask,
        #                            memory_key_padding_mask = ~mask).permute(1,0,2)
        
        decoder_out = self.En_decoder(enc_out, enc_coors, edges, mask)[0]
        logits = self.classifier(decoder_out)
        
        return logits
        


    def sample(self, feats, coors, edges, mask):
        # encoder 
        seq_len = feats.size(1)
        in_feats =torch.cat([torch.sin(feats), torch.cos(feats)], axis = -1)
        in_feats = self.feat_enc(in_feats)
        enc_feat, enc_coors = self.SEn_encoder(in_feats, coors, edges, mask)
        
        seq_out = torch.zeros(1, seq_len, 20).to(self.device)
        prob_out = torch.zeros(1, seq_len, 20).to(self.device)
        for t in range(seq_len):
            #decoder
            decoder_feat = self.seq_enc(seq_out) + enc_feat
            dec_feat = self.SEn_decoder(decoder_feat, enc_coors, edges, mask)[0][0,t,:]
            logits = self.classifier(dec_feat)
            
            #sampling step
            aa_t = torch.zeros(1,20)
            prob = F.softmax(logits, dim = 0) 
            aa_idx = torch.torch.multinomial(prob, 1).item()
            aa_t[0,aa_idx] = 1
            seq_out[0,t,:] = aa_t
            prob_out[0,t,:] = prob
        return seq_out, prob_out
    
    

if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = EnNet(device = device).to(device).eval()
    with torch.cuda.amp.autocast():
        for i in range(10):
            feats = torch.randn(1, 500, 3).to(device)
            coors = torch.randn(1, 500, 3).to(device)
            edges = torch.randn(1, 500, 500, 1).to(device)
            mask = torch.ones(1, 500).bool().to(device)
            seq = torch.randn(1, 500, 20).to(device)
            #out_feats, out_coors = model(feats, coors, edges, mask,seq)
            prob_out = model(feats, coors, edges, mask, seq)
            print(prob_out.size())
            #print(out_feats.size())