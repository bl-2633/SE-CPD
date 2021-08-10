import torch
from se3_transformer_pytorch import SE3Transformer
from en_transformer import EnTransformer
from torch import nn
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append('/mnt/local/blai/SE-CPD/src/models/')
import PE_module

class EnNet(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.encoder_hidden_dim = self.model_config['encoder_hidden_dim']
        self.decoder_type = self.model_config['decoder_type']

        assert self.decoder_type in ['LSTM', 'EGNN'], 'please provide a valid decoder' 

        self.SEn_encoder = SE3Transformer(
            dim = self.feat_dim,
            depth = 4,
            dim_head = self.feat_dim,
            heads = 8,
            num_degrees = 1,
            edge_dim = 17,
            egnn_hidden_dim = self.feat_dim,
            use_egnn = True,
        )
        
        if self.decoder_type == 'EGNN':        
            self.decoder = SE3Transformer(
                dim = self.feat_dim,
                depth = 4,
                dim_head = self.feat_dim,
                heads = 8,
                num_degrees = 1,
                edge_dim = 17,
                egnn_hidden_dim = self.feat_dim,
                use_egnn = True,
            )
        elif self.decoder_type == 'LSTM':
            self.decoder =  nn.LSTM(input_size = self.feat_dim, hidden_size = 512, num_layers = 3,
                                        batch_first = True, dropout = 0.2, bidirectional = True)
            
            self.LSTM_feat_enc =nn.Sequential(
            nn.Linear(1024, self.feat_dim),
            nn.ReLU(inplace = True)
            )
        
        
        self.PE = PE_module.PositionalEncoding(d_model = self.feat_dim)

        # Input node feature encoder
        self.feat_enc = nn.Sequential(
            nn.Linear(6, self.feat_dim),
            nn.ReLU(inplace = True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 20),
        )

    def forward(self, feats, coors, edges, mask):

        # encoder 
        in_feats =torch.cat([torch.sin(feats), torch.cos(feats)], axis = -1)
        in_feats = self.feat_enc(in_feats)
        in_feats = self.PE(in_feats)
        enc_out= self.SEn_encoder(in_feats, coors, mask, edges = edges)['0']
        
        #decoder
        decoder_feat = enc_out
        #dec_feat = self.SEn_decoder(decoder_feat, coors, mask, edges = edges)['0']
        dec_feat = self.dec_feat_enc(self.LSTM_decoder(decoder_feat)[0])
        logits = self.classifier(decoder_feat)
        log_prob = F.log_softmax(logits, dim = 2)
        # training does not require sampling sequences
        ''''
        for t in range(seq_len):
            aa_t = torch.zeros(1,20)
            aa_idx = torch.torch.multinomial(F.softmax(logits[0,t,:], dim = 0), 1).item()
            aa_t[0,aa_idx] = 1
            seq_out[0,t,:] = aa_t
        '''
        return log_prob
        


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
            prob_out = model(feats, coors, edges, mask)
            print(prob_out.size())
            exit()
            #print(out_feats.size())