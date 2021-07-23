import EnNet, BCE_loss
import torch
import argparse
import data_loader
import sys
sys.path.append('../data_processing/')
import data_utils
import numpy as np
import tqdm
from torch.utils.data import DataLoader

if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = EnNet.EnNet().to(device)
    seq_len = 500
    feat_n = 64
    coors_n = 3
    
    loss_fn = BCE_loss.BCE_loss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4)
    dset = data_loader.CATH_data(feat_dir = '../../data/features/', partition = 'train')
    scaler = torch.cuda.amp.GradScaler()

    pbar = tqdm.tqdm(total=dset.__len__())
    for seq, Ca_coord, torsion_angles, distance in dset: 
        pbar.update()
        seq_len = seq.size(0)
        seq, Ca_coord, torsion_angles, distance = seq.unsqueeze(0).to(device), Ca_coord.unsqueeze(0).to(device), torsion_angles.unsqueeze(0).to(device), distance.unsqueeze(0).unsqueeze(-1).to(device)
        mask = torch.ones(1, seq_len).bool().to(device)
        

        with torch.cuda.amp.autocast():
            out = model(torsion_angles, Ca_coord, distance, mask)
            loss = loss_fn(out, seq)
        
        