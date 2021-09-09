import numpy as np
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('/'.join(dir_path.split('/')[:-1]) + '/data_processing')
from numpy.core.fromnumeric import partition
import torch
from torch.utils import data as D
import data_utils
import tqdm


class CATH_data(D.Dataset):
    def __init__(self, feat_dir, partition):
        self.feat_dir = feat_dir
        assert partition in ['train', 'test', 'validation'], 'Please indicate train, test, validation partition'
        self.chain_list = open('/'.join(dir_path.split('/')[:-2]) + '/data/' + partition + '.txt', 'r').readlines()
        self.MAX_LEN = 500
        
    def __getitem__(self, index):
        feat_path = self.feat_dir + self.chain_list[index].strip()
        feat = torch.load(feat_path)
        seq = feat['seq']
        Ca_coord = feat['Ca_coord']
        torsion_angles = feat['torsion_angles']
        distance = feat['distance']
        vec_feats = feat['vec_feats']
        
        seq_len = seq.size(0)
        pad_len = self.MAX_LEN - seq_len
        seq = torch.cat([torch.zeros(1,20), seq], dim = 0)
        padder_1d = torch.nn.ConstantPad1d((0,pad_len), 0)
        padder_2d = torch.nn.ConstantPad2d((0, pad_len, 0, pad_len), 0)
        seq, Ca_coord, torsion_angles, vec_feats = padder_1d(seq.T).T, padder_1d(Ca_coord.T).T, padder_1d(torsion_angles.T).T, padder_1d(vec_feats.T).T
        distance = padder_2d(distance)

        mask = torch.zeros(self.MAX_LEN)
        mask[:seq_len] = 1
        seq_orig = seq[1:,:]
        seq_shifted = seq[:-1, :] 
        return seq_orig.float(), seq_shifted.float(), Ca_coord.float(), torsion_angles.float(), vec_feats.float(), distance.float(), mask.bool()
    
    def __len__(self):
        return len(self.chain_list)


if __name__ == '__main__':
    dset = CATH_data(feat_dir='../../data/features/', partition = 'train')
    pbar = tqdm.tqdm(total = dset.__len__())
    for seq, seq_shifted, Ca_coord, torsion_angles, distance, mask in dset:
        print(seq.size())
        print(seq_shifted.size())
        print(Ca_coord.size())
        print(torsion_angles.size())
        print(distance.size())
        print(mask.sum())
        exit()
        try:
            assert(seq.size(0) == Ca_coord.size(0) == torsion_angles.size(0) == distance.size(0))
        except:
            print(seq.size())
            print(Ca_coord.size())
            print(torsion_angles.size())
            print(distance.size())
            
        
        