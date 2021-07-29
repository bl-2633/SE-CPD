import numpy as np
import sys
sys.path.append('/mnt/storage_1/blai/projects/SE-CPD/src/data_processing/')
from numpy.core.fromnumeric import partition
import torch
from torch.utils import data as D
import data_utils
import tqdm


class CATH_data(D.Dataset):
    def __init__(self, feat_dir, partition):
        self.feat_dir = feat_dir
        assert partition in ['train', 'test', 'validation'], 'Please indicate train, test, validation partition'
        self.chain_list = open('/mnt/storage_1/blai/projects/SE-CPD/data/' + partition + '.txt', 'r').readlines()
        
        
    def __getitem__(self, index):
        feat_path = self.feat_dir + self.chain_list[index].strip()
        feat = torch.load(feat_path)
        seq = feat['seq']
        Ca_coord = feat['Ca_coord']
        torsion_angles = feat['torsion_angles']
        distance = feat['distance']
        
        return seq, Ca_coord, torsion_angles, distance
    
    def __len__(self):
        return len(self.chain_list)


if __name__ == '__main__':
    dset = CATH_data(feat_dir='../../data/features/', partition = 'train')
    pbar = tqdm.tqdm(total = dset.__len__())
    for seq, Ca_coord, torsion_angles, distance in dset:
        #print(seq.size())
        #print(Ca_coord.size())
        #print(torsion_angles.size())
        #print(distance.size())
        try:
            assert(seq.size(0) == Ca_coord.size(0) == torsion_angles.size(0) == distance.size(0))
        except:
            print(seq.size())
            print(Ca_coord.size())
            print(torsion_angles.size())
            print(distance.size())
            
        
        