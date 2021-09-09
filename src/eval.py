import torch
import argparse
from models import EnNet,data_loader
import numpy as np
import tqdm
from se3_transformer_pytorch.utils import fourier_encode
import torch.nn.functional as F

def sample(pred_dist):
    out_idx = torch.multinomial(torch.tensor(pred_dist), 1)
    return out_idx

def recovery(pred, gts, masks):
    accuracy = []
    for i, pred in enumerate(preds):
        gt = gts[i]
        mask = masks[i]
        c = 0
        for j in range(mask.sum()):
            pos_gt = gt[j,:]
            pos_pred = pred[j,:]
            gt_id = np.argmax(pos_gt)
            pred_id = np.argmax(pos_pred)
            if gt_id==pred_id:
                c += 1
        accuracy.append(c/mask.sum() * 1.0)
    recovery = np.mean(accuracy)
    return recovery

def perplexity(preds, gts, masks):
    prob = []
    for i, pred in enumerate(preds):
        gt = gts[i]
        mask = masks[i]
        for j in range(mask.sum()):
            pos_gt = gt[j,:]
            pos_pred = pred[j,:]
            gt_id = np.argmax(pos_gt)
            prob.append(np.log(pos_pred[gt_id]))
    
    perp = np.exp(-np.mean(prob))
    return perp
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training script for EnNet')
    parser.add_argument(
        '--Device', type=str, help='CUDA device for training', default='0')
    parser.add_argument(
        '--Model_path', type = str, help='path to the model to be tested'
    )

    args = parser.parse_args() 
    device = torch.device('cuda:' + args.Device)
    model = EnNet.EnNet(device = device)
    
    state_dcit = torch.load(args.Model_path, map_location = torch.device('cpu'))
    model.load_state_dict(state_dcit['state_dict'])
    model.to(device).eval()

    test_dset = data_loader.CATH_data(feat_dir = '../data/features/', partition = 'test')
    
    tmp_preds = []
    preds = []
    gts = []
    masks = []
    pbar = tqdm.tqdm(total = test_dset.__len__())
    for seq, seq_shifted,  Ca_coord, torsion_angles, distance, pad_mask in test_dset:
        pbar.update()
        seq_shifted, Ca_coord, torsion_angles, distance = seq_shifted.unsqueeze(0).to(device), Ca_coord.unsqueeze(0).to(device), torsion_angles.unsqueeze(0).to(device), distance.unsqueeze(0).unsqueeze(-1).to(device)
        pad_mask = pad_mask.unsqueeze(0).to(device)
        #distance = fourier_encode(distance, num_encodings  = 8, include_self = True)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                logits = model(torsion_angles, Ca_coord, distance, pad_mask, seq_shifted)
                tmp_prob = F.softmax(logits/0.1, dim = 2)
                prob = F.softmax(logits, dim = 2)
        preds.append(prob.cpu().detach().numpy().squeeze())
        tmp_preds.append(tmp_prob.cpu().detach().numpy().squeeze())
        gts.append(seq.cpu().detach().numpy())
        masks.append(pad_mask.cpu().detach().numpy().squeeze())
    perp = perplexity(preds, gts, masks)
    rec = recovery(tmp_preds, gts, masks)
    print(perp)
    print(rec)
        
        
        
        

        


