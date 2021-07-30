import torch
import argparse
from models import EnNet,data_loader
import numpy as np
import tqdm
from se3_transformer_pytorch.utils import fourier_encode

def sample(pred_dist):
    out_idx = torch.multinomial(torch.tensor(pred_dist), 1)
    return out_idx

def recovery(pred, gts):
    accuracy = []
    for i, pred in enumerate(preds):
        gt = gts[i]
        c = 0
        for j in range(gt.shape[0]):
            pos_gt = gt[j,:]
            pos_pred = pred[j,:]
            gt_id = np.argmax(pos_gt)
            pred_id = np.argmax(pos_pred)
            if gt_id==pred_id:
                c += 1
        accuracy.append(c/gt.shape[0] * 1.0)
    recovery = np.mean(accuracy)
    return recovery

def perplexity(preds, gts):
    prob = []
    for i, pred in enumerate(preds):
        gt = gts[i]
        for j in range(gt.shape[0]):
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
    
    preds = []
    gts = []
    pbar = tqdm.tqdm(total = test_dset.__len__())
    for seq, Ca_coord, torsion_angles, distance in test_dset:
        pbar.update()
        Ca_coord, torsion_angles, distance = Ca_coord.unsqueeze(0).to(device), torsion_angles.unsqueeze(0).to(device), distance.unsqueeze(0).unsqueeze(-1).to(device)
        mask = torch.ones(1, seq.size(0)).bool().to(device) 
        distance = fourier_encode(distance, num_encodings  = 8, include_self = True)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                log_prob = model(torsion_angles, Ca_coord, distance, mask)

        preds.append(np.exp(log_prob.cpu().detach().numpy().squeeze()))
        #preds.append(out)
        gts.append(seq.detach().numpy())
    perp = perplexity(preds, gts)
    rec = recovery(preds, gts)
    print(perp)
    print(rec)
        
        
        
        

        


