import SE3Net, EnNet
import torch
import argparse


def bytesto(bytes, to, bsize=1024): 
        a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
        r = float(bytes)
        return bytes / (bsize ** a[to])



if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = EnNet.EnNet().to(device)
    seq_len = 500
    feat_n = 64
    coors_n = 3
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4)
    scaler = torch.cuda.amp.GradScaler()

    
    for i in range(20):
        optimizer.zero_grad()
        feats = torch.randn(1, seq_len, feat_n, dtype = torch.half).to(device)
        coors = 5 * torch.randn(1, seq_len, coors_n, dtype = torch.half).to(device)
        edges = 10 * torch.randn(1, seq_len, seq_len, 1).to(device)
        mask  = torch.ones(1, seq_len, dtype = torch.half).bool().to(device)
        gt = torch.ones(1, seq_len, 20).to(device)
        with torch.cuda.amp.autocast():            
            pred_out = model(feats, coors, edges, mask)
            loss = loss_fn(gt, pred_out)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


    max_mem = torch.cuda.max_memory_allocated(device)
    print('Sequence lenth = ' + str(seq_len))
    print('Feature Dimension = ' + str(feat_n))
    print('Coordinate Dimension = ' + str(coors_n))
    print('VRAM peak = ' + str(bytesto(max_mem, to = 'g'))[:6] + 'G')