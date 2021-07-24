import argparse
import sys
from models import data_loader, SE3Net, BCE_loss
import torch.optim as optim
import torch
import numpy as np
from tqdm import tqdm



torch.manual_seed(100)

def train(param_dict, epoch, model_path):
    model = param_dict['model'].train()
    running_loss = []
    pbar = tqdm(total = len(param_dict['train_loader']))
    device = param_dict['device']
    loss_fn = param_dict['loss_fn']
    train_loader = param_dict['train_loader']
    val_loader = param_dict['val_loader']
    optimizer = param_dict['optim']
    scaler = torch.cuda.amp.GradScaler()

    c = 0
    for seq, Ca_coord, torsion_angles, distance in train_loader:
        optimizer.zero_grad()
        seq, Ca_coord, torsion_angles, distance = seq.unsqueeze(0).to(device), Ca_coord.unsqueeze(0).to(device), torsion_angles.unsqueeze(0).to(device), distance.unsqueeze(0).unsqueeze(-1).to(device)
        mask = torch.ones(1, seq.size(1)).bool().to(device) 
        with torch.cuda.amp.autocast():
            out = model(torsion_angles, Ca_coord, mask, edges = distance)
            loss = loss_fn(out, seq)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss.append(loss.item())
        pbar.update()
        c += 1
        if c == 100:
            pbar.set_description(str(epoch) + '/' + str(param_dict['train_epoch']) + ' ' + str(np.mean(running_loss))[:6])
            c = 0
        
    val_loss = []
    model.eval()
    for seq, Ca_coord, torsion_angles, distance in val_loader:
        seq, Ca_coord, torsion_angles, distance = seq.unsqueeze(0).to(device), Ca_coord.unsqueeze(0).to(device), torsion_angles.unsqueeze(0).to(device), distance.unsqueeze(0).unsqueeze(-1).to(device)
        mask = torch.ones(1, seq.size(1)).bool().to(device)
        with torch.cuda.amp.autocast():
            out = model(torsion_angles, Ca_coord, mask, edges = distance)
            loss = loss_fn(out, seq)
        val_loss.append(loss.item())
    pbar.set_description(str(epoch) + '/' + str(np.mean(val_loss))[:6])

    model_states = {
        "epoch":epoch,
        "state_dict":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "loss":running_loss
    }
    torch.save(model_states, model_path)
    pbar.close()
    return np.mean(val_loss)

if __name__ == "__main__":
    print('------------Starting------------' + '\n')

    parser = argparse.ArgumentParser(
        description='Training script for EnNet')
    parser.add_argument(
        '--Device', type=str, help='CUDA device for training', default='0')

    args = parser.parse_args()
    train_set = data_loader.CATH_data(feat_dir = '../data/features/', partition = 'train')
    val_set = data_loader.CATH_data(feat_dir = '../data/features/', partition = 'validation')

    device = torch.device('cuda:'+ args.Device)
    model = SE3Net.SE3Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-5)
    loss_fn = BCE_loss.BCE_loss()

    model_out = '../trained_models/SE3Transformers/SE3Net_1'

    param_dict = {
        'train_epoch': 100,
        'model': model,
        'optim': optimizer,
        'loss_fn': loss_fn,
        'train_loader': train_set,
        'val_loader': val_set,
        'device': device,
    }

    print('Number of Training Sequence: ' + str(train_set.__len__()))
    print('Batch Size: ' + str(1))
    print('Learning Rate: ' + str(1e-4))
    print('Max training Epochs: ' + str(100))
    print('Early stopping patience: ' + str(15))
    print('Saving trained model at: ' + model_out)


    early_stop_cout = 0
    pre_val_loss = 100
    min_loss = 100
    for epoch in range(1, 100):
        model_path = model_out
        val_loss = train(param_dict=param_dict, epoch=epoch, model_path=model_path)
        if val_loss < min_loss:
            early_stop_cout = 0
            min_loss = val_loss
        else:
            early_stop_cout += 1
        if early_stop_cout >= 15:
            break
