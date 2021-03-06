import argparse
from functools import WRAPPER_UPDATES
import sys
from models import data_loader, EnNet, BCE_loss, NLL_loss, utils
#import torch.optim as optim
import torch_optimizer as optim
import torch
import numpy as np
from tqdm import tqdm
from se3_transformer_pytorch.utils import fourier_encode
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import autograd

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
    step = param_dict['step']
    warmup = param_dict['warmup']
    writer = param_dict['tb_writer']
    scaler = torch.cuda.amp.GradScaler()

    total_loss = 0
    s = 0
    for seq, seq_shifted, Ca_coord, torsion_angles, vec_feats, distance, mask in train_loader:
        optimizer.zero_grad()
        seq, seq_shifted, Ca_coord, torsion_angles, vec_feats, distance = seq.to(device), seq_shifted.to(device), Ca_coord.to(device), torsion_angles.to(device), vec_feats.to(device), distance.unsqueeze(-1).to(device)
        mask = mask.to(device)
        
        with torch.cuda.amp.autocast():
            out = model(torsion_angles, Ca_coord, distance, vec_feats, mask, seq_shifted)    
            loss = loss_fn(out, seq, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        step += 1 
        total_loss += loss.item()
        pbar.update()
        s+=1
        pbar.set_description(str(epoch) + '/' + str(param_dict['train_epoch']) + ' ' + str(total_loss/s)[:6])
        writer.add_scalar('Loss/train', loss.item() , step)
            
        
    val_loss = []
    model.eval()
    for seq, seq_shifted,  Ca_coord, torsion_angles, vec_feats, distance, mask in val_loader:
        seq, seq_shifted, Ca_coord, torsion_angles, vec_feat, distance = seq.to(device), seq_shifted.to(device) , Ca_coord.to(device), torsion_angles.to(device), vec_feats.to(device), distance.unsqueeze(-1).to(device)
        mask = mask.to(device) 
        with torch.cuda.amp.autocast():
            out = model(torsion_angles, Ca_coord, distance, vec_feats, mask, seq_shifted)
            loss = loss_fn(out, seq, mask)
        val_loss.append(loss.item())
    pbar.set_description(str(epoch) + '/' + str(np.mean(val_loss))[:6])
    writer.add_scalar('Loss/val', np.mean(val_loss), epoch)

    model_states = {
        "epoch":epoch,
        "state_dict":model.module.state_dict(),
        "optimizer":optimizer.state_dict(),
        "loss":running_loss
    }
    torch.save(model_states, model_path)
    pbar.close()
    return np.mean(val_loss), step

if __name__ == "__main__":
    print('------------Starting------------' + '\n')

    parser = argparse.ArgumentParser(
        description='Training script for EnNet')
    parser.add_argument(
        '--Device', type=str, help='CUDA device for training', default='0')

    args = parser.parse_args()
    train_set = data_loader.CATH_data(feat_dir = '../data/features/', partition = 'train')
    val_set = data_loader.CATH_data(feat_dir = '../data/features/', partition = 'validation')
    train_set = DataLoader(train_set, batch_size = 16, num_workers = 5, shuffle=True)
    val_set = DataLoader(val_set, batch_size = 1, num_workers = 5, shuffle = False)


    device = torch.device('cuda:'+ args.Device)
    model = EnNet.EnNet(device = device)
    model = torch.nn.DataParallel(model, device_ids=[1, 2, 3]).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    '''
    optimizer = optim.Adafactor(
    model.parameters(),
    lr= 1e-3,
    eps2= (1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.1,
    scale_parameter=True,
    relative_step=True,
    warmup_init=False,
    )
    '''
    loss_fn = BCE_loss.BCE_loss()
    model_out = '../trained_models/EnTransformers/EnNet_Transformer_12_512'

    param_dict = {
        'train_epoch': 100,
        'model': model,
        'optim': optimizer,
        'loss_fn': loss_fn,
        'train_loader': train_set,
        'val_loader': val_set,
        'device': device,
        'step': 0,
        'warmup': 4000,
        'tb_writer': SummaryWriter(log_dir = 'runs/EnNet_Transformer_12_256')
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
        val_loss, step = train(param_dict=param_dict, epoch=epoch, model_path=model_path)
        param_dict['step'] = step
        if val_loss < min_loss:
            early_stop_cout = 0
            min_loss = val_loss
        else:
            early_stop_cout += 1
        if early_stop_cout >= 15:
            break
