import argparse
import json
import numpy as np
import os
from pathlib import Path
import glob
import time
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import swin_mae
from utils.engine_pretrain import train_one_epoch
import h5py

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # common parameters
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--checkpoint_encoder', default='', type=str)
    parser.add_argument('--checkpoint_decoder', default='', type=str)
   # fill in the dataset path here
    parser.add_argument('--mask_ratio', default=0.35, type=float,
                        help='Masking ratio (percentage of removed patches).')
    # model parameters
    parser.add_argument('--model', default='swin_mae', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    # optimizer parameters
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    # other parameters
    parser.add_argument('--output_dir', default='./output_dir_T2',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_T2/Masking0.35',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=1, type=int)
    #parser.add_argument('--pin_mem', action='store_true',
                        #help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    #parser.set_defaults(pin_mem=True)
    return parser

import sys
sys.path.insert(0,'/content')  # TODO: Update to your Google Drive path if needed for swin_mae or other custom modules

# Custom dataset to load preprocessed .h5 files (each containing multiple slices)
class H5SliceDataset(torch.utils.data.Dataset):
    def __init__(self, h5_dir):
        self.h5_paths = sorted(glob.glob(os.path.join(h5_dir, '*.h5')))
        self.cum_lengths = [0]
        for path in self.h5_paths:
            with h5py.File(path, 'r') as f:
                n_slices = f['image'].shape[3]  # Assuming shape (1, H, W, D)
            self.cum_lengths.append(self.cum_lengths[-1] + n_slices)

    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, idx):
        # Find which file and slice
        file_idx = np.searchsorted(self.cum_lengths, idx, side='right') - 1
        slice_idx = idx - self.cum_lengths[file_idx]
        path = self.h5_paths[file_idx]
        with h5py.File(path, 'r') as f:
            img_slice = f['image'][0, :, :, slice_idx]  # (H, W)
        img_slice = torch.from_numpy(img_slice).float().unsqueeze(0)  # (1, H, W)
        img_slice = img_slice.repeat(3, 1, 1)  # Repeat to 3 channels for MAE (3, H, W)
        return img_slice  # Note: Labels are not used for MAE pretraining

class AverageMeter(object):
    """
    compute and store the average and current value
    """
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import math

def main(args):
    # Fixed random seeds
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Set up training equipment
    device = torch.device("cuda")
    cudnn.benchmark = True

    ############## Data loader using preprocessed .h5 files
    h5_dir = '/content/drive/MyDrive/ie643_course_project_24M1644/Swin_mae_data/train'  # TODO: Update to your actual preprocessed .h5 directory (e.g., output_dir/train from preprocessing)
    dataset = H5SliceDataset(h5_dir)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # Log output
    model_dir = '/content/drive/MyDrive/ie643_course_project_24M1644/swin_saved_models'  # TODO: Update to your Google Drive path
    save_dir = '/content/drive/MyDrive/ie643_course_project_24M1644/swin_mae_pretrain_wt'  # TODO: Update to your Google Drive path
    log_writer = SummaryWriter(log_dir=save_dir)
    # Set model
    model = swin_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, mask_ratio=args.mask_ratio)
    model_without_ddp = model
    # Set optimizer
    param_groups = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=5e-2, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    # Load pretrained weights
    state_dict = torch.load('/content/drive/MyDrive/ie643_course_project_24M1644/swin_mae_pretrain_wt/checkpoint-250_T2.pth')  # TODO: Update to your Google Drive path
    swin_unet_t = state_dict['model']
    key_swin = list(swin_unet_t.keys())
    for i, key_mae in enumerate(model_without_ddp.state_dict()):
        if key_mae in key_swin:
            model_without_ddp.state_dict()[key_mae] = swin_unet_t[key_mae]
    model_without_ddp.to(device)
    # Start the training process
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model_without_ddp, train_loader,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and ((epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch + 1)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

if __name__ == '__main__':
    arg = get_args_parser()
    arg = arg.parse_args()
    if arg.output_dir:
        Path(arg.output_dir).mkdir(parents=True, exist_ok=True)
    main(arg)