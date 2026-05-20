#!/usr/bin/env python3
"""
Main training script for SARDInn.
Usage: python3 run_training.py --variant baseline --mlp_layers 8 --sine_layer 4
"""

import argparse
import torch
from pathlib import Path
from torch.utils.data import random_split
import sys

sys.path.insert(0, '/scratch/rkabir5/fyexam/code/model')
sys.path.insert(0, '/scratch/rkabir5/fyexam/code/data')
sys.path.insert(0, '/scratch/rkabir5/fyexam/code/training')

from sardinn import SARDInn
from dataset import HCPSliceDataset
from train   import train_sardinn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--variant',      default='baseline')
    p.add_argument('--mlp_layers',   type=int,   default=8)
    p.add_argument('--sine_layer',   type=int,   default=4)
    p.add_argument('--n_epochs',     type=int,   default=400)
    p.add_argument('--lr',           type=float, default=1e-5)
    p.add_argument('--batch_size',   type=int,   default=10)
    p.add_argument('--data_dir',
                   default='/scratch/rkabir5/fyexam/data/hcp_processed')
    p.add_argument('--subjects_file',
                   default='/scratch/rkabir5/fyexam/subjects.txt')
    p.add_argument('--checkpoint_dir',
                   default='/scratch/rkabir5/fyexam/checkpoints/baseline')
    p.add_argument('--log_path',
                   default='/scratch/rkabir5/fyexam/logs/baseline/log.json')
    return p.parse_args()


def load_subjects(subjects_file, data_dir):
    train_dirs, test_dirs = [], []
    with open(subjects_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            split, subj = line.split()
            path = str(Path(data_dir) / subj)
            if split == 'TRAIN':
                train_dirs.append(path)
            else:
                test_dirs.append(path)
    return train_dirs, test_dirs


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Variant:    {args.variant}")
    print(f"MLP layers: {args.mlp_layers}, Sine at layer: {args.sine_layer}")
    print(f"Device:     {device}")
    if device == 'cuda':
        print(f"GPU:        {torch.cuda.get_device_name(0)}")

    # Load subject directories
    train_dirs, test_dirs = load_subjects(
        args.subjects_file, args.data_dir)
    print(f"Train subjects: {len(train_dirs)}")
    print(f"Test subjects:  {len(test_dirs)}")

    # Build datasets
    train_dataset = HCPSliceDataset(
        train_dirs, shell='b1000', scale_r=3.0, n_slices=72)
    val_dataset   = HCPSliceDataset(
        test_dirs, shell='b1000', scale_r=3.0, n_slices=72)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # Build model
    model = SARDInn(
        n_resblocks=18,
        n_feats=64,
        embed_dim=128,
        mlp_hidden=256,
        mlp_layers=args.mlp_layers,
        sine_layer=args.sine_layer,
        sigma_q=0.5,
    )

    params = model.count_parameters()
    print(f"Model parameters: {params['total']:,}")

    # Train
    log = train_sardinn(
        model          = model,
        train_dataset  = train_dataset,
        val_dataset    = val_dataset,
        checkpoint_dir = args.checkpoint_dir,
        log_path       = args.log_path,
        n_epochs       = args.n_epochs,
        lr             = args.lr,
        batch_size     = args.batch_size,
        device         = device,
        resume_checkpoint = f"{args.checkpoint_dir}/best_model.pth",
    )

    print(f"Done. Log saved to {args.log_path}")


if __name__ == '__main__':
    main()
