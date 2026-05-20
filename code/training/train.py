import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import time


def train_sardinn(model,
                  train_dataset,
                  val_dataset,
                  checkpoint_dir: str,
                  log_path:        str,
                  n_epochs:        int   = 400,
                  lr:              float = 1e-5,
                  batch_size:      int   = 10,
                  device:          str   = 'cuda',
                  resume_checkpoint: str = None):

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        val_dataset, batch_size=1,
        shuffle=False, num_workers=2)

    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    start_epoch   = 1
    best_val_psnr = 0.0
    log           = []

    if resume_checkpoint and Path(resume_checkpoint).exists():
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_psnr = ckpt.get('val_psnr', 0.0)
        print(f"Resumed from epoch {ckpt['epoch']}, "
              f"PSNR={best_val_psnr:.2f} dB", flush=True)

    print(f"Training on {device}", flush=True)
    print(f"Epochs: {start_epoch} to {n_epochs}", flush=True)
    print(f"Batch size: {batch_size}, LR: {lr}", flush=True)
    print(f"Train samples: {len(train_dataset)}, "
          f"Val samples: {len(val_dataset)}", flush=True)

    training_start = time.time()

    for epoch in range(start_epoch, n_epochs + 1):
        epoch_start = time.time()

        # ── Training ──────────────────────────────────────────────
        model.train()
        train_losses = []

        for batch in train_loader:
            I_in     = batch['I_in'].to(device)
            I_target = batch['I_target'].to(device)
            q_star   = batch['q_star'].to(device)
            q_in     = batch['q_in'].to(device)

            optimizer.zero_grad()
            I_pred = model(I_in, q_star, q_in)
            loss   = criterion(I_pred, I_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        epoch_time    = time.time() - epoch_start
        mean_loss     = float(np.mean(train_losses))
        elapsed_total = (time.time() - training_start) / 3600
        eta_hours     = (epoch_time * (n_epochs - epoch)) / 3600

        # ── Validation every 10 epochs ────────────────────────────
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            val_psnrs = []

            with torch.no_grad():
                for batch in val_loader:
                    I_in     = batch['I_in'].to(device)
                    I_target = batch['I_target'].to(device)
                    q_star   = batch['q_star'].to(device)
                    q_in     = batch['q_in'].to(device)
                    I_pred   = model(I_in, q_star, q_in)
                    mse      = ((I_pred - I_target)**2).mean().item()
                    psnr     = -10.0 * np.log10(mse + 1e-8)
                    val_psnrs.append(psnr)

            mean_psnr = float(np.mean(val_psnrs))

            print(f"Epoch {epoch:4d}/{n_epochs} | "
                  f"Loss: {mean_loss:.6f} | "
                  f"PSNR: {mean_psnr:.2f} dB | "
                  f"Ep: {epoch_time:.1f}s | "
                  f"Elapsed: {elapsed_total:.2f}h | "
                  f"ETA: {eta_hours:.2f}h",
                  flush=True)

            if mean_psnr > best_val_psnr:
                best_val_psnr = mean_psnr
                torch.save({
                    'epoch':           epoch,
                    'model_state':     model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_psnr':        mean_psnr,
                    'train_loss':      mean_loss,
                }, f"{checkpoint_dir}/best_model.pth")
                print(f"  → Saved best checkpoint (PSNR {mean_psnr:.2f})",
                      flush=True)

            if epoch % 50 == 0:
                torch.save({
                    'epoch':       epoch,
                    'model_state': model.state_dict(),
                    'val_psnr':    mean_psnr,
                }, f"{checkpoint_dir}/epoch_{epoch:04d}.pth")

            log.append({
                'epoch':      epoch,
                'loss':       mean_loss,
                'psnr':       mean_psnr,
                'epoch_time': epoch_time,
            })
            with open(log_path, 'w') as f:
                json.dump(log, f, indent=2)

        else:
            print(f"Epoch {epoch:4d}/{n_epochs} | "
                  f"Loss: {mean_loss:.6f} | "
                  f"Ep: {epoch_time:.1f}s | "
                  f"Elapsed: {elapsed_total:.2f}h | "
                  f"ETA: {eta_hours:.2f}h",
                  flush=True)

    print(f"\nTraining complete. Best PSNR: {best_val_psnr:.2f} dB",
          flush=True)
    return log
