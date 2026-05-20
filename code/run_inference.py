#!/usr/bin/env python3
"""
Fast correct inference for SARDInn.
Key insight: run DWFE once per slice (expensive), 
then loop IRR over target directions (cheap).
"""

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from skimage.metrics import structural_similarity as ssim_func
import json
import time
import sys

sys.path.insert(0, '/scratch/rkabir5/fyexam/code/model')
sys.path.insert(0, '/scratch/rkabir5/fyexam/code/data')

from sardinn import SARDInn
from fps import fps_on_sphere


CHECKPOINT    = '/scratch/rkabir5/fyexam/checkpoints/baseline/best_model.pth'
DATA_DIR      = Path('/scratch/rkabir5/fyexam/data/hcp_processed')
SUBJECTS_FILE = '/scratch/rkabir5/fyexam/subjects.txt'
RESULTS_DIR   = Path('/scratch/rkabir5/fyexam/results/baseline_r3')
SHELL         = 'b1000'
SCALE_R       = 3.0
N_SLICES      = 72
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_metrics(pred, target):
    h, w  = target.shape
    rmse  = np.sqrt(np.sum((pred - target)**2) / (h * w))
    psnr  = 10.0 * np.log10(target.max()**2 / (rmse**2 + 1e-10))
    s     = ssim_func(pred, target, data_range=1.0)
    return float(psnr), float(s), float(rmse)


def load_test_subjects():
    subjects = []
    with open(SUBJECTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith('TEST'):
                subj = line.split()[1]
                subjects.append(str(DATA_DIR / subj))
    return subjects


def reconstruct_subject(model, subj_dir):
    subj_id    = Path(subj_dir).name
    data_path  = Path(subj_dir) / f'{SHELL}_data.nii.gz'
    bvecs_path = Path(subj_dir) / f'{SHELL}_bvecs.txt'

    data  = nib.load(str(data_path)).get_fdata(dtype=np.float32)
    bvecs = np.loadtxt(str(bvecs_path))
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T

    N = bvecs.shape[0]

    N_in   = max(int(round(N / SCALE_R)), 3)
    in_idx = fps_on_sphere(bvecs, N_in, seed=42)
    q_in   = bvecs[in_idx]

    tar_mask = np.ones(N, dtype=bool)
    tar_mask[in_idx] = False
    tar_idx  = np.where(tar_mask)[0]
    q_tar    = bvecs[tar_idx]
    N_tar    = len(tar_idx)

    total_z = data.shape[2]
    z_start = (total_z - N_SLICES) // 2

    # Fixed tensors
    q_in_t = torch.from_numpy(
        q_in.astype(np.float32)
    ).unsqueeze(0).to(DEVICE)   # (1, N_in, 3)

    slice_psnrs, slice_ssims, slice_rmses = [], [], []

    model.eval()
    with torch.no_grad():
        for z in range(z_start, z_start + N_SLICES):
            slice_data = data[:, :, z, :]

            # Normalize input
            I_in_raw  = slice_data[:, :, in_idx]
            I_in_norm = np.zeros_like(I_in_raw)
            for i in range(N_in):
                mx = I_in_raw[:, :, i].max()
                if mx > 0:
                    I_in_norm[:, :, i] = I_in_raw[:, :, i] / mx

            I_in_t = torch.from_numpy(
                I_in_norm.transpose(2, 0, 1).astype(np.float32)
            ).unsqueeze(0).to(DEVICE)   # (1, N_in, H, W)

            # ── Run DWFE once for this slice ──────────────────────
            M = model.dwfe(I_in_t)   # (1, N_in, D, H, W)

            # Normalize GT
            gt_raw  = slice_data[:, :, tar_idx]
            gt_norm = np.zeros_like(gt_raw)
            for t in range(N_tar):
                mx = gt_raw[:, :, t].max()
                if mx > 0:
                    gt_norm[:, :, t] = gt_raw[:, :, t] / mx

            dir_psnrs, dir_ssims, dir_rmses = [], [], []

            # ── Loop only IRR over target directions ──────────────
            for t_idx in range(N_tar):
                q_star_t = torch.from_numpy(
                    q_tar[t_idx].astype(np.float32)
                ).unsqueeze(0).to(DEVICE)   # (1, 3)

                # IRR only — DWFE already done
                pred    = model.irr(M, q_star_t, q_in_t)  # (1,1,H,W)
                pred_np = pred.squeeze().cpu().numpy()

                p, s, r = compute_metrics(pred_np, gt_norm[:, :, t_idx])
                dir_psnrs.append(p)
                dir_ssims.append(s)
                dir_rmses.append(r)

            slice_psnrs.append(np.mean(dir_psnrs))
            slice_ssims.append(np.mean(dir_ssims))
            slice_rmses.append(np.mean(dir_rmses))

    return {
        'subject':   subj_id,
        'psnr_mean': float(np.mean(slice_psnrs)),
        'psnr_std':  float(np.std(slice_psnrs)),
        'ssim_mean': float(np.mean(slice_ssims)),
        'ssim_std':  float(np.std(slice_ssims)),
        'rmse_mean': float(np.mean(slice_rmses)),
        'rmse_std':  float(np.std(slice_rmses)),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Strategy: DWFE once per slice, IRR loops over 60 target directions")

    print(f"\nLoading: {CHECKPOINT}")
    model = SARDInn().to(DEVICE)
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded epoch {ckpt['epoch']}, val PSNR={ckpt['val_psnr']:.2f} dB")

    test_subjects = load_test_subjects()
    print(f"\nTest subjects: {len(test_subjects)}")

    all_results = []
    t_start = time.time()

    for i, subj_dir in enumerate(test_subjects):
        subj_id = Path(subj_dir).name
        t0 = time.time()
        print(f"\n[{i+1}/{len(test_subjects)}] {subj_id}...", flush=True)

        result = reconstruct_subject(model, subj_dir)
        elapsed = time.time() - t0
        all_results.append(result)

        print(f"  PSNR: {result['psnr_mean']:.2f} ± {result['psnr_std']:.2f} dB")
        print(f"  SSIM: {result['ssim_mean']:.4f} ± {result['ssim_std']:.4f}")
        print(f"  RMSE: {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}")
        print(f"  Time: {elapsed:.0f}s", flush=True)

    all_psnr = [r['psnr_mean'] for r in all_results]
    all_ssim = [r['ssim_mean'] for r in all_results]
    all_rmse = [r['rmse_mean'] for r in all_results]

    summary = {
        'variant':        'baseline',
        'scale_r':        SCALE_R,
        'shell':          SHELL,
        'n_subjects':     len(test_subjects),
        'psnr_mean':      float(np.mean(all_psnr)),
        'psnr_std':       float(np.std(all_psnr)),
        'ssim_mean':      float(np.mean(all_ssim)),
        'ssim_std':       float(np.std(all_ssim)),
        'rmse_mean':      float(np.mean(all_rmse)),
        'rmse_std':       float(np.std(all_rmse)),
        'per_subject':    all_results,
        'total_time_min': (time.time() - t_start) / 60,
    }

    print(f"\n{'='*55}")
    print(f"OVERALL — baseline, scale r={SCALE_R}, {SHELL}")
    print(f"{'='*55}")
    print(f"PSNR: {summary['psnr_mean']:.2f} ± {summary['psnr_std']:.2f} dB")
    print(f"SSIM: {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
    print(f"RMSE: {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f}")
    print(f"Total time: {summary['total_time_min']:.1f} min")

    out_path = RESULTS_DIR / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
