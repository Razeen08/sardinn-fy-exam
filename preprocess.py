#!/usr/bin/env python3
"""
Preprocessing for SARDInn training.
Step 1: Shell separation — split 288-volume data.nii.gz into b1000, b2000, b3000
Step 2: Patch2Self denoising — applied per shell per subject
Output saved to /scratch/rkabir5/fyexam/data/hcp_processed/{subject_id}/
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from dipy.core.gradients import gradient_table
from dipy.denoise.patch2self import patch2self
import time

RAW_DIR  = Path('/scratch/rkabir5/fyexam/data/hcp_raw')
PROC_DIR = Path('/scratch/rkabir5/fyexam/data/hcp_processed')
SHELLS   = [1000, 2000, 3000]
TOL      = 100   # b-value tolerance — volumes within ±100 of target count as that shell

def separate_and_denoise(subject_id):
    raw_dir  = RAW_DIR  / subject_id
    proc_dir = PROC_DIR / subject_id
    proc_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"Subject: {subject_id}")
    print(f"{'='*55}")

    # Load full data
    print("  Loading data.nii.gz...", flush=True)
    img   = nib.load(str(raw_dir / 'data.nii.gz'))
    data  = img.get_fdata(dtype=np.float32)   # (145, 174, 145, 288)
    affine = img.affine
    bvals  = np.loadtxt(str(raw_dir / 'bvals'))
    bvecs  = np.loadtxt(str(raw_dir / 'bvecs'))

    # Ensure bvecs is (3, N)
    if bvecs.shape[0] != 3:
        bvecs = bvecs.T

    print(f"  Data shape: {data.shape}")
    print(f"  b-values unique: {np.unique(np.round(bvals, -2)).astype(int)}")

    # ── Process each shell ──────────────────────────────────────────
    for shell_b in SHELLS:
        out_data  = proc_dir / f'b{shell_b}_data.nii.gz'
        out_bvals = proc_dir / f'b{shell_b}_bvals.txt'
        out_bvecs = proc_dir / f'b{shell_b}_bvecs.txt'

        if out_data.exists():
            print(f"  SKIP b{shell_b} (already processed)")
            continue

        # Find b0 + this shell indices
        b0_idx    = np.where(bvals < 50)[0]
        shell_idx = np.where(np.abs(bvals - shell_b) < TOL)[0]
        all_idx   = np.concatenate([b0_idx, shell_idx])
        all_idx   = np.sort(all_idx)

        shell_data  = data[..., all_idx]           # (145, 174, 145, N_b0+N_shell)
        shell_bvals = bvals[all_idx]
        shell_bvecs = bvecs[:, all_idx]

        print(f"\n  Shell b{shell_b}:")
        print(f"    Volumes: {len(b0_idx)} b0 + {len(shell_idx)} DWI = {len(all_idx)} total")

        # ── Patch2Self denoising ──────────────────────────────────
        print(f"    Running Patch2Self...", flush=True)
        t0 = time.time()

        # gradient_table needs bvecs as (N, 3)
        gtab = gradient_table(shell_bvals, shell_bvecs.T)

        denoised = patch2self(
            shell_data,
            bvals=shell_bvals,
            model='ols',          # ordinary least squares — stable
            shift_intensity=False,
            clip_negative_vals=True,
            b0_threshold=50,
            verbose=False,
        )

        elapsed = time.time() - t0
        print(f"    Patch2Self done in {elapsed:.0f}s")

        # Keep only DWI directions (drop b0 from saved output)
        # SARDInn trains on DWI only — b0 used internally by Patch2Self
        dwi_only_idx = np.arange(len(b0_idx), len(all_idx))
        denoised_dwi  = denoised[..., dwi_only_idx]
        final_bvals   = shell_bvals[dwi_only_idx]
        final_bvecs   = shell_bvecs[:, dwi_only_idx]

        print(f"    Saving {denoised_dwi.shape[3]} DWI volumes...")

        # Save
        nib.save(nib.Nifti1Image(denoised_dwi, affine), str(out_data))
        np.savetxt(str(out_bvals), final_bvals, fmt='%g')
        np.savetxt(str(out_bvecs), final_bvecs, fmt='%.6f')

        size_gb = out_data.stat().st_size / 1e9
        print(f"    Saved: b{shell_b}_data.nii.gz ({size_gb:.2f} GB)")

    print(f"\n  Subject {subject_id} complete.")


def main():
    # Read subject list
    subjects = []
    with open('/scratch/rkabir5/fyexam/subjects.txt') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            _, subj = line.split()
            subjects.append(subj)

    print(f"Preprocessing {len(subjects)} subjects")
    print(f"Output: {PROC_DIR}\n")

    t_total = time.time()
    for i, subj in enumerate(subjects):
        print(f"\n[{i+1}/{len(subjects)}]", end='')
        try:
            separate_and_denoise(subj)
        except Exception as e:
            print(f"\n  ERROR on {subj}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = (time.time() - t_total) / 3600
    print(f"\n\nAll done. Total time: {elapsed:.1f} hours")

    # Final verification
    print("\nVerification:")
    for subj in subjects:
        proc_dir = PROC_DIR / subj
        missing  = []
        for b in SHELLS:
            if not (proc_dir / f'b{b}_data.nii.gz').exists():
                missing.append(f'b{b}')
        if missing:
            print(f"  INCOMPLETE {subj}: missing shells {missing}")
        else:
            sizes = [
                nib.load(str(proc_dir/f'b{b}_data.nii.gz')).shape
                for b in SHELLS
            ]
            print(f"  OK {subj}: {sizes}")

if __name__ == '__main__':
    main()
