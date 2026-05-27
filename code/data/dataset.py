import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '/scratch/rkabir5/fyexam/code/data')
from fps import fps_on_sphere


class HCPSliceDataset(Dataset):
    """
    HCP dMRI dataset for SARDInn training.
    Wang et al. 2025, Section 3.1.

    Loads from hcp_processed/ (Patch2Self already applied).
    Each sample = one 2D axial slice + one query direction.

    Key details:
    - bvecs on disk: shape (3, 90) — transposed to (90, 3) on load
    - Normalization: per-volume max at runtime (paper requirement)
    - Only middle 72 axial slices used (paper Section 3.1)
    - Training shell: b1000 only
    - TSC: dual FPS downsampling with r=3
    """
    def __init__(self,
                 subject_dirs: list,
                 shell:        str   = 'b1000',
                 scale_r:      float = 3.0,
                 n_slices:     int   = 72):
        self.scale_r = scale_r
        self.samples = []

        for subj_dir in subject_dirs:
            subj_dir   = Path(subj_dir)
            data_path  = subj_dir / f'{shell}_data.nii.gz'
            bvecs_path = subj_dir / f'{shell}_bvecs.txt'

            if not data_path.exists():
                print(f"WARNING: {data_path} not found, skipping")
                continue

            # Load data and bvecs
            data  = nib.load(str(data_path)).get_fdata(dtype=np.float32)
            bvecs = np.loadtxt(str(bvecs_path))   # (3, 90) on disk

            # Transpose bvecs to (90, 3)
            if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
                bvecs = bvecs.T   # → (90, 3)

            # Middle n_slices axial slices
            total_z = data.shape[2]
            z_start = (total_z - n_slices) // 2
            z_end   = z_start + n_slices

            for z in range(z_start, z_end):
                self.samples.append((data, bvecs, z))

        print(f"Dataset: {len(self.samples)} samples "
              f"from {len(subject_dirs)} subjects")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        data, bvecs, z = self.samples[idx]
        N = bvecs.shape[0]   # 90 directions

        # Extract axial slice: (H, W, N)
        slice_data = data[:, :, z, :]   # (145, 174, 90)
        H, W, _    = slice_data.shape

        # ── TSC: Dual FPS downsampling ────────────────────────────
        r    = self.scale_r
        N_l  = max(int(round(N / r)), 3)       # 30 LAR directions
        N_in = max(int(round(N_l / r)), 2)     # 10 input directions

        # Step 1: HAR → LAR
        lar_idx = fps_on_sphere(bvecs, N_l)
        q_l     = bvecs[lar_idx]               # (30, 3)
        I_l     = slice_data[:, :, lar_idx]    # (H, W, 30)

        # Step 2: LAR → I_in + I_tar
        in_idx   = fps_on_sphere(q_l, N_in)
        tar_mask = np.ones(N_l, dtype=bool)
        tar_mask[in_idx] = False
        tar_idx  = np.where(tar_mask)[0]

        I_in  = I_l[:, :, in_idx]             # (H, W, N_in)
        I_tar = I_l[:, :, tar_idx]            # (H, W, N_tar)
        q_in  = q_l[in_idx]                   # (N_in, 3)
        q_tar = q_l[tar_idx]                  # (N_tar, 3)

        # Sample one target direction as query
        t        = np.random.randint(len(tar_idx))
        q_star   = q_tar[t].astype(np.float32)      # (3,)
        I_target = I_tar[:, :, t].astype(np.float32) # (H, W)

        # ── Per-volume max normalization (paper requirement) ──────
        I_norm = np.zeros_like(I_in)
        for i in range(N_in):
            mx = I_in[:, :, i].max()
            if mx > 0:
                I_norm[:, :, i] = I_in[:, :, i] / mx

        mx = I_target.max()
        if mx > 0:
            I_target = I_target / mx

        return {
            'I_in':     torch.from_numpy(
                            I_norm.transpose(2, 0, 1).astype(np.float32)),
            'I_target': torch.from_numpy(I_target).unsqueeze(0),
            'q_star':   torch.from_numpy(q_star),
            'q_in':     torch.from_numpy(q_in.astype(np.float32)),
        }


if __name__ == '__main__':
    from pathlib import Path

    PROC_DIR = Path('/scratch/rkabir5/fyexam/data/hcp_processed')
    SUBJ_FILE = Path('/scratch/rkabir5/fyexam/subjects.txt')

    # Read one training subject
    train_subjects = []
    with open(SUBJ_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith('TRAIN'):
                train_subjects.append(str(PROC_DIR / line.split()[1]))
                if len(train_subjects) == 1:
                    break  # just one subject for testing

    print(f"Testing dataset with subject: {train_subjects[0]}")
    dataset = HCPSliceDataset(train_subjects, shell='b1000',
                               scale_r=3.0, n_slices=72)

    print(f"Dataset size: {len(dataset)} samples")
    assert len(dataset) == 72, f"Expected 72 slices, got {len(dataset)}"

    # Load one sample
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  I_in:     {sample['I_in'].shape}")     # (10, 145, 174)
    print(f"  I_target: {sample['I_target'].shape}") # (1, 145, 174)
    print(f"  q_star:   {sample['q_star'].shape}")   # (3,)
    print(f"  q_in:     {sample['q_in'].shape}")     # (10, 3)

    # Verify value ranges (should be 0-1 after normalization)
    print(f"\nValue ranges:")
    print(f"  I_in    min/max: {sample['I_in'].min():.4f} / "
          f"{sample['I_in'].max():.4f}")
    print(f"  I_target min/max: {sample['I_target'].min():.4f} / "
          f"{sample['I_target'].max():.4f}")

    # Verify N_in = 10
    N_in = sample['I_in'].shape[0]
    assert N_in == 10, f"Expected 10 input directions, got {N_in}"

    print("\nDataset PASSED")
