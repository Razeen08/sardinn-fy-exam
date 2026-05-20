# SARDInn IRR Module Ablation Study
**BME First Year Qualifying Exam — University of Rochester, 2026**

Independent reimplementation of SARDInn (Wang et al., *Medical Physics* 2025) from scratch in PyTorch, with a systematic ablation study of the Implicit Representation and Reconstruction (IRR) module. This repository contains all code, SLURM scripts, and results associated with the engineering analysis submitted as part of the BME first year qualifying exam.

---

## Paper Reference

Wang S, Wang L, Cao Y, Deng Z, Ye C, Wang R, Zhu Y, Wei H. Self-supervised arbitrary-scale super-angular resolution diffusion MRI reconstruction. *Medical Physics*. 2025;52(5):2976-2998.

---

## Project Structure

```
sardinn-fy-exam/
├── code/
│   ├── model/
│   │   ├── sardinn.py                      # Full SARDInn model (TSC + DWFE + IRR)
│   │   ├── dwfe.py                         # Diffusion Weighted Feature Extraction module
│   │   └── irr.py                          # Implicit Representation and Reconstruction module
│   ├── training/
│   │   └── train.py                        # Training loop with validation PSNR tracking
│   ├── run_training.py                     # Entry point for training a single variant
│   ├── run_inference.py                    # Entry point for baseline inference
│   └── run_inference_ablation.py           # Entry point for ablation inference (all 9 variants)
├── slurm_scripts/
│   ├── ablation_depth_4.slurm
│   ├── ablation_depth_6.slurm
│   ├── ablation_depth_8_verify.slurm       # Paper default — reference variant
│   ├── ablation_depth_10.slurm
│   ├── ablation_depth_12.slurm
│   ├── ablation_sine_none.slurm
│   ├── ablation_sine_L2.slurm
│   ├── ablation_sine_L6.slurm
│   └── ablation_sine_all.slurm
├── results/
│   ├── ablation_results_all.csv            # Aggregated metrics for all 9 variants
│   ├── ablation_depth_4/results.json
│   ├── ablation_depth_6/results.json
│   ├── ablation_depth_8_verify/results.json
│   ├── ablation_depth_10/results.json
│   ├── ablation_depth_12/results.json
│   ├── ablation_sine_none/results.json
│   ├── ablation_sine_L2/results.json
│   ├── ablation_sine_L6/results.json
│   └── ablation_sine_all/results.json
├── preprocess.py                           # HCP data preprocessing pipeline
├── preprocess.slurm                        # SLURM script for preprocessing
├── compile_results.py                      # Aggregates per-variant JSON results into CSV
├── train_baseline_h100.slurm              # Baseline training on H100 GPU
├── train_baseline_l40s.slurm              # Baseline training on L40S GPU
├── run_inference.slurm                    # Baseline inference SLURM script
├── run_ablation_inference.slurm           # Ablation inference SLURM script
└── subjects.txt                           # Train/test subject split (20 train, 7 test)
```

---

## Environment Setup

Tested on BlueHive3 cluster (University of Rochester CIRC) with NVIDIA H100 NVL GPU.

```bash
# Load required modules
module load git
module load cuda/11.8

# Create and activate conda environment
conda create -n sardinn python=3.10
conda activate sardinn

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy nibabel scikit-image tqdm pandas
```

---

## Data

This project uses the **Human Connectome Project (HCP) Young Adult** dataset.

- **Access:** Requires registration at https://db.humanconnectome.org
- **Required files:** Preprocessed diffusion data (`data.nii.gz`, `bvals`, `bvecs`) per subject
- **Shells used:** b=1000 s/mm2 (primary), b=2000, b=3000 for evaluation
- **Directions per shell:** 90
- **Subjects:** 20 for training, 7 for held-out test (see `subjects.txt`)

After downloading, organize data as:

```
data/
└── {subject_id}/
    └── T1w/Diffusion/
        ├── data.nii.gz
        ├── bvals
        └── bvecs
```

Then run preprocessing:

```bash
sbatch preprocess.slurm
# or locally:
python preprocess.py
```

---

## Training

Each ablation variant is trained independently. To train the paper default (reference variant):

```bash
sbatch slurm_scripts/ablation_depth_8_verify.slurm
```

To train all 9 variants:

```bash
for script in slurm_scripts/ablation_*.slurm; do
    sbatch $script
done
```

**Training configuration (all variants):**

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-5 |
| Batch size | 10 |
| Loss | L1 |
| Epochs | 80 |
| Shell | b=1000 s/mm2 |
| Slices per subject | 72 (middle axial) |
| Upsampling scale | r=3 |

---

## Ablation Design

### Ablation A — MLP Depth
MLP depth varied with Sine activation at the midpoint layer in each case:

| Variant | Layers | Sine Layer |
|---|---|---|
| ablation_depth_4 | 4 | L2 |
| ablation_depth_6 | 6 | L3 |
| **ablation_depth_8_verify (paper default)** | **8** | **L4** |
| ablation_depth_10 | 10 | L5 |
| ablation_depth_12 | 12 | L6 |

### Ablation B — Sine Activation Position
MLP depth fixed at 8 layers, Sine position varied:

| Variant | Sine Layer |
|---|---|
| ablation_sine_none | None (ReLU only) |
| ablation_sine_L2 | L2 (early) |
| **ablation_depth_8_verify (paper default)** | **L4 (middle)** |
| ablation_sine_L6 | L6 (late) |
| ablation_sine_all | All layers (full SIREN) |

---

## Inference

After training, run inference on all 9 variants:

```bash
sbatch run_ablation_inference.slurm
# or locally:
python code/run_inference_ablation.py
```

**Inference configuration:**
- 7 held-out test subjects
- Scale r=3, b=1000 s/mm2
- 30 FPS-selected input directions
- 60 target directions reconstructed
- Metrics: PSNR, SSIM, RMSE per direction per slice, averaged across subjects

---

## Aggregating Results

```bash
python compile_results.py
```

Output: `results/ablation_results_all.csv`

---

## Key Results

All 9 variants evaluated at r=3, b=1000 s/mm2, 7 test subjects:

| Variant | Layers | Sine | PSNR (dB) | SSIM | RMSE |
|---|---|---|---|---|---|
| depth_4 | 4 | L2 | 28.20 +/- 0.61 | 0.8705 +/- 0.0126 | 0.0527 +/- 0.0047 |
| depth_6 | 6 | L3 | 28.31 +/- 0.64 | 0.8761 +/- 0.0136 | 0.0527 +/- 0.0048 |
| **depth_8 (paper)** | **8** | **L4** | **28.21 +/- 0.65** | **0.8774 +/- 0.0137** | **0.0531 +/- 0.0050** |
| depth_10 | 10 | L5 | 28.21 +/- 0.66 | 0.8723 +/- 0.0133 | 0.0535 +/- 0.0050 |
| depth_12 | 12 | L6 | 27.95 +/- 0.69 | 0.8755 +/- 0.0139 | 0.0548 +/- 0.0053 |
| sine_none | 8 | None | 28.15 +/- 0.62 | 0.8760 +/- 0.0137 | 0.0530 +/- 0.0048 |
| sine_L2 | 8 | L2 | 28.09 +/- 0.57 | 0.8770 +/- 0.0132 | 0.0524 +/- 0.0045 |
| sine_L6 | 8 | L6 | 28.28 +/- 0.63 | 0.8770 +/- 0.0137 | 0.0527 +/- 0.0048 |
| sine_all (SIREN) | 8 | All | 28.29 +/- 0.64 | 0.8774 +/- 0.0135 | 0.0526 +/- 0.0048 |

**Finding:** All variants within 0.36 dB PSNR. Neither MLP depth nor Sine activation position constitutes a critical design parameter in the IRR module.

---

## Important Notes

- This is an independent reimplementation from the paper description. The authors' original code is not publicly available.
- Absolute PSNR values (~28 dB) are lower than the paper's reported values (~38 dB) due to differences in training subject selection, FPS seed, inference direction assignment, and training duration. Relative differences between variants are valid.
- All variants were trained for 80 epochs under identical conditions to ensure fair comparison.

---

## Citation

```bibtex
@article{Wang2025,
  author  = {Wang, Shuangxing and Wang, Lihui and Cao, Ying and
             Deng, Zeyu and Ye, Chen and Wang, Rongpin and
             Zhu, Yuemin and Wei, Hongjiang},
  title   = {Self-supervised arbitrary-scale super-angular resolution
             diffusion {MRI} reconstruction},
  journal = {Medical Physics},
  volume  = {52},
  number  = {5},
  pages   = {2976--2998},
  year    = {2025}
}
```

---

## Author

Raiyun Kabir
PhD Student, Department of Biomedical Engineering
University of Rochester
rkabir5@ur.rochester.edu
