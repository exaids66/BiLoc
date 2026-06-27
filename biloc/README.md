# BiLoc Mainline

This directory contains the BiLoc training and evaluation code. It implements a
1-bit BHViT student model for DiffLoc-style outdoor LiDAR localization, plus KD
training with a full-precision DiffLoc teacher.

## Layout

- `train.py`: 1-bit student training without the auxiliary objective.
- `train_kd.py`: BiLoc training with the auxiliary objective.
- `test.py`: evaluation entry. It loads `cfg.ckpt` and writes metrics,
  trajectory plots, and prediction/error text files to `cfg.exp_dir`.
- `cfgs/oxford.yaml`: Oxford configuration and default released student
  checkpoint path.
- `cfgs/nclt.yaml`: NCLT configuration. NCLT checkpoints are not released by
  default; provide your own paths before evaluation.
- `models/`: DiffusionLoc model, BHViT feature extractor wrapper, binarized
  denoiser, Gaussian diffusion, and quantization layers.
- `loss/`: feature KD and structure KD losses used by BiLoc.
- `datasets/`: Oxford/NCLT loaders and range projection utilities.
- `utils/`: pose, embedding, training, and logging utilities.
- `preprocess/`: dataset preprocessing scripts.

## Environment

The experiments were run with the local conda environment `biloc` on a single
NVIDIA RTX 5090 GPU.

Verified local environment:

- Conda environment: `biloc`
- Python: 3.9.25
- PyTorch: 2.8.0+cu128
- CUDA runtime used by PyTorch: 12.8
- GPU: single NVIDIA RTX 5090

Activate the environment before running commands:

```bash
conda activate biloc
```

## Data

Both configs expect datasets under:

```text
../data
```

Expected pose stats are loaded as:

```text
<dataroot>/<dataset>/<dataset>_pose_stats.txt
```

For example:

```text
../data/Oxford/Oxford_pose_stats.txt
../data/NCLT/NCLT_pose_stats.txt
```

The dataloader returns multi-frame clips:

```text
image: [B, steps, 5, 32, 512]
pose:  [B, steps, 6]
mask:  [B, steps, 32, 512]
```

Current configs use `steps: 3` and `skip: 2`.

## Main Commands

Run from this directory:

```bash
cd biloc
```

Train the student without the auxiliary objective:

```bash
python train.py
```

Train BiLoc with the auxiliary objective:

```bash
python train_kd.py
```

Evaluate the checkpoint configured in the selected YAML:

```bash
python test.py
```

Important: the script entrypoints currently hard-code which config they load at
the bottom of each file. Change only that final `OmegaConf.load(...)` line or
make a copied config when running new experiments.

## Configuration Notes

The release configs use relative placeholder paths such as:

```text
../data
../checkpoints/biloc_oxford.pth
../checkpoints/your_diffloc_teacher_oxford.pth
```

Only the Oxford BiLoc student checkpoint is planned for the public release.
Teacher checkpoints and NCLT checkpoints are not released; their checkpoint
fields use `your_*` placeholders.

Before launching new experiments, verify these fields:

- `ckpt`
- `exp_dir`
- `train.dataroot`
- `KD.teacher_cfg`
- `KD.teacher_ckpt`

## Model Summary

The main model is `models.DiffusionLocModel`.

Training:

1. Input range images are reshaped from `[B, N, 5, H, W]` to `[B*N, 5, H, W]`.
2. `ImageFeatureExtractor(backbone="bhvit")` wraps the sibling `BHViT/`
   implementation and emits a 384-dimensional global feature.
3. The feature is reshaped to `[B, N, 384]`.
4. `GaussianDiffusion` trains a pose denoiser on normalized 6DoF poses.
5. `train_kd.py` additionally distills `z_out4distil` from the full-precision
   teacher.

Inference:

1. The feature extractor produces conditioning features.
2. `GaussianDiffusion.ddim_sample(...)` predicts poses with
   `sampling_timesteps` from the config.

## KD Losses

The default BiLoc configuration uses:

- `KD.loss_type: entropy`
- `KD.struct_loss_type: lckt`

Other loss implementations in `loss/` are kept only when required by the main
training code paths.
