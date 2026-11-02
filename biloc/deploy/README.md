# BiLoc 1-bit deployment runtimes (binary release)

This directory contains two GPU inference runtimes for the BiLoc **Oxford** model in which the binary layers run
as real 1-bit arithmetic: weights and activations are bit-packed into 32-bit words and every binary layer is an
XOR + popcount product on CUDA cores. The float parts of the network (LayerNorm, RPReLU, attention softmax,
patch merges, the DDIM update, ...) run in strict FP32.

The runtimes are shipped as prebuilt modules: a small Python entry point (`__init__.py`), one compiled runtime module
and one compiled CUDA extension, plus the bit-packed Oxford weights. The C++/CUDA/Python sources of the runtimes are
not part of this release. `benchmark.py` and `evaluate.py` are plain Python.

| Variant | Kernels per inference (B=1) | Description |
|---|---:|---|
| `fused` | 52 | the encoder runs in 50 kernels (fused per layer); the denoiser and the whole 10-step DDIM loop run in one kernel launch |
| `eqk` | 1,564 | one kernel per operator group (binarizer, binary GEMM / conv, normalization group, attention core, ...); its kernel count is close to that of the real-valued DiffLoc (about 1,570) |

Both variants compute the same binary arithmetic and load the same weights.

## Requirements

- Linux x86_64, Python 3.9
- PyTorch 2.8.0+cu128 (the pip wheel; it brings the CUDA 12.8 runtime). No CUDA toolkit or compiler is needed.
- An NVIDIA GPU of compute capability 12.0 (sm_120) with a driver that supports CUDA 12.8. The binaries contain
  sm_120 code only. They were built and tested on an RTX PRO 6000 Blackwell Workstation Edition.
- The BiLoc Python dependencies (the same environment as `test.py`: `hydra-core`, `omegaconf`, `timm`,
  `transformers`, `pytorch3d`, `h5py`, ...). The loader builds the model architecture from `../cfgs/oxford.yaml`
  and the code in `../models`, `../utils` and `../../BHViT`.

`build_runtime()` checks the Python version, the torch version and the GPU architecture and raises a clear error
if they do not match.

## Files

| File | Content |
|---|---|
| `__init__.py` | entry point: `build_runtime(variant='fused', device='cuda', prepare=(1, 50))` |
| `_runtime*.so` | the `fused` and `eqk` runtimes; paths, model construction, weight loading and checks |
| `_cuda*.so` | CUDA kernels of both runtimes (one PyTorch extension) |
| `weights/biloc_oxford_bitpacked.pth` | bit-packed Oxford weights (13,303,938 bytes, SHA-256 `30fcf216fa1e5948cee4f26bf8ec982f65b8bcafe48f62e830b201ec7d458670`) |
| `benchmark.py` | latency benchmark |
| `evaluate.py` | Oxford accuracy evaluation |

The weights file holds the sign bits of all 157 binary layers (packed 32 per word), their scales and biases, and
the floating-point parameters of the rest of the network. The latent floating-point weights of the binary layers
are not included. It was exported from the released BiLoc Oxford checkpoint (SHA-256
`4acf6c439993b9694d38bf39ee943762b1d2c076b20779d3aaaa18490cfb5fb1`). When loading, the runtime checks the SHA-256
of the file, the SHA-256 of each of the 676 tensors of the binary layers and one SHA-256 over all 1,457
floating-point tensors.

## Usage

Run from the `biloc/` directory, so that `deploy`, `models` and `cfgs` can be found:

```python
import torch
import deploy

rt = deploy.build_runtime('fused')        # or 'eqk'; prepares B=1 and B=50
pose = rt.infer(image)                    # image: [B, 3, 5, 32, 512] float32 CUDA -> pose [B, 3, 6]

z = rt.encode(image)                      # [B, 3, 384]
pose = rt.ddim(z, noise, calls=10)        # noise: [B, 3, 6] initial DDIM sample
pose = rt.infer(image, noise)             # = rt.ddim(rt.encode(image), noise)
```

- Inputs are normalized range images as produced by the BiLoc data loader (3 frames per clip); the output is the
  normalized pose of every frame (translation scaled by `Oxford_pose_stats.txt`, rotation as log-quaternion), the
  same as `pred_pose` of the PyTorch model.
- `infer(image)` without `noise` draws one `torch.randn([B, 3, 6])`. The model's 9 further random draws are
  multiplied by sigma = 0 and are skipped, so the CUDA random stream afterwards differs from the PyTorch model.
- `calls=9` skips the 10th denoiser call, whose result is not used; the output is identical to `calls=10`.
- `rt.prepare(B)` does the one-time setup for a batch size (synchronizes with the host). `build_runtime` calls it
  for every batch size in `prepare` (default `(1, 50)`; `prepare=()` for none). Other batch sizes are prepared on
  their first call. After `prepare(B)`, `encode`, `ddim` and `infer` do not synchronize with the host
  and can be captured in a CUDA graph (call `prepare(B)` before capturing).
- One runtime instance must not be used on two CUDA streams at the same time (shared workspaces).

## Latency benchmark

```bash
cd biloc
python -m deploy.benchmark                                   # fused and eqk, B=1, random input
python -m deploy.benchmark --dataroot ../data                # real Oxford clips (one per test sequence)
python -m deploy.benchmark --with-diffloc --batch 1 --json latency_b1.json
```

Protocol: `torch.inference_mode`, TF32 off, `cudnn.benchmark` on, 1 CPU thread. For every (variant, mode) pair
(modes: eager, CUDA graph) and in each of 5 rounds: 15 warm-up calls, then 30 timed calls (wall clock and CUDA
events), then 30 calls each synchronized before and after for per-call percentiles. The order of the
(variant, mode) pairs is shuffled every round (fixed seed). The table shows the median of the round means and the
per-call p50 / p90 in ms. In graph mode one call is captured and replayed. The default stage is `infer`;
`--stages encode,ddim,infer` measures the stages one after another, each with the full protocol. `--with-diffloc`
adds the real-valued DiffLoc from `../../diffoc_full_precision` in FP32 and with FP16 autocast; it uses random
weights unless `--diffloc-ckpt` is given (latency does not depend on the weight values). Other options:
`--variants`, `--modes`, `--rounds`, `--warmup`, `--iters`. Close other GPU work before measuring.

## Oxford evaluation

```bash
cd biloc
python -m deploy.evaluate --dataroot ../data --variant fused       # all 4 test sequences
python -m deploy.evaluate --dataroot ../data --variant eqk --seqs 2019-01-18-14-14-42 --max-batches 3
```

`--dataroot` is the directory that contains `Oxford/` (as `train.dataroot` in `../cfgs/oxford.yaml`). The protocol
is that of Table 2 in the paper: each test sequence is evaluated separately (through a temporary data root with
its own `valid_split.txt`; the dataset is not modified), batch size 50, seed 7 per sequence, and per batch one
`torch.randn` initial noise followed by 9 discarded draws, i.e. the random stream of the PyTorch model. The metric
is the mean over all frames for 2019-01-15-13-06-37 and over the last frame of each clip for the other three
sequences; the average is the mean of the four. Options: `--json`, `--save-per-frame`, `--bs`, `--workers`.
The data loader needs `velodyne_left_False.h5` in every sequence directory (created by the training/test code).

## Measured results

Measured on 2026-11-02 with the files in this directory:

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (sm_120, 188 SMs), driver 570.181, no other compute
  process on the GPU. CPU: Intel Core i7-14700KF. PyTorch 2.8.0+cu128, CUDA 12.8, cuDNN 9.10.2, Python 3.9.25.
- Latency: `benchmark.py` with the default protocol described above (TF32 off) and real Oxford input (B=1: the middle
  clip of 2019-01-15-13-06-37), run from `biloc/`:

  ```bash
  OMP_NUM_THREADS=2 python -m deploy.benchmark --dataroot ../data --batch 1 --stages encode,ddim,infer \
      --with-diffloc --diffloc-ckpt <DiffLoc Oxford checkpoint>
  ```

- `infer` is a complete inference: encoder, initial noise and the 10-step DDIM loop. `diffloc_fp32` is the
  real-valued DiffLoc of `../../diffoc_full_precision` with its trained Oxford weights in strict FP32, `diffloc_fp16`
  the same model under FP16 autocast; both run as plain (uncompiled) PyTorch operators.

### Latency, B=1 (one clip per call)

`infer` in ms. Two runs of the same command, given as run 1 / run 2.

| Mode | Statistic | `fused` | `eqk` | `diffloc_fp32` | `diffloc_fp16` |
|---|---|---:|---:|---:|---:|
| CUDA graph | median of round means | 0.624 / 0.651 | 2.639 / 2.645 | 9.442 / 9.577 | 6.338 / 6.393 |
| CUDA graph | per-call p50 | 0.623 / 0.623 | 2.510 / 2.589 | 9.020 / 9.384 | 6.260 / 6.200 |
| eager | median of round means | 0.726 / 0.713 | 5.063 / 5.098 | 11.558 / 11.746 | 17.775 / 17.557 |
| eager | per-call p50 | 0.699 / 0.705 | 4.886 / 5.076 | 11.314 / 11.904 | 17.686 / 17.529 |

Speed-up over `diffloc_fp32` in the same mode and run, run 1 / run 2 (ratio of the medians of round means; ratio of
the per-call p50 in brackets):

| Mode | `fused` | `eqk` |
|---|---:|---:|
| CUDA graph | 15.14x / 14.71x [14.49x / 15.05x] | 3.58x / 3.62x [3.59x / 3.62x] |
| eager | 15.92x / 16.48x [16.19x / 16.89x] | 2.28x / 2.30x [2.32x / 2.35x] |

Stages, run 1 (median of round means, eager / CUDA graph, ms):

| Stage | `fused` | `eqk` | `diffloc_fp32` |
|---|---:|---:|---:|
| `encode` | 0.342 / 0.264 | 1.339 / 0.748 | 6.497 / 6.016 |
| `ddim` (10 denoiser calls) | 0.370 / 0.359 | 3.664 / 1.825 | 8.042 / 3.071 |

- Eager mode launches every kernel from the host (per B=1 inference: 52 for `fused`, 1,564 for `eqk`, about 1,570
  for `diffloc_fp32`), so eager latencies depend on the CPU. In CUDA graph mode one captured inference is replayed
  by a single host call.
- For B=1 `infer` the five round means of a run spread by up to 9.4%, and the median of round means is up to 5.2%
  above the per-call p50. For `eqk` in CUDA graph mode all ten round means (2.541-2.703 ms in run 1, 2.613-2.656 ms
  in run 2) were above the per-call p50. The CUDA-event round means are the same to within 0.006 ms, i.e. the extra
  time is spent on the GPU; the cause is unknown.
- A second session on the same day (same files and command, GPU again to itself, two B=1 runs) was faster in
  every variant, stage and mode: medians of round means 0.2-6.4% and per-call p50 0.05-4.6% lower than above (mean
  of the two runs of each session). CUDA graph `infer`: `fused` 0.612 / 0.615 ms, `eqk` 2.495 / 2.509 ms,
  `diffloc_fp32` 8.868 / 8.935 ms, i.e. speed-ups over `diffloc_fp32` of 14.49x / 14.54x (`fused`) and
  3.55x / 3.56x (`eqk`); 8 of the 10 `eqk` CUDA graph round means were within 0.4% of the per-call p50.
- `diffloc_fp16` (autocast) converts the weights to FP16 inside every call; in eager mode this makes it slower than
  `diffloc_fp32`.
- The speed-ups compare two different networks: DiffLoc's real-valued ViT-S/16 encoder (run as released, including
  its dense feature head, which the pose does not use) and BiLoc's binary BHViT encoder. Faster real-valued
  baselines exist (e.g. `torch.compile` with FP16 weights); they are not part of `benchmark.py`. BiLoc is also less
  accurate than DiffLoc on Oxford (paper, Table 2: 7.56 m / 1.91 deg vs 3.53 m / 0.72 deg).

### Latency, B=50 (one run)

Same command with `--batch 50` and without `--with-diffloc` (the DiffLoc baselines were not measured at B=50).
`infer` in ms, median of round means [per-call p50]:

| Mode | `fused` | `eqk` |
|---|---:|---:|
| CUDA graph | 2.576 [2.586] (51.5 us per clip) | 5.566 [5.541] (111.3 us per clip) |
| eager | 2.642 [2.654] | 7.762 [7.664] |

### Oxford accuracy

`python -m deploy.evaluate --dataroot ../data --variant fused` (and `--variant eqk`): all 159,701 clips of the 4
test sequences, protocol of Table 2 (see above). Mean translation error (m) / mean rotation error (deg):

| Sequence (frames) | `fused` | `eqk` | Paper, Table 2 |
|---|---:|---:|---:|
| 2019-01-15-13-06-37 (all) | 5.9535 / 2.0640 | 5.9523 / 2.0599 | 5.97 / 2.03 |
| 2019-01-17-13-26-39 (last) | 10.9608 / 2.1890 | 10.9383 / 2.1851 | 10.96 / 2.18 |
| 2019-01-17-14-03-00 (last) | 8.5037 / 2.0063 | 8.5085 / 1.9990 | 8.42 / 2.00 |
| 2019-01-18-14-14-42 (last) | 4.8605 / 1.4356 | 4.8593 / 1.4401 | 4.89 / 1.45 |
| **Average** | **7.5696 / 1.9237** | **7.5646 / 1.9210** | **7.56 / 1.91** |

- With the same protocol, the PyTorch model (released checkpoint) gives 7.5797 m / 1.9242 deg in strict FP32 and
  7.5617 m / 1.9177 deg with PyTorch's default TF32 convolutions (the setting of `test.py`).
- Every per-frame prediction of this release is bitwise identical to that of the source version of the runtimes.

## Numerics and limitations

- Oxford model only, 10 DDIM steps. NCLT is not included.
- The binaries are tied to this environment: Linux x86_64, CPython 3.9, PyTorch 2.8.0+cu128, sm_120 GPUs. Other
  versions will not load.
- Launch configurations were tuned on an RTX PRO 6000 Blackwell (188 SMs). Other sm_120 GPUs are untested. For
  B <= 15 the `fused` denoiser is a cooperative launch whose CTAs (at most one per SM) must all be resident at
  once: 24*B CTAs for B <= 7 (168 at B=7) and 12*B for B = 8 to 15 (180 at B=15). On GPUs with fewer SMs some of
  these batch sizes are rejected with an error; which ones depends on the SM count (below 24 SMs even B=1, which
  `build_runtime` prepares by default).
- Binary layers are bit-exact: `float(popcount dot) * scale + bias` with IEEE single-precision rounding and no
  fused multiply-add, and sign(0) = 0 as in the model. The float parts use a different summation order than
  PyTorch. About 60% of the Oxford test frames agree with the strict-FP32 PyTorch model to within 1 mm; when a
  value close to a binarization threshold flips sign, the difference propagates and the pose of that frame can
  differ by meters (rarely much more). On the full Oxford test set the average accuracy matches the PyTorch model (within
  0.02 m / 0.01 deg in our runs).
- `fused` computes the encoder's float GEMMs in an order that depends on the number of frames, so B=1 and B=50
  encoder outputs differ at the rounding level (and some poses more, through sign flips); for B >= 17 its stem convolution uses cuDNN, whose algorithm choice under
  `torch.backends.cudnn.benchmark = True` can change between processes. `eqk` gives the same result for every
  batch size.
