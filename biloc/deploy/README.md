# BiLoc 1-bit GPU runtimes (Oxford)

Prebuilt 1-bit GPU runtimes for the BiLoc Oxford model, with the bit-packed Oxford weights. Two variants are
available: `fused` and `eqk`.

## Requirements

- Linux x86_64, Python 3.9
- PyTorch 2.8.0+cu128
- An NVIDIA GPU of compute capability 12.0 (sm_120)
- The BiLoc Python environment (the same as for `test.py`)

## Usage

Run from the `biloc/` directory:

```python
import deploy

rt = deploy.build_runtime('fused')   # or 'eqk'
pose = rt.infer(image)               # image: [B, 3, 5, 32, 512] float32 CUDA tensor -> pose: [B, 3, 6]
```

`image` is a batch of normalized range images as produced by the BiLoc data loader; `pose` is the normalized pose
of every frame, the same as `pred_pose` of the PyTorch model.

## Evaluation

```bash
cd biloc
python -m deploy.evaluate --dataroot ../data --variant fused   # or --variant eqk
```

`--dataroot` is the directory that contains `Oxford/`.

## Latency

```bash
cd biloc
python -m deploy.benchmark --dataroot ../data                  # fused and eqk
python -m deploy.benchmark --dataroot ../data --with-diffloc   # also the real-valued DiffLoc
```

Run either script with `--help` for all options.
