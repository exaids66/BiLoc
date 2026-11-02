# BiLoc 1-bit bit-packed inference runtimes for the Oxford model (prebuilt binaries, see README.md)
import sys

VARIANTS = ('fused', 'eqk')
TORCH_VERSION = '2.8.0+cu128'
GPU_ARCH = (12, 0)


def _check_environment(device):
    import torch
    if sys.version_info[:2] != (3, 9):
        raise RuntimeError(f'the prebuilt modules need Python 3.9, found {sys.version.split()[0]}')
    if torch.__version__ != TORCH_VERSION:
        raise RuntimeError(f'the prebuilt modules need torch {TORCH_VERSION}, found {torch.__version__}')
    if not torch.cuda.is_available():
        raise RuntimeError('the 1-bit runtimes need a CUDA GPU')
    cap = torch.cuda.get_device_capability(torch.device(device))
    if tuple(cap) != GPU_ARCH:
        raise RuntimeError(f'the prebuilt kernels are compiled for sm_120 only, found sm_{cap[0]}{cap[1]}')


def build_runtime(variant='fused', device='cuda', prepare=(1, 50)):
    # 'fused': fused kernels (encoder ~50 launches, the whole DDIM loop in one launch)
    # 'eqk': one kernel per operator group (kernel count close to the real-valued DiffLoc)
    if variant not in VARIANTS:
        raise ValueError(f'unknown variant {variant!r}; choose one of {VARIANTS}')
    _check_environment(device)
    if variant == 'fused':
        from ._runtime import build_fused_runtime
        return build_fused_runtime(device=device, prepare=prepare)
    from ._runtime import build_runtime as build_eqk_runtime
    return build_eqk_runtime(device=device, prepare=prepare)


__all__ = ['build_runtime', 'VARIANTS']
