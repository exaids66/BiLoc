# Latency benchmark of the BiLoc 1-bit runtimes (and, optionally, the real-valued DiffLoc baseline).
#
#   cd biloc
#   python -m deploy.benchmark                                  # fused + eqk, B=1, random input
#   python -m deploy.benchmark --dataroot ../data --with-diffloc --json latency.json
#
# Protocol: torch.inference_mode, TF32 off, cudnn.benchmark on, 1 CPU thread. Per (stage, mode) unit and round:
# 15 warm-up calls, synchronize, 30 timed calls bracketed by perf_counter and CUDA events (-> round mean), then 30
# calls each synchronized before and after (-> per-call samples). 5 rounds; the order of the units is shuffled every
# round. Reported: median of the round means (wall and CUDA events) and p50 / p90 of the per-call samples.
# Graph mode captures ONE call into a CUDA graph (static inputs) and times its replay.
# infer = encode + initial noise (torch.randn, drawn inside the timed call) + 10-step DDIM.
import argparse
import contextlib
import importlib.util
import io
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

FP_ROOT = ROOT.parent / 'diffoc_full_precision'
SEED = 20260928
TEST_SEQS = ['2019-01-15-13-06-37', '2019-01-17-13-26-39', '2019-01-17-14-03-00', '2019-01-18-14-14-42']
VARIANTS = ['fused', 'eqk', 'diffloc_fp32', 'diffloc_fp16']
STAGES = ['encode', 'ddim', 'infer']


# ------------------------------------------------------------------------------------------ real-valued DiffLoc
def load_fp_models():
    # the DiffLoc package is also called `models`: import it as `fp_models` (as train_kd.py does)
    name = 'fp_models'
    if name in sys.modules:
        return sys.modules[name]
    path = FP_ROOT / 'models'
    spec = importlib.util.spec_from_file_location(name, str(path / '__init__.py'))
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    return module


def _replace_model_targets(node):
    from omegaconf import DictConfig, ListConfig
    if isinstance(node, (DictConfig, dict)):
        for key in list(node.keys()):
            value = node[key]
            if isinstance(value, str) and value.startswith('models'):
                node[key] = 'fp_models' + value[len('models'):]
            else:
                _replace_model_targets(value)
    elif isinstance(node, (ListConfig, list, tuple)):
        for value in node:
            _replace_model_targets(value)


def _temb_forward_syncfree(self, timesteps):
    # TimeStepEmbedding.forward with the CPU-computed frequencies copied to the GPU once (no per-call host copy)
    args = timesteps[:, None].float() * self._sf_freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if self.dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return self.linear(embedding)


def build_diffloc(ckpt=None, device='cuda'):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import models  # noqa: F401  (the DiffLoc package resolves its absolute `models.*` imports here)
    load_fp_models()
    cfg = OmegaConf.load(str(FP_ROOT / 'cfgs' / 'oxford.yaml'))
    OmegaConf.set_struct(cfg, False)
    _replace_model_targets(cfg.MODEL)
    with contextlib.redirect_stdout(io.StringIO()):
        model = instantiate(cfg.MODEL, _recursive_=False)
    if ckpt is not None:
        sd = torch.load(str(ckpt), map_location='cpu')
        sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
        model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    emb = sys.modules['utils.embedding']
    n = 0
    for m in model.modules():
        if isinstance(m, emb.TimeStepEmbedding):
            m._sf_freqs = m._compute_freqs(m.dim // 2).to(device=device)
            m.forward = types.MethodType(_temb_forward_syncfree, m)
            n += 1
    assert n == 1, n
    return cfg, model


class DiffLocRuntime:
    # real-valued DiffLoc: encode / ddim / infer with the op sequence of GaussianDiffusion.ddim_sample
    graphable = True

    def __init__(self, cfg, model, precision='fp32'):
        assert precision in ('fp32', 'fp16')
        self.name = f'diffloc_{precision}'
        self.precision = precision
        self.model = model
        self.diff = model.diffuser
        self.extractor = model.image_feature_extractor
        self.device = self.diff.betas.device
        self.target_dim = model.target_dim
        steps = int(cfg.sampling_timesteps)
        times = torch.linspace(-1, self.diff.num_timesteps - 1, steps=steps + 1)
        times = list(reversed(times.int().tolist()))
        self.time_pairs = list(zip(times[:-1], times[1:]))

    def prepare(self, B):
        pass

    def _amp(self):
        if self.precision == 'fp16':
            return torch.autocast('cuda', dtype=torch.float16)
        return contextlib.nullcontext()

    @torch.no_grad()
    def encode(self, image):
        shape = list(image.shape)
        with self._amp():
            z, feats = self.extractor(image.reshape(shape[0] * shape[1], *shape[2:]))
        return z.float().reshape(shape[0], shape[1], -1)

    @torch.no_grad()
    def ddim(self, z, noise, calls=10):
        diff = self.diff
        img = noise
        x_start = None
        for k, (time_, time_next) in enumerate(self.time_pairs):
            if k >= calls:
                img = x_start
                break
            time_cond = torch.full((z.shape[0],), time_, device=self.device, dtype=torch.long)
            with self._amp():
                pred = diff.model_predictions(img, time_cond, z)
            pred_noise = pred.pred_noise.float()
            if time_next < 0:
                img = x_start
                continue
            alpha = diff.alphas_cumprod[time_]
            alpha_next = diff.alphas_cumprod[time_next]
            eta = 0
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise_k = torch.randn_like(img)
            x_start = (img - torch.sqrt((1. - alpha)) * pred_noise) / torch.sqrt(alpha)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise_k
        return img

    @torch.no_grad()
    def infer(self, image, noise=None, calls=10):
        z = self.encode(image)
        if noise is None:
            noise = torch.randn([z.shape[0], z.shape[1], self.target_dim], device=self.device)
        return self.ddim(z, noise, calls)


# ------------------------------------------------------------------------------------------ inputs
def real_clips(dataroot):
    # one interior clip (the middle one) of every Oxford test sequence, preprocessed by the release data loader
    from .evaluate import load_cfg, oxford_dir, sequence_dataset
    oxford = oxford_dir(dataroot)
    cfg = load_cfg()
    clips = []
    tmp = tempfile.mkdtemp(prefix='biloc_bench_')
    try:
        for seq in TEST_SEQS:
            ds = sequence_dataset(cfg, oxford, seq, tmp)
            clips.append(ds[len(ds) // 2]['image'])
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return torch.stack(clips)


def make_inputs(args):
    if args.dataroot:
        clips = real_clips(args.dataroot)
        idx = torch.arange(args.batch) % clips.shape[0]
        image = clips[idx].contiguous()
        desc = (f'real Oxford clips (release data loader): the middle clip of each test sequence, '
                f'cycled to B={args.batch}' + (f' (B=1: {TEST_SEQS[0]})' if args.batch == 1 else ''))
    else:
        g = torch.Generator().manual_seed(SEED)
        image = torch.randn([args.batch, 3, 5, 32, 512], generator=g)
        desc = f'RANDOM input (fixed seed, standard normal, i.e. normalized range-image scale), B={args.batch}; ' \
               'pass --dataroot for real Oxford clips'
    g = torch.Generator().manual_seed(SEED + 1)
    noise = torch.randn([args.batch, 3, 6], generator=g)
    return image.cuda(), noise.cuda(), desc


# ------------------------------------------------------------------------------------------ timing
def time_calls(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    begin.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return 1000 * (time.perf_counter() - t0) / iters, begin.elapsed_time(end) / iters


def time_percall(fn, iters):
    walls, events = [], []
    for _ in range(iters):
        b, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        b.record()
        fn()
        e.record()
        e.synchronize()
        walls.append(1000 * (time.perf_counter() - t0))
        events.append((b, e))
    return walls, [b.elapsed_time(e) for b, e in events]


def capture_graph(fn, warmup=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def stage_fn(rt, stage, inp):
    if stage == 'encode':
        return lambda: rt.encode(inp['image'])
    if stage == 'ddim':
        return lambda: rt.ddim(inp['z'], inp['noise'], 10)
    if stage == 'infer':
        return lambda: rt.infer(inp['image'])
    raise ValueError(stage)


def pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def gpu_processes():
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=20).stdout
        return [x.strip() for x in out.splitlines() if x.strip() and x.split(',')[0].strip() != str(os.getpid())]
    except Exception:
        return None


# ------------------------------------------------------------------------------------------ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='Latency of the BiLoc 1-bit runtimes (optionally vs real-valued DiffLoc)')
    ap.add_argument('--variants', default='fused,eqk', help=f'comma list from {VARIANTS}')
    ap.add_argument('--with-diffloc', action='store_true', help='add diffloc_fp32 and diffloc_fp16 (FP16 autocast)')
    ap.add_argument('--diffloc-ckpt', type=Path, default=None,
                    help='DiffLoc Oxford checkpoint (default: random weights; latency does not depend on them)')
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--stages', default='infer', help=f'comma list from {STAGES}')
    ap.add_argument('--modes', default='eager,graph')
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--warmup', type=int, default=15)
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--dataroot', default=None, help='directory that contains Oxford/ (real input clips)')
    ap.add_argument('--json', type=Path, default=None, help='also write all results to this JSON file')
    args = ap.parse_args(argv)
    uniq = lambda xs: list(dict.fromkeys(x for x in xs if x))         # noqa: E731
    args.variants = uniq(args.variants.split(','))
    if args.with_diffloc:
        args.variants = uniq(args.variants + ['diffloc_fp32', 'diffloc_fp16'])
    args.stages = uniq(args.stages.split(','))
    args.modes = uniq(args.modes.split(','))
    bad = [v for v in args.variants if v not in VARIANTS] + [s for s in args.stages if s not in STAGES]
    bad += [m for m in args.modes if m not in ('eager', 'graph')]
    if bad:
        ap.error(f'unknown variant/stage/mode: {bad}')
    if not args.variants or not args.stages or not args.modes:
        ap.error('empty --variants / --stages / --modes')
    if args.batch < 1 or args.rounds < 1 or args.iters < 1 or args.warmup < 0:
        ap.error('--batch, --rounds, --iters must be >= 1 and --warmup >= 0')
    return args


def build(name, args, cache):
    if name in ('fused', 'eqk'):
        import deploy
        return deploy.build_runtime(name, prepare=(args.batch,))
    if 'diffloc' not in cache:
        cache['diffloc'] = build_diffloc(args.diffloc_ckpt)
    cfg, model = cache['diffloc']
    return DiffLocRuntime(cfg, model, name.split('_')[1])


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(SEED)
    image, noise, input_desc = make_inputs(args)
    others = gpu_processes()
    env = dict(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
               torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
               python=platform.python_version(), omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
               other_gpu_processes=others)
    print(f'GPU {env["gpu"]} (sm_{"".join(map(str, env["capability"]))}), torch {env["torch"]}, CUDA {env["cuda"]}, '
          f'cuDNN {env["cudnn"]}, Python {env["python"]}')
    print(f'input: {input_desc}')
    if others:
        print(f'WARNING: other processes use the GPU ({len(others)}); numbers will be noisy')
    runtimes, build_s = {}, {}
    cache = {}
    for name in args.variants:
        t0 = time.time()
        runtimes[name] = build(name, args, cache)
        build_s[name] = round(time.time() - t0, 1)
        print(f'built {name} ({build_s[name]} s)', flush=True)
    if any(v.startswith('diffloc') for v in runtimes):
        print('DiffLoc weights: ' + (str(args.diffloc_ckpt) if args.diffloc_ckpt else 'random (latency only)'))

    order_rng = random.Random(SEED)
    results = {}
    for stage in args.stages:
        inputs = {}
        for name, rt in runtimes.items():
            rt.prepare(args.batch)
            inp = dict(image=image, noise=noise)
            if stage == 'ddim':
                inp['z'] = rt.encode(image).clone()
            inputs[name] = inp
        torch.cuda.synchronize()
        units = [(n, m) for n in runtimes for m in args.modes]
        graphs = {}
        rec = {u: dict(wall=[], cuda=[], pc_wall=[], pc_cuda=[]) for u in units}
        for r in range(args.rounds):
            order = list(units)
            order_rng.shuffle(order)
            for name, mode in order:
                fn = stage_fn(runtimes[name], stage, inputs[name])
                if mode == 'graph':
                    if name not in graphs:
                        graphs[name] = capture_graph(fn)[0]
                    fn = graphs[name].replay
                w, c = time_calls(fn, args.warmup, args.iters)
                pw, pc = time_percall(fn, args.iters)
                d = rec[(name, mode)]
                d['wall'].append(w)
                d['cuda'].append(c)
                d['pc_wall'] += pw
                d['pc_cuda'] += pc
            print(f'{stage}: round {r + 1}/{args.rounds} done', flush=True)
        del graphs
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        for (name, mode), d in rec.items():
            results.setdefault(name, {}).setdefault(stage, {})[mode] = dict(
                wall_ms=statistics.median(d['wall']), cuda_ms=statistics.median(d['cuda']),
                p50_wall_ms=pct(d['pc_wall'], 50), p90_wall_ms=pct(d['pc_wall'], 90),
                p50_cuda_ms=pct(d['pc_cuda'], 50), p90_cuda_ms=pct(d['pc_cuda'], 90),
                round_means_wall_ms=d['wall'], round_means_cuda_ms=d['cuda'])

    print()
    print(f'B={args.batch}; {args.rounds} round(s) x ({args.warmup} warm-up + {args.iters} timed calls); '
          f'median of round means; per-call p50 / p90 (ms)')
    hdr = f'{"variant":14s} {"stage":7s} {"mode":6s} {"wall":>9s} {"cuda":>9s} {"p50 wall":>9s} {"p90 wall":>9s} ' \
          f'{"p50 cuda":>9s} {"p90 cuda":>9s}'
    print(hdr)
    print('-' * len(hdr))
    for name in runtimes:
        for stage in args.stages:
            for mode in args.modes:
                x = results[name][stage][mode]
                print(f'{name:14s} {stage:7s} {mode:6s} {x["wall_ms"]:9.3f} {x["cuda_ms"]:9.3f} {x["p50_wall_ms"]:9.3f} '
                      f'{x["p90_wall_ms"]:9.3f} {x["p50_cuda_ms"]:9.3f} {x["p90_cuda_ms"]:9.3f}')
    if args.json is not None:
        report = dict(environment=env, input=input_desc, batch=args.batch, variants=args.variants, stages=args.stages,
                      modes=args.modes, build_s=build_s,
                      protocol=dict(rounds=args.rounds, warmup=args.warmup, iters=args.iters, order_seed=SEED,
                                    inference_mode=True, tf32=False, cudnn_benchmark=True, torch_threads=1,
                                    graph='one captured call per (variant, stage), replayed',
                                    reported='median of round means (wall perf_counter, CUDA events); '
                                             'per-call p50 / p90 over all rounds'),
                      diffloc_weights=(str(args.diffloc_ckpt) if args.diffloc_ckpt else 'random')
                      if any(v.startswith('diffloc') for v in runtimes) else None,
                      results=results)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1) + '\n')
        print(f'wrote {args.json}')
    return results


if __name__ == '__main__':
    main()
