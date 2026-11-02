# Oxford accuracy of a BiLoc 1-bit runtime with the paper's Table-2 protocol.
#
#   cd biloc
#   python -m deploy.evaluate --dataroot ../data --variant fused
#
# Protocol: every test sequence is evaluated on its own (a temporary data root with its own valid_split.txt and
# symlinks to the dataset; the dataset itself is never modified), batch size 50, seed 7 per sequence, and per batch
# the initial DDIM noise is one torch.randn followed by 9 discarded draws (the random stream of the PyTorch model).
# Metrics as in test.py; the paper metric is 'all' frames for 2019-01-15-13-06-37 and the 'last' frame of each clip
# for the other three sequences; the reported average is the mean of the four per-sequence paper metrics.
import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from datasets.composition import MF  # noqa: E402  (import the data code before any model is built)
from utils.pose_util import qexp_t  # noqa: E402
from utils.utils import seed_all_random_engines  # noqa: E402

TEST_SEQS = ['2019-01-15-13-06-37', '2019-01-17-13-26-39', '2019-01-17-14-03-00', '2019-01-18-14-14-42']
PAPER_PROTOCOL = {'2019-01-15-13-06-37': 'all', '2019-01-17-13-26-39': 'last',
                  '2019-01-17-14-03-00': 'last', '2019-01-18-14-14-42': 'last'}
PAPER_TABLE2 = {'2019-01-15-13-06-37': (5.97, 2.03), '2019-01-17-13-26-39': (10.96, 2.18),
                '2019-01-17-14-03-00': (8.42, 2.00), '2019-01-18-14-14-42': (4.89, 1.45), 'average': (7.56, 1.91)}
SEED = 7
CFG = ROOT / 'cfgs' / 'oxford.yaml'


def load_cfg():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(str(CFG))
    OmegaConf.set_struct(cfg, False)
    return cfg


def oxford_dir(dataroot):
    # accepts the directory that contains Oxford/ (cfg.train.dataroot) or the Oxford directory itself
    p = Path(dataroot).expanduser().resolve()
    for cand in (p / 'Oxford', p):
        if (cand / 'Oxford_pose_stats.txt').is_file():
            return cand
    raise FileNotFoundError(f'no Oxford/Oxford_pose_stats.txt under {p}')


def make_seq_root(oxford, seq, tmp):
    seq_dir = oxford / f'{seq}-radar-oxford-10k'
    if not (seq_dir / 'velodyne_left_False.h5').is_file():
        # the Oxford loader would otherwise write an interpolated pose file into the dataset
        raise FileNotFoundError(f'{seq_dir}/velodyne_left_False.h5 is missing (run the training/test code once to create it)')
    od = Path(tmp) / seq / 'Oxford'
    od.mkdir(parents=True, exist_ok=True)
    (od / 'valid_split.txt').write_text(seq + '\n')
    os.symlink(oxford / 'Oxford_pose_stats.txt', od / 'Oxford_pose_stats.txt')
    os.symlink(seq_dir, od / seq_dir.name)
    return str(Path(tmp) / seq)


def sequence_dataset(cfg, oxford, seq, tmp):
    cfg.train.dataroot = make_seq_root(oxford, seq, tmp)
    with contextlib.redirect_stdout(io.StringIO()):
        return MF(cfg.train.dataset, cfg, split='eval')


def frame_errors(pred, gt, pm, ps):
    et = torch.linalg.norm((gt[:, :3] * ps + pm) - (pred[:, :3] * ps + pm), dim=1)
    dot = torch.clamp(torch.sum(qexp_t(pred[:, 3:]) * qexp_t(gt[:, 3:]), dim=1).abs(), -1.0, 1.0)
    return et, torch.rad2deg(2 * torch.acos(dot))


def stats(et, eq):
    return dict(all_mean_t=float(et.mean()), all_mean_q=float(eq.mean()),
                all_median_t=float(np.median(et)), all_median_q=float(np.median(eq)),
                last_mean_t=float(et[:, -1].mean()), last_mean_q=float(eq[:, -1].mean()),
                last_median_t=float(np.median(et[:, -1])), last_median_q=float(np.median(eq[:, -1])))


@torch.inference_mode()
def eval_sequence(rt, cfg, oxford, seq, args, tmp, pm, ps):
    ds = sequence_dataset(cfg, oxford, seq, tmp)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, num_workers=args.workers, pin_memory=True, shuffle=False)
    seed_all_random_engines(SEED)
    ET, EQ, PR, GT = [], [], [], []
    gpu_s, t0, n_batches = 0.0, time.time(), 0
    for step, batch in enumerate(dl):
        if 0 <= args.max_batches <= step:
            break
        poses = batch['pose']
        B, N = poses.shape[:2]
        img = batch['image'].cuda()
        noise = torch.randn([B, N, 6], device='cuda')                 # initial DDIM noise of the model
        for _ in range(9):                                            # the model's 9 further draws (sigma = 0)
            torch.randn([B, N, 6], device='cuda')
        gt = poses.reshape(B * N, 6).cuda()
        torch.cuda.synchronize()
        a = time.perf_counter()
        pred = rt.ddim(rt.encode(img), noise).reshape(B * N, 6)
        torch.cuda.synchronize()
        gpu_s += time.perf_counter() - a
        et, eq = frame_errors(pred, gt, pm, ps)
        ET.append(et.reshape(B, N).cpu().numpy())
        EQ.append(eq.reshape(B, N).cpu().numpy())
        PR.append(pred.reshape(B, N, 6).cpu().numpy())
        GT.append(poses.numpy())
        n_batches += 1
        if step % 200 == 0:
            print(f'  {seq}: batch {step + 1}/{len(dl)}, running last-frame mean t {np.concatenate(ET)[:, -1].mean():.3f} m',
                  flush=True)
    et, eq = np.concatenate(ET), np.concatenate(EQ)
    r = stats(et, eq)
    proto = PAPER_PROTOCOL[seq]
    r.update(seq=seq, clips=int(et.shape[0]), of_total=len(ds), batches=n_batches, protocol=proto,
             paper_t=r[f'{proto}_mean_t'], paper_q=r[f'{proto}_mean_q'], wall_s=round(time.time() - t0, 1),
             model_s_per_batch=gpu_s / max(1, n_batches))
    arrays = dict(et=et, eq=eq, pred=np.concatenate(PR), gt=np.concatenate(GT))
    return r, arrays


def main(argv=None):
    ap = argparse.ArgumentParser(description='Oxford accuracy of a BiLoc 1-bit runtime (Table-2 protocol)')
    ap.add_argument('--dataroot', required=True, help='directory that contains Oxford/ (or the Oxford directory)')
    ap.add_argument('--variant', default='fused', choices=['fused', 'eqk'])
    ap.add_argument('--seqs', default=','.join(TEST_SEQS), help='comma list of test sequences')
    ap.add_argument('--max-batches', type=int, default=-1, help='limit per sequence (smoke test); -1 = all')
    ap.add_argument('--bs', type=int, default=50)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--json', type=Path, default=None, help='also write the results to this JSON file')
    ap.add_argument('--save-per-frame', type=Path, default=None,
                    help='also write per-frame errors and predictions (<seq>__et/eq/pred/gt) to this .npz file')
    args = ap.parse_args(argv)
    seqs = [s for s in args.seqs.split(',') if s]
    bad = [s for s in seqs if s not in TEST_SEQS]
    if bad:
        ap.error(f'unknown sequence(s) {bad}; choose from {TEST_SEQS}')
    oxford = oxford_dir(args.dataroot)
    torch.set_num_threads(5)
    torch.backends.cuda.matmul.allow_tf32 = False                     # strict FP32 glue
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    import deploy
    rt = deploy.build_runtime(args.variant)
    cfg = load_cfg()
    cfg.train.val_batch_size = args.bs
    pm_np, ps_np = np.loadtxt(str(oxford / 'Oxford_pose_stats.txt'))
    pm, ps = torch.from_numpy(pm_np).cuda(), torch.from_numpy(ps_np).cuda()
    print(f'variant {args.variant}, batch {args.bs}, seed {SEED} per sequence, sequences {seqs}'
          + (f', at most {args.max_batches} batches per sequence' if args.max_batches >= 0 else ''), flush=True)
    res, per_frame = {}, {}
    tmp = tempfile.mkdtemp(prefix='biloc_eval_')
    try:
        for seq in seqs:
            r, arrays = eval_sequence(rt, cfg, oxford, seq, args, tmp, pm, ps)
            res[seq] = r
            per_frame.update({f'{seq}__{k}': v for k, v in arrays.items()})
            part = '' if r['clips'] == r['of_total'] else f' (partial: {r["clips"]}/{r["of_total"]} clips)'
            print(f'{seq}: all {r["all_mean_t"]:.4f} m / {r["all_mean_q"]:.4f} deg, last {r["last_mean_t"]:.4f} m / '
                  f'{r["last_mean_q"]:.4f} deg -> paper metric ({r["protocol"]}) {r["paper_t"]:.4f} m / {r["paper_q"]:.4f} deg'
                  f'{part}  [paper: {PAPER_TABLE2[seq][0]:.2f} / {PAPER_TABLE2[seq][1]:.2f}]', flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    complete = len(seqs) == 4 and all(r['clips'] == r['of_total'] for r in res.values())
    avg_t = float(np.mean([r['paper_t'] for r in res.values()]))
    avg_q = float(np.mean([r['paper_q'] for r in res.values()]))
    print(f'average over {len(res)} sequence(s): {avg_t:.4f} m / {avg_q:.4f} deg'
          + ('' if complete else ' (not the full test set)')
          + f'  [paper Table 2: {PAPER_TABLE2["average"][0]:.2f} / {PAPER_TABLE2["average"][1]:.2f}]')
    if args.save_per_frame is not None:
        args.save_per_frame.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_per_frame, **per_frame)
        print(f'wrote {args.save_per_frame}')
    if args.json is not None:
        out = dict(variant=args.variant, bs=args.bs, seed=SEED, max_batches=args.max_batches, sequences=res,
                   average=dict(t=avg_t, q=avg_q, complete=complete), paper_table2=PAPER_TABLE2,
                   gpu=torch.cuda.get_device_name(), torch=torch.__version__)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1) + '\n')
        print(f'wrote {args.json}')
    return res


if __name__ == '__main__':
    main()
