"""
@author: Wen Li
@file: train.py
@time: 2023/9/20 14:57
"""
import io
import os
import time
import pstats
import cProfile
import numpy as np
import torch
from tensorboardX import SummaryWriter

from hydra.utils import instantiate
from collections import OrderedDict
from omegaconf import OmegaConf, DictConfig
from utils.train_util import *
from utils.pose_util import qexp_t, val_rotation
from datasets.composition import MF
from tqdm import tqdm


def prefix_with_module(checkpoint):
    prefixed_checkpoint = OrderedDict()
    for key, value in checkpoint.items():
        prefixed_key = "module." + key
        prefixed_checkpoint[prefixed_key] = value
    return prefixed_checkpoint


# Wrapper for cProfile.Profile for easily make optional, turn on/off and printing
class Profiler:
    def __init__(self, active: bool):
        self.c_profiler = cProfile.Profile()
        self.active = active

    def enable(self):
        if self.active:
            self.c_profiler.enable()

    def disable(self):
        if self.active:
            self.c_profiler.disable()

    def print(self):
        if self.active:
            s = io.StringIO()
            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(self.c_profiler, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())


def get_thread_count(var_name):
    return os.environ.get(var_name)


def train_fn(cfg: DictConfig):
    # NOTE carefully double check the instruction from huggingface!

    OmegaConf.set_struct(cfg, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Model Config:")
    print(OmegaConf.to_yaml(cfg))
    print(f"Using device: {device}")

    torch.backends.cudnn.benchmark = cfg.train.cudnnbenchmark

    set_seed_and_print(cfg.seed)

    writer = SummaryWriter(log_dir=cfg.exp_dir)

    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!  OMP_NUM_THREADS: {get_thread_count('OMP_NUM_THREADS')}")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!  MKL_NUM_THREADS: {get_thread_count('MKL_NUM_THREADS')}")

    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!  SLURM_CPU_BIND: {get_thread_count('SLURM_CPU_BIND')}")
    print(
        f"!!!!!!!!!!!!!!!!!!!!!!!!!!  SLURM_JOB_CPUS_PER_NODE: {get_thread_count('SLURM_JOB_CPUS_PER_NODE')}")

    train_dataset = MF(cfg.train.dataset, cfg, split='train')
    eval_dataset = MF(cfg.train.dataset, cfg, split='eval')

    if cfg.train.num_workers > 0:
        persistent_workers = cfg.train.persistent_workers
    else:
        persistent_workers = False

    # ✅ only valid when num_workers > 0
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        shuffle=True,
        drop_last=True,
        persistent_workers=persistent_workers,
    )  # collate_fn

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        shuffle=False,
        persistent_workers=persistent_workers,
    )  # collate_fn

    print("length of train dataloader is: ", len(dataloader))
    print("length of eval dataloader is: ", len(eval_dataloader))

    # Instantiate the model
    model = instantiate(cfg.MODEL, _recursive_=False)

    model = model.to(device)

    # Define the numer of epoch
    num_epochs = cfg.train.epochs

    # log
    if os.path.exists(cfg.exp_dir) == 0:
        os.mkdir(cfg.exp_dir)
    # Define the optimizer
    if cfg.train.warmup_sche:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.train.lr)
        lr_scheduler = WarmupCosineLR(optimizer=optimizer, lr=cfg.train.lr,
                                      warmup_steps=cfg.train.restart_num * len(dataloader), momentum=0.9,
                                      max_steps=len(dataloader) * (cfg.train.epochs - cfg.train.restart_num))
    else:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)


    print(f"xxxxxxxxxxxxxxxxxx dataloader has {dataloader.num_workers} num_workers")
    start_epoch = 0

    to_plot = ("loss", "lr", "diffloss","error_t", "error_q")

    stats = VizStats(to_plot)

    best_val_error_t = None
    best_val_epoch_t = None
    best_val_error_q = None
    best_val_epoch_q = None

    pose_stats = os.path.join(cfg.train.dataroot, cfg.train.dataset, cfg.train.dataset + '_pose_stats.txt')
    pose_m, pose_s = np.loadtxt(pose_stats)
    pose_s = torch.from_numpy(pose_s).to(device)
    pose_m = torch.from_numpy(pose_m).to(device)

    for epoch in range(start_epoch, num_epochs):
        stats.new_epoch()

        set_seed_and_print(cfg.seed + epoch)

        # Evaluation (only when epoch >= 100)
        if (epoch >= 100) and (epoch % cfg.train.eval_interval == 0):
            print(f"----------Start to eval at epoch {epoch}----------")
            val_error_t, val_error_q = _train_or_eval_fn(
                model,
                eval_dataloader,
                cfg,
                optimizer,
                stats,
                lr_scheduler,
                training=False,
                device=device,
                pose_m=pose_m,
                pose_s=pose_s,
            )
            print(f"----------Finish the eval at epoch {epoch}----------")

            # Track best models from epoch 50 onwards based on validation error_t/error_q (lower is better).
            if (val_error_t is not None) and (epoch >= 0):
                is_best_t = (
                    best_val_error_t is None or val_error_t < best_val_error_t
                )
                if is_best_t:
                    best_val_error_t = val_error_t
                    best_val_epoch_t = epoch
                    ckpt_path_t = os.path.join(cfg.exp_dir, "best_t.pth")
                    print(
                        f"----------New best (translation) at epoch {epoch} with error_t={val_error_t:.6f}, saving to {ckpt_path_t}----------"
                    )
                    torch.save(model.state_dict(), ckpt_path_t)
                else:
                    print(
                        f"----------Validation error_t={val_error_t:.6f} (best so far: {best_val_error_t:.6f} at epoch {best_val_epoch_t})----------"
                    )

            if (val_error_q is not None) and (epoch >= 0):
                is_best_q = (
                    best_val_error_q is None or val_error_q < best_val_error_q
                )
                if is_best_q:
                    best_val_error_q = val_error_q
                    best_val_epoch_q = epoch
                    ckpt_path_q = os.path.join(cfg.exp_dir, "best_q.pth")
                    print(
                        f"----------New best (rotation) at epoch {epoch} with error_q={val_error_q:.6f}, saving to {ckpt_path_q}----------"
                    )
                    torch.save(model.state_dict(), ckpt_path_q)
                else:
                    print(
                        f"----------Validation error_q={val_error_q:.6f} (best so far: {best_val_error_q:.6f} at epoch {best_val_epoch_q})----------"
                    )
        else:
            print(f"----------Skip the eval at epoch {epoch}----------")

        # Training
        print(f"----------Start to train at epoch {epoch}----------")
        _train_or_eval_fn(
            model,
            dataloader,
            cfg,
            optimizer,
            stats,
            lr_scheduler,
            training=True,
            device=device,
            pose_m=pose_m,
            pose_s=pose_s,
        )
        print(f"----------Finish the train at epoch {epoch}----------")

        for g in optimizer.param_groups:
            lr = g['lr']
            break
        print(f"----------LR is {lr}----------")
        print(f"----------Saving stats to {cfg.exp_name}----------")
        stats.update({"lr": lr}, stat_set="train")
        # TensorBoard logging for latest epoch averages
        for mode in ["train", "eval"]:
            for metric in to_plot:
                try:
                    vals = stats.stats[mode][metric].get_epoch_averages()
                except KeyError:
                    continue
                if vals is None or len(vals) == 0:
                    continue
                writer.add_scalar(f"{mode}/{metric}", vals[-1], epoch)
        writer.flush()
        print(f"----------Done----------")
        stats.save(cfg.exp_dir + "stats")

    writer.close()
    return True


def _train_or_eval_fn(
    model,
    dataloader,
    cfg,
    optimizer,
    stats,
    lr_scheduler,
    training=True,
    device=None,
    pose_m=None,
    pose_s=None,
):
    if training:
        model.train()
    else:
        model.eval()

    time_start = time.time()
    max_it = len(dataloader)
    train_print_interval = max(cfg.train.print_interval, 50) if training else cfg.train.print_interval
    use_cuda_timing = device is not None and torch.cuda.is_available()
    if use_cuda_timing and isinstance(device, torch.device):
        use_cuda_timing = device.type == "cuda"
    elif use_cuda_timing:
        use_cuda_timing = str(device).startswith("cuda")

    assert pose_m is not None and pose_s is not None, "pose_m/pose_s must be provided (loaded once in train_fn)."

    tqdm_loader = tqdm(dataloader, total=len(dataloader))
    sum_error_t = 0.0
    sum_error_q = 0.0
    total_frames = 0
    for step, batch in enumerate(tqdm_loader):
        images = batch["image"].to(device)  # [B, N, 5, 32, 512]
        batch_size, frame_size = images.size(0), images.size(1)
        poses = batch["pose"].to(device)  # [B, N, 6]

        if training:
            predictions = model(images, poses, training=True)
            loss = predictions["diffloss"]
        else:
            with torch.no_grad():
                predictions = model(images, training=False)

        # calculate metric
        frame_num = frame_size * batch_size
        pred_poses = predictions['pred_pose'].reshape(frame_num, 6)  # [B*N, 6]
        gt_poses = poses.reshape(frame_num, 6)  # [B*N, 6]

        # Only compute translation/rotation errors during evaluation to save training time.
        if not training:
            with torch.no_grad():
                pred_t = (pred_poses[:, :3] * pose_s) + pose_m
                gt_t = (gt_poses[:, :3] * pose_s) + pose_m
                trans_err = torch.linalg.norm(gt_t - pred_t, dim=1)
                predictions['error_t'] = trans_err.mean()

                pred_q = qexp_t(pred_poses[:, 3:])
                gt_q = qexp_t(gt_poses[:, 3:])
                dot = torch.sum(pred_q * gt_q, dim=1).abs()
                dot = torch.clamp(dot, -1.0, 1.0)
                rot_err = torch.rad2deg(2 * torch.acos(dot))
                predictions['error_q'] = rot_err.mean()

        if training:
            optimizer.zero_grad()
            loss.backward()
            if cfg.train.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
            optimizer.step()
            # Step scheduler after the optimizer to avoid skipping the first LR value.
            lr_scheduler.step()

        if training:
            stats.update(predictions, time_start=time_start, stat_set="train")
            if step % train_print_interval == 0:
                print(stats.print(stat_set='train', max_it=max_it))
        else:
            stats.update(predictions, time_start=time_start, stat_set="eval")
            if step % cfg.train.print_interval == 0:
                print(stats.print(stat_set='eval', max_it=max_it))
            # Accumulate validation error_t weighted by frame count for epoch-level mean.
            frame_num = frame_size * batch_size
            sum_error_t += predictions['error_t'].item() * frame_num
            if 'error_q' in predictions:
                sum_error_q += predictions['error_q'].item() * frame_num
            total_frames += frame_num
    if not training and total_frames > 0:
        mean_error_t = sum_error_t / total_frames
        mean_error_q = sum_error_q / total_frames if total_frames > 0 else None
        return mean_error_t, mean_error_q
    return True


def t_error(pred_poses, gt_poses, pose_s, pose_mean):
    with torch.no_grad():
        error_t = val_translation(pred_poses, gt_poses, pose_s, pose_mean)

    return error_t


def val_translation(pred_p, gt_p, pose_s, pose_mean):
    """
    test model, compute error (numpy)
    input:
        pred_p: [3,]
        gt_p: [3,]
    returns:
        translation error (m):
    """
    pred_p = (pred_p * pose_s) + pose_mean
    gt_p = (gt_p * pose_s) + pose_mean
    error = torch.linalg.norm(gt_p - pred_p)

    return error


if __name__ == '__main__':
    # oxford.yaml / nclt.yaml
    conf = OmegaConf.load('cfgs/oxford.yaml')
    train_fn(conf)
