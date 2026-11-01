import io
import os
import sys
import time
import pstats
import cProfile
import importlib.util
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter

from hydra.utils import instantiate
from collections import OrderedDict
from omegaconf import OmegaConf, DictConfig, ListConfig
from loss import (
    EntropyLoss,
    KdWeightedMSELoss,
    KdHardMaskMSELoss,
    KdTopKSampleLoss,
    KdDualBranchLoss,
    KdAdaptiveTemperatureLoss,
    KdSigmaMapWeightedLoss,
    KdITLoss,
    KdLCKTLoss,
    build_importance,
    load_sigma_maps,
)
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


def _load_full_precision_models(fp_root):
    module_name = "fp_models"
    if module_name in sys.modules:
        return sys.modules[module_name]
    fp_models_path = os.path.join(fp_root, "models")
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(fp_models_path, "__init__.py")
    )
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [fp_models_path]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _replace_model_targets(cfg_node):
    if isinstance(cfg_node, (DictConfig, dict)):
        for key in list(cfg_node.keys()):
            value = cfg_node[key]
            if isinstance(value, str) and value.startswith("models"):
                cfg_node[key] = "fp_models" + value[len("models"):]
            else:
                _replace_model_targets(value)
    elif isinstance(cfg_node, (ListConfig, list, tuple)):
        for value in cfg_node:
            _replace_model_targets(value)


def _load_state_dict_flexible(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        pass
    try:
        model.load_state_dict(prefix_with_module(state_dict), strict=True)
        return
    except RuntimeError:
        pass
    stripped = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            stripped[k[len("module."):]] = v
        else:
            stripped[k] = v
    model.load_state_dict(stripped, strict=False)


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


def _resolve_teacher_cfg_path(cfg, project_root):
    kd_cfg = cfg.get("KD", {})
    if "teacher_cfg" in kd_cfg:
        return kd_cfg.teacher_cfg
    dataset = cfg.train.dataset.lower()
    if dataset == "oxford":
        return os.path.join(project_root, "diffoc_full_precision", "cfgs", "oxford.yaml")
    if dataset == "nclt":
        return os.path.join(project_root, "diffoc_full_precision", "cfgs", "nclt.yaml")
    return None


def _get_feature_dim(model):
    extractor = getattr(model, "image_feature_extractor", None)
    if extractor is None:
        return None
    if hasattr(extractor, "get_output_dim"):
        return extractor.get_output_dim()
    return None


def train_fn(cfg: DictConfig):
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

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train.batch_size,
                                             num_workers=cfg.train.num_workers,
                                             pin_memory=cfg.train.pin_memory,
                                             shuffle=True, drop_last=True,
                                             persistent_workers=persistent_workers
                                             )  # collate_fn
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.train.batch_size,
                                                  num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory,
                                                  shuffle=False, persistent_workers=persistent_workers)  # collate_fn

    print("length of train dataloader is: ", len(dataloader))
    print("length of eval dataloader is: ", len(eval_dataloader))

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    model = instantiate(cfg.MODEL, _recursive_=False).to(device)

    teacher_cfg_path = _resolve_teacher_cfg_path(cfg, project_root)
    if teacher_cfg_path is None:
        raise ValueError("KD.teacher_cfg is required when dataset is not Oxford/NCLT.")
    teacher_cfg = OmegaConf.load(teacher_cfg_path)
    _load_full_precision_models(os.path.join(project_root, "diffoc_full_precision"))
    _replace_model_targets(teacher_cfg)
    teacher_model = instantiate(teacher_cfg.MODEL, _recursive_=False).to(device)
    teacher_ckpt = cfg.get("KD", {}).get("teacher_ckpt", None) or teacher_cfg.get("ckpt", None)
    if teacher_ckpt is None:
        raise ValueError("Teacher checkpoint is missing. Set KD.teacher_ckpt or ckpt in teacher cfg.")
    _load_state_dict_flexible(teacher_model, teacher_ckpt, device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    kd_cfg = cfg.get("KD", {})
    s_dim = kd_cfg.get("s_dim", None) or _get_feature_dim(model)
    t_dim = kd_cfg.get("t_dim", None) or _get_feature_dim(teacher_model)
    kd_loss_type = kd_cfg.get("loss_type", "entropy")
    if kd_loss_type == "weighted":
        kd_loss_fn = KdWeightedMSELoss(
            threshold=kd_cfg.get("kp_threshold", 0.0),
            tau=kd_cfg.get("kp_tau", 1.0),
            use_sigmoid=kd_cfg.get("kp_use_sigmoid", True),
        ).to(device)
    elif kd_loss_type == "hard_mask":
        kd_loss_fn = KdHardMaskMSELoss(
            threshold=kd_cfg.get("kp_threshold", 0.5),
            top_ratio=kd_cfg.get("mask_ratio", None),
        ).to(device)
    elif kd_loss_type == "topk":
        kd_loss_fn = KdTopKSampleLoss(
            top_ratio=kd_cfg.get("mask_ratio", 0.3),
        ).to(device)
    elif kd_loss_type == "dual":
        kd_loss_fn = KdDualBranchLoss(
            alpha=kd_cfg.get("dual_alpha", 1.0),
            threshold=kd_cfg.get("kp_threshold", 0.0),
            tau=kd_cfg.get("kp_tau", 1.0),
        ).to(device)
    elif kd_loss_type == "adaptive_temp":
        kd_loss_fn = KdAdaptiveTemperatureLoss(
            t_min=kd_cfg.get("t_min", 0.5),
            t_max=kd_cfg.get("t_max", 2.0),
        ).to(device)
    elif kd_loss_type == "sigma_map":
        kd_loss_fn = KdSigmaMapWeightedLoss(
            invert=kd_cfg.get("sigma_invert", True),
            normalize=kd_cfg.get("sigma_normalize", True),
        ).to(device)
    elif kd_loss_type == "it_corr":
        if s_dim is None or t_dim is None:
            raise ValueError("ITLoss requires s_dim/t_dim. Set KD.s_dim or KD.t_dim if missing.")
        kd_loss_fn = KdITLoss(
            s_dim=s_dim,
            t_dim=t_dim,
            n_data=kd_cfg.get("it_n_data", 1),
            alpha_it=kd_cfg.get("it_alpha", 1.0),
            mode="corr",
        ).to(device)
    elif kd_loss_type == "it_mutual":
        if s_dim is None or t_dim is None:
            raise ValueError("ITLoss requires s_dim/t_dim. Set KD.s_dim or KD.t_dim if missing.")
        kd_loss_fn = KdITLoss(
            s_dim=s_dim,
            t_dim=t_dim,
            n_data=kd_cfg.get("it_n_data", 1),
            alpha_it=kd_cfg.get("it_alpha", 1.0),
            mode="mutual",
        ).to(device)
    else:
        kd_loss_fn = EntropyLoss().to(device)

    struct_weight = kd_cfg.get("struct_weight", 0.0)
    struct_loss_type = kd_cfg.get("struct_loss_type", "lckt")
    struct_loss_fn = None
    if struct_weight > 0:
        if struct_loss_type == "lckt":
            struct_loss_fn = KdLCKTLoss(
                eps=kd_cfg.get("lckt_eps", 0.05),
                T=kd_cfg.get("lckt_T", 1),
                K=kd_cfg.get("lckt_K", 20),
                alpha=kd_cfg.get("lckt_alpha", 0.5),
                trans_w=kd_cfg.get("lckt_trans_w", 1.0),
                rot_w=kd_cfg.get("lckt_rot_w", 1.0),
            ).to(device)
        elif struct_loss_type == "it_mutual":
            if s_dim is None or t_dim is None:
                raise ValueError("ITLoss requires s_dim/t_dim. Set KD.s_dim or KD.t_dim if missing.")
            struct_loss_fn = KdITLoss(
                s_dim=s_dim,
                t_dim=t_dim,
                n_data=kd_cfg.get("it_n_data", 1),
                alpha_it=kd_cfg.get("it_alpha", 1.0),
                mode="mutual",
            ).to(device)
        else:
            raise ValueError(f"Unknown KD.struct_loss_type: {struct_loss_type}")

    # Define the numer of epoch
    num_epochs = cfg.train.epochs

    # log
    if os.path.exists(cfg.exp_dir) == 0:
        os.mkdir(cfg.exp_dir)
    extra_params = list(kd_loss_fn.parameters())
    if struct_loss_fn is not None:
        extra_params += list(struct_loss_fn.parameters())
    if cfg.train.warmup_sche:
        optimizer = torch.optim.AdamW(params=list(model.parameters()) + extra_params,
                                      lr=cfg.train.lr)
        lr_scheduler = WarmupCosineLR(optimizer=optimizer, lr=cfg.train.lr,
                                      warmup_steps=cfg.train.restart_num * len(dataloader), momentum=0.9,
                                      max_steps=len(dataloader) * (cfg.train.epochs - cfg.train.restart_num))
    else:
        optimizer = torch.optim.AdamW(params=list(model.parameters()) + extra_params,
                                      lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)

    print(f"xxxxxxxxxxxxxxxxxx dataloader has {dataloader.num_workers} num_workers")

    start_epoch = 0

    to_plot = ("loss", "lr", "diffloss", "kd_loss", "struct_loss", "error_t", "error_q")

    stats = VizStats(to_plot)

    best_val_error_t = None
    best_val_epoch_t = None
    best_val_error_q = None
    best_val_epoch_q = None

    for epoch in range(start_epoch, num_epochs):
        stats.new_epoch()

        set_seed_and_print(cfg.seed + epoch)

        if (epoch >= 150):
            print(f"----------Start to eval at epoch {epoch}----------")
            val_error_t, val_error_q = _train_or_eval_fn(
                model,
                teacher_model,
                kd_loss_fn,
                struct_loss_fn,
                criterion,
                eval_dataloader,
                cfg,
                optimizer,
                stats,
                lr_scheduler,
                epoch=epoch,
                training=False,
                device=device,
            )
            print(f"----------Finish the eval at epoch {epoch}----------")

            # Track best models from epoch 50 onwards based on validation error_t/error_q (lower is better).
            if (val_error_t is not None) and (epoch >= 150):
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

            if (val_error_q is not None) and (epoch >= 150):
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

        print(f"----------Start to train at epoch {epoch}----------")
        _train_or_eval_fn(
            model,
            teacher_model,
            kd_loss_fn,
            struct_loss_fn,
            criterion,
            dataloader,
            cfg,
            optimizer,
            stats,
            lr_scheduler,
            epoch=epoch,
            training=True,
            device=device,
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
                if mode == "train" and metric == "error_t":
                    continue
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


def _train_or_eval_fn(model, teacher_model, kd_loss_fn, struct_loss_fn, criterion, dataloader, cfg, optimizer, stats,
                      lr_scheduler, epoch=None, training=True, device=None):
    if training:
        model.train()
    else:
        model.eval()

    time_start = time.time()
    max_it = len(dataloader)

    pose_stats = os.path.join(cfg.train.dataroot, cfg.train.dataset, cfg.train.dataset + '_pose_stats.txt')
    pose_m, pose_s = np.loadtxt(pose_stats)
    pose_s = torch.from_numpy(pose_s).to(device)
    pose_m = torch.from_numpy(pose_m).to(device)

    kd_cfg = cfg.get("KD", {})
    kd_weight = kd_cfg.get("weight", 1.0)
    kd_loss_type = kd_cfg.get("loss_type", "entropy")
    struct_weight = kd_cfg.get("struct_weight", 0.0)
    struct_loss_type = kd_cfg.get("struct_loss_type", "lckt")
    kd_stop_epoch = kd_cfg.get("kd_stop_epoch", None)
    struct_stop_epoch = kd_cfg.get("struct_stop_epoch", None)
    if kd_stop_epoch is not None and epoch is not None and epoch >= kd_stop_epoch:
        kd_weight = 0.0
    if struct_stop_epoch is not None and epoch is not None and epoch >= struct_stop_epoch:
        struct_weight = 0.0

    tqdm_loader = tqdm(dataloader, total=len(dataloader))
    sum_error_t = 0.0
    sum_error_q = 0.0
    total_frames = 0
    for step, batch in enumerate(tqdm_loader):
        images = batch["image"].to(device)  # [B, N, 5, 32, 512]
        batch_size, frame_size = images.size(0), images.size(1)
        poses = batch["pose"].to(device)  # [B, N, 6]
        H, W = images.size(-2), images.size(-1)
        pose_flat = poses.reshape(batch_size * frame_size, 6)

        if training:
            predictions = model(images, poses, training=True)
            predictions["diffloss"] = predictions["diffloss"]
            need_kd = kd_weight > 0.0
            need_struct = struct_weight > 0.0
            if need_kd or need_struct:
                with torch.no_grad():
                    teacher_pred = teacher_model(images, poses, training=True)
                student_feat = predictions["z_out4distil"]
                teacher_feat = teacher_pred["z_out4distil"]
            else:
                student_feat = None
                teacher_feat = None

            if need_kd:
                if kd_loss_type == "sigma_map":
                    sigma_root = kd_cfg.get("sigma_map_dir", None)
                    if sigma_root is None:
                        raise ValueError("KD.sigma_map_dir is required for sigma_map loss.")
                    sigma_maps = load_sigma_maps(
                        batch.get("name", None),
                        sigma_root,
                        sigma_filename=kd_cfg.get("sigma_filename", "sigma.npy"),
                        missing=kd_cfg.get("sigma_missing", "error"),
                    ).to(device)
                    kd_loss = kd_loss_fn(teacher_feat, student_feat, sigma_maps) * kd_weight
                elif kd_loss_type in ("it_corr", "it_mutual"):
                    kd_loss = kd_loss_fn(teacher_feat, student_feat, None) * kd_weight
                else:
                    importance = build_importance(teacher_feat, student_feat, kd_cfg)
                    kd_loss = kd_loss_fn(teacher_feat, student_feat, importance) * kd_weight
                predictions["kd_loss"] = kd_loss
            else:
                kd_loss = torch.zeros((), device=device)
                predictions["kd_loss"] = kd_loss
            if struct_loss_fn is not None and struct_weight > 0.0:
                struct_pose = pose_flat if struct_loss_type == "lckt" else None
                struct_loss = struct_loss_fn(teacher_feat, student_feat, struct_pose) * struct_weight
                predictions["struct_loss"] = struct_loss
            else:
                struct_loss = torch.zeros((), device=device)
            loss = predictions["diffloss"] + kd_loss + struct_loss
            predictions["loss"] = loss
        else:
            with torch.no_grad():
                predictions = model(images, training=False)

        if not training:
            frame_num = frame_size * batch_size
            pred_poses = predictions['pred_pose'].reshape(frame_num, 6)  # [B*N, 6]
            gt_poses = poses.reshape(frame_num, 6)  # [B*N, 6]

            # Vectorized translation error on device.
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
            stats.update(predictions, time_start=time_start, stat_set="train")
            if step % cfg.train.print_interval == 0:
                original_log_vars = stats.log_vars
                if "error_t" in original_log_vars:
                    stats.log_vars = tuple(v for v in original_log_vars if v != "error_t")
                print(stats.print(stat_set="train", max_it=max_it))
                stats.log_vars = original_log_vars
        else:
            stats.update(predictions, time_start=time_start, stat_set="eval")
            if step % cfg.train.print_interval == 0:
                print(stats.print(stat_set="eval", max_it=max_it))
            # Accumulate validation error_t weighted by frame count for epoch-level mean.
            frame_num = frame_size * batch_size
            sum_error_t += predictions['error_t'].item() * frame_num
            if 'error_q' in predictions:
                sum_error_q += predictions['error_q'].item() * frame_num
            total_frames += frame_num

        if training:
            optimizer.zero_grad()
            loss.backward()
            if cfg.train.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
            optimizer.step()
            # Step scheduler after the optimizer to avoid skipping the first LR value.
            lr_scheduler.step()

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
    pred_p = (pred_p * pose_s) + pose_mean
    gt_p = (gt_p * pose_s) + pose_mean
    error = torch.linalg.norm(gt_p - pred_p)

    return error


if __name__ == '__main__':
    # oxford.yaml / nclt.yaml
    conf = OmegaConf.load('cfgs/nclt.yaml')
    train_fn(conf)
