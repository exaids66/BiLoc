"""
@author: Wen Li
@file: eavl.py
@time: 2023/9/23 18:20
"""
import time
import matplotlib
import os.path as osp
matplotlib.use('Agg')

from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from utils.train_util import *
from utils.utils import seed_all_random_engines
from utils.pose_util import qexp_t
from datasets.composition import MF
from tensorboardX import SummaryWriter


TOTAL_ITERATIONS = 0

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def test(cfg: DictConfig):
    # NOTE carefully double check the instruction from huggingface!
    global TOTAL_ITERATIONS
    OmegaConf.set_struct(cfg, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate the model
    model = instantiate(cfg.MODEL, _recursive_=False)

    eval_dataset = MF(cfg.train.dataset, cfg, split='eval')

    ckpt_path = os.path.join(cfg.ckpt)
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded checkpoint from: {ckpt_path}")
    else:
        raise ValueError(f"No checkpoint found at: {ckpt_path}")

    if cfg.train.num_workers > 0:
        persistent_workers = cfg.train.persistent_workers
    else:
        persistent_workers = False

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.train.val_batch_size,
                                                  num_workers=cfg.train.num_workers,
                                                  pin_memory=cfg.train.pin_memory,
                                                  persistent_workers=persistent_workers,
                                                  shuffle=False)  # collate

    # Move model and images to the GPU
    model = model.to(device)

    # Evaluation Mode
    model.eval()

    # Seed random engines
    seed_all_random_engines(cfg.seed)

    # pose mean and std
    pose_stats = os.path.join(cfg.train.dataroot, cfg.train.dataset, cfg.train.dataset + '_pose_stats.txt')
    pose_m, pose_s = np.loadtxt(pose_stats)
    pose_s_t = torch.from_numpy(pose_s).to(device)
    pose_m_t = torch.from_numpy(pose_m).to(device)
    # results (per-frame, consistent with train_kd.py)
    total_frames = len(eval_dataset) * cfg.train.steps
    gt_translation = np.zeros((total_frames, 3))
    pred_translation = np.zeros((total_frames, 3))
    gt_rotation = np.zeros((total_frames, 4))
    pred_rotation = np.zeros((total_frames, 4))
    error_t = np.zeros(total_frames)
    error_q = np.zeros(total_frames)
    frame_cursor = 0
    for step, batch in enumerate(eval_dataloader):
        poses = batch["pose"]  # [B, N, 6]
        batch_size, frame_size = poses.size(0), poses.size(1)
        frame_num = batch_size * frame_size
        start_idx = frame_cursor
        end_idx = frame_cursor + frame_num
        gt_pose_flat = poses.reshape(frame_num, 6)
        images = batch["image"].to(device)
        with torch.no_grad():
            predictions = model(images, sampling_timesteps=cfg.sampling_timesteps, training=False)
        # predicted pose (all frames)
        pred = predictions['pred_pose'].reshape(frame_num, 6)
        pred_t = pred[:, :3]
        pred_q = pred[:, 3:]

        # translation error (vectorized, on device)
        with torch.no_grad():
            pred_t_denorm = (pred_t * pose_s_t) + pose_m_t
            gt_t_denorm = (gt_pose_flat[:, :3].to(device) * pose_s_t) + pose_m_t
            trans_err = torch.linalg.norm(gt_t_denorm - pred_t_denorm, dim=1)

            pred_q_t = qexp_t(pred_q)
            gt_q_t = qexp_t(gt_pose_flat[:, 3:].to(device))
            dot = torch.sum(pred_q_t * gt_q_t, dim=1).abs()
            dot = torch.clamp(dot, -1.0, 1.0)
            rot_err = torch.rad2deg(2 * torch.acos(dot))

        pred_translation[start_idx:end_idx, :] = pred_t_denorm.cpu().numpy()
        gt_translation[start_idx:end_idx, :] = gt_t_denorm.cpu().numpy()
        pred_rotation[start_idx:end_idx, :] = pred_q_t.cpu().numpy()
        gt_rotation[start_idx:end_idx, :] = gt_q_t.cpu().numpy()
        error_t[start_idx:end_idx] = trans_err.cpu().numpy()
        error_q[start_idx:end_idx] = rot_err.cpu().numpy()

        log_string('MeanTE(m): %f' % np.mean(error_t[start_idx:end_idx], axis=0))
        log_string('MeanRE(degrees): %f' % np.mean(error_q[start_idx:end_idx], axis=0))
        log_string('MedianTE(m): %f' % np.median(error_t[start_idx:end_idx], axis=0))
        log_string('MedianRE(degrees): %f' % np.median(error_q[start_idx:end_idx], axis=0))
        frame_cursor = end_idx

    mean_ATE = np.mean(error_t)
    mean_ARE = np.mean(error_q)
    median_ATE = np.median(error_t)
    median_ARE = np.median(error_q)

    log_string('Mean Position Error(m): %f' % mean_ATE)
    log_string('Mean Orientation Error(degrees): %f' % mean_ARE)
    log_string('Median Position Error(m): %f' % median_ATE)
    log_string('Median Orientation Error(degrees): %f' % median_ARE)

    val_writer.add_scalar('MeanATE', mean_ATE, TOTAL_ITERATIONS)
    val_writer.add_scalar('MeanARE', mean_ARE, TOTAL_ITERATIONS)
    val_writer.add_scalar('MedianATE', median_ATE, TOTAL_ITERATIONS)
    val_writer.add_scalar('MedianARE', median_ARE, TOTAL_ITERATIONS)

    # trajectory
    fig = plt.figure()
    real_pose = pred_translation - pose_m
    gt_pose = gt_translation - pose_m
    plt.scatter(gt_pose[:, 1], gt_pose[:, 0], s=1, c='black')
    plt.scatter(real_pose[:, 1], real_pose[:, 0], s=1, c='red')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=10)
    image_filename = os.path.join(os.path.expanduser(cfg.exp_dir), '{:s}.png'.format('trajectory'))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # save error and trajectory
    error_t_filename = osp.join(cfg.exp_dir, 'error_t.txt')
    error_q_filename = osp.join(cfg.exp_dir, 'error_q.txt')
    pred_t_filename = osp.join(cfg.exp_dir, 'pred_t.txt')
    gt_t_filename = osp.join(cfg.exp_dir, 'gt_t.txt')
    pred_q_filename = osp.join(cfg.exp_dir, 'pred_q.txt')
    gt_q_filename = osp.join(cfg.exp_dir, 'gt_q.txt')
    np.savetxt(error_t_filename, error_t, fmt='%8.7f')
    np.savetxt(error_q_filename, error_q, fmt='%8.7f')
    np.savetxt(pred_t_filename, real_pose, fmt='%8.7f')
    np.savetxt(gt_t_filename, gt_pose, fmt='%8.7f')
    np.savetxt(pred_q_filename, pred_rotation, fmt='%8.7f')
    np.savetxt(gt_q_filename, gt_rotation, fmt='%8.7f')


if __name__ == '__main__':
    # oxford.yaml / nclt.yaml
    conf = OmegaConf.load('cfgs/nclt.yaml')
    LOG_FOUT = open(os.path.join(conf.exp_dir, 'log.txt'), 'w')
    LOG_FOUT.write(str(conf) + '\n')
    val_writer = SummaryWriter(os.path.join(conf.exp_dir, 'valid'))
    # 5 cpu core
    torch.set_num_threads(5)
    test(conf)
