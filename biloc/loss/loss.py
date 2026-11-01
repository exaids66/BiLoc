import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pose_util import qexp_t


def _normalize_importance(weight, eps=1e-6):
    if weight is None:
        return None
    dims = list(range(1, weight.dim()))
    w_min = weight.amin(dim=dims, keepdim=True)
    w_max = weight.amax(dim=dims, keepdim=True)
    return (weight - w_min) / (w_max - w_min + eps)


def _topk_mask(weight, top_ratio):
    flat = weight.view(weight.size(0), -1)
    k = max(1, int(round(flat.size(1) * top_ratio)))
    topk_idx = torch.topk(flat, k=k, dim=1, largest=True).indices
    mask = torch.zeros_like(flat)
    mask.scatter_(1, topk_idx, 1.0)
    return mask.view_as(weight)


class EntropyLoss(nn.Module):
    def __init__(self, init_num=1.000001):
        super().__init__()
        self.sigma = nn.Parameter(self._init_sigma(init_num))

    @staticmethod
    def _init_sigma(init_num):
        return torch.full(size=(1, 1, 1, 1), fill_value=init_num, dtype=torch.float32)

    def forward(self, teacher_feat, student_feat, importance=None):
        if importance is None:
            importance = torch.ones_like(teacher_feat)
        norm = max(1.0, importance.sum())
        diff = teacher_feat - student_feat
        mse = torch.pow(diff, 2) * importance
        corre = torch.pow(self.sigma, -2) * torch.pow(diff, 2) * importance
        det = torch.log(torch.pow(self.sigma, 2))
        return (mse + corre + 1e-3 * det).sum() / norm


# class EntropyLoss(nn.Module):
#     def __init__(self, init_num=1.000001, beta=1.0):
#         super().__init__()
#         self.sigma = nn.Parameter(self._init_sigma(init_num))
#
#     @staticmethod
#     def _init_sigma(init_num):
#         return torch.full(size=(1, 1, 1, 1), fill_value=init_num, dtype=torch.float32)
#
#     def forward(self, teacher_feat, student_feat, importance=None):
#         if importance is None:
#             importance = torch.ones_like(teacher_feat)
#
#         norm = importance.sum().clamp_min(1.0)
#
#         diff = teacher_feat - student_feat  # [B, C]
#
#         s1 = F.smooth_l1_loss(
#             teacher_feat, student_feat,
#             beta=self.beta,
#             reduction="none"
#         )  # [B, C]
#
#         base = s1 * importance
#
#         corre = torch.pow(self.sigma, -2) * s1 * importance
#
#         det = torch.log(torch.pow(self.sigma, 2))
#
#         return (base + corre + 1e-3 * det).sum() / norm



class KdWeightedMSELoss(nn.Module):
    def __init__(self, threshold=0.0, tau=1.0, use_sigmoid=True, eps=1e-6):
        super().__init__()
        self.threshold = threshold
        self.tau = tau
        self.use_sigmoid = use_sigmoid
        self.eps = eps

    def forward(self, teacher_feat, student_feat, importance=None):
        diff = teacher_feat - student_feat
        if importance is None:
            return torch.mean(diff ** 2)
        if self.use_sigmoid:
            weight = torch.sigmoid((importance - self.threshold) / max(self.tau, self.eps))
        else:
            weight = _normalize_importance(importance, eps=self.eps)
        weighted = (diff ** 2) * weight
        return weighted.sum() / (weight.sum() + self.eps)


class KdHardMaskMSELoss(nn.Module):
    def __init__(self, threshold=0.5, top_ratio=None, eps=1e-6):
        super().__init__()
        self.threshold = threshold
        self.top_ratio = top_ratio
        self.eps = eps

    def forward(self, teacher_feat, student_feat, importance=None):
        diff = teacher_feat - student_feat
        if importance is None:
            importance = torch.abs(diff.detach())
        if self.top_ratio is not None:
            mask = _topk_mask(importance, self.top_ratio)
        else:
            mask = (importance > self.threshold).float()
        return ((diff ** 2) * mask).sum() / (mask.sum() + self.eps)


class KdTopKSampleLoss(nn.Module):
    def __init__(self, top_ratio=0.3, eps=1e-6):
        super().__init__()
        self.top_ratio = top_ratio
        self.eps = eps

    def forward(self, teacher_feat, student_feat, importance=None):
        diff = teacher_feat - student_feat
        if importance is None:
            importance = torch.abs(diff.detach())
        mask = _topk_mask(importance, self.top_ratio)
        return ((diff ** 2) * mask).sum() / (mask.sum() + self.eps)


class KdDualBranchLoss(nn.Module):
    def __init__(self, alpha=1.0, threshold=0.0, tau=1.0):
        super().__init__()
        self.alpha = alpha
        self.weighted = KdWeightedMSELoss(threshold=threshold, tau=tau, use_sigmoid=True)

    def forward(self, teacher_feat, student_feat, importance=None):
        diff = teacher_feat - student_feat
        global_loss = torch.mean(diff ** 2)
        weighted_loss = self.weighted(teacher_feat, student_feat, importance)
        return global_loss + self.alpha * weighted_loss


class KdAdaptiveTemperatureLoss(nn.Module):
    def __init__(self, t_min=0.5, t_max=2.0, eps=1e-6):
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        self.eps = eps

    def forward(self, teacher_feat, student_feat, importance=None):
        diff = teacher_feat - student_feat
        if importance is None:
            temp = torch.ones_like(diff)
        else:
            weight = _normalize_importance(importance, eps=self.eps)
            temp = self.t_max - (self.t_max - self.t_min) * weight
            temp = temp.clamp(min=self.t_min, max=self.t_max)
        loss = (diff ** 2) / (temp ** 2) + torch.log(temp + self.eps)
        return loss.mean()


class KdSigmaMapWeightedLoss(nn.Module):
    def __init__(self, invert=True, normalize=True, eps=1e-6):
        super().__init__()
        self.invert = invert
        self.normalize = normalize
        self.eps = eps

    def forward(self, teacher_feat, student_feat, sigma_map):
        if sigma_map is None:
            raise ValueError("sigma_map is required for sigma_map loss.")
        if sigma_map.dim() == 2:
            sigma_map = sigma_map.unsqueeze(0).unsqueeze(0)
        elif sigma_map.dim() == 3:
            sigma_map = sigma_map.unsqueeze(1)
        elif sigma_map.dim() == 4 and sigma_map.size(1) != 1:
            sigma_map = sigma_map.mean(dim=1, keepdim=True)
        if sigma_map.size(0) != teacher_feat.size(0):
            raise ValueError(f"sigma_map batch {sigma_map.size(0)} != feature batch {teacher_feat.size(0)}")

        weight = sigma_map
        if self.invert:
            weight = 1.0 / (weight + self.eps)
        if self.normalize:
            dims = list(range(1, weight.dim()))
            w_min = weight.amin(dim=dims, keepdim=True)
            w_max = weight.amax(dim=dims, keepdim=True)
            weight = (weight - w_min) / (w_max - w_min + self.eps)
        weight = weight.mean(dim=(1, 2, 3))

        diff = teacher_feat - student_feat
        mse_per = torch.mean(diff ** 2, dim=1)
        return (mse_per * weight).sum() / (weight.sum() + self.eps)


class Embed(nn.Module):
    def __init__(self, s_dim, t_dim, n=1):
        super().__init__()
        self.proj = nn.Linear(s_dim, t_dim)

    def forward(self, x):
        return self.proj(x)


class ITLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.s_dim = opt.s_dim
        self.t_dim = opt.t_dim
        self.n_data = opt.n_data
        self.alpha_it = opt.alpha_it
        self.embed = Embed(opt.s_dim, opt.t_dim, n=1)

    def forward_correlation_it(self, z_s, z_t):
        f_s = z_s
        f_t = z_t

        f_s = self.embed(f_s)

        n, _ = f_s.shape

        f_s_norm = (f_s - f_s.mean(0)) / f_s.std(0)
        f_t_norm = (f_t - f_t.mean(0)) / f_t.std(0)
        c_st = torch.einsum('bx,bx->x', f_s_norm, f_t_norm) / n
        c_diff = c_st - torch.ones_like(c_st)

        alpha = self.alpha_it
        c_diff = torch.abs(c_diff)
        c_diff = c_diff.pow(2.0)
        c_diff = c_diff.pow(alpha)

        loss = torch.log2(c_diff.sum())
        return loss

    def forward_mutual_it(self, z_s, z_t):
        f_s = z_s
        f_t = z_t

        if self.s_dim != self.t_dim:
            f_s = self.embed(f_s)

        f_s_norm = F.normalize(f_s)
        f_t_norm = F.normalize(f_t)

        # 1. Polynomial kernel
        G_s = torch.einsum('bx,dx->bd', f_s_norm, f_s_norm)
        G_t = torch.einsum('bx,dx->bd', f_t_norm, f_t_norm)
        G_st = G_s * G_t

        # Norm before difference
        z_s = torch.trace(G_s)
        z_st = torch.trace(G_st)

        G_s = G_s / z_s
        G_st = G_st / z_st

        g_diff = G_s.pow(2) - G_st.pow(2)
        loss = g_diff.sum()
        return loss


class ITLossOptions:
    def __init__(self, s_dim, t_dim, n_data, alpha_it):
        self.s_dim = s_dim
        self.t_dim = t_dim
        self.n_data = n_data
        self.alpha_it = alpha_it


class KdITLoss(nn.Module):
    def __init__(self, s_dim, t_dim, n_data=1, alpha_it=1.0, mode="corr"):
        super().__init__()
        self.mode = mode
        self.it = ITLoss(ITLossOptions(s_dim=s_dim, t_dim=t_dim, n_data=n_data, alpha_it=alpha_it))

    def forward(self, teacher_feat, student_feat, importance=None):
        if self.mode == "corr":
            return self.it.forward_correlation_it(student_feat, teacher_feat)
        if self.mode == "mutual":
            return self.it.forward_mutual_it(student_feat, teacher_feat)
        raise ValueError(f"Unknown ITLoss mode: {self.mode}")


def cosine_cost_matrix(hT, hS):
    # C_ij = 1 - cos(hT_i, hS_j)
    hTn = F.normalize(hT, dim=1)
    hSn = F.normalize(hS, dim=1)
    cos = hTn @ hSn.t()
    return 1.0 - cos


def pose_cost_matrix(pose, trans_w=1.0, rot_w=1.0, eps=1e-12):
    # pose: [n, 6] -> translation L2 + rotation angle (rad)
    if pose is None:
        return None
    if pose.dim() != 2 or pose.size(1) != 6:
        raise ValueError(f"pose must be [n, 6], got {tuple(pose.shape)}")
    t = pose[:, :3]
    r = pose[:, 3:]
    t_dist = torch.cdist(t, t, p=2)

    q = qexp_t(r)
    q = F.normalize(q, dim=1)
    dot = torch.matmul(q, q.t()).abs().clamp(-1.0 + eps, 1.0 - eps)
    ang = 2.0 * torch.acos(dot)
    dist = trans_w * t_dist + rot_w * ang
    return dist


def sinkhorn_transport(C, u, v, eps, T=1, K=20):
    n = C.shape[0]
    A = torch.exp(-C / eps)
    pi = torch.ones((n, n), device=C.device, dtype=C.dtype)
    sigma = torch.ones(n, device=C.device, dtype=C.dtype) / n

    with torch.no_grad():
        for _ in range(T):
            Q = A * pi
            for _ in range(K):
                delta = u / (Q @ sigma + 1e-12)
                sigma = v / (Q.t() @ delta + 1e-12)
            pi = torch.diag(delta) @ Q @ torch.diag(sigma)
    return pi


class KdLCKTLoss(nn.Module):
    #

    def __init__(self, eps=0.05, T=1, K=20, alpha=1.0, trans_w=1.0, rot_w=1.0):
        super().__init__()
        self.eps = eps
        self.T = T
        self.K = K
        self.alpha = alpha
        self.trans_w = trans_w
        self.rot_w = rot_w

    def forward(self, teacher_feat, student_feat, pose=None):
        n = teacher_feat.shape[0]

        if n == 0:
            return torch.zeros((), device=student_feat.device)

        u = torch.ones(n, device=student_feat.device, dtype=student_feat.dtype) / n
        v = torch.ones(n, device=student_feat.device, dtype=student_feat.dtype) / n

        C_feat = cosine_cost_matrix(teacher_feat, student_feat)

        C_pose = pose_cost_matrix(pose, trans_w=self.trans_w, rot_w=self.rot_w)

        if C_pose is None:
            C = C_feat
        else:
            C_pose = C_pose / (C_pose.max() + 1e-12)

            C = self.alpha * C_feat + (1.0 - self.alpha) * C_pose

        pi = sinkhorn_transport(C.detach(), u, v, self.eps, T=self.T, K=self.K)

        W = (pi * C).sum()

        H = (pi * torch.log(pi + 1e-12)).sum()

        return W + self.eps * H



def ida_mask_1d(teacher_feat, student_feat, top_ratio=0.6):
    with torch.no_grad():
        diff = torch.abs(teacher_feat.detach() - student_feat.detach())
        k = max(1, int(round(diff.size(1) * top_ratio)))
        topk_idx = torch.topk(diff, k=k, dim=1, largest=True).indices
        mask = torch.zeros_like(diff)
        mask.scatter_(1, topk_idx, 1.0)
    return mask


def build_importance(teacher_feat, student_feat, kd_cfg):
    mode = kd_cfg.get("importance_mode", "ida_mask")
    if mode == "ida_mask":
        top_ratio = kd_cfg.get("mask_ratio", 0.6)
        return ida_mask_1d(teacher_feat, student_feat, top_ratio=top_ratio)
    diff = torch.abs(teacher_feat.detach() - student_feat.detach())
    if mode == "diff_soft":
        return _normalize_importance(diff)
    if mode == "diff_sigmoid":
        threshold = kd_cfg.get("kp_threshold", 0.0)
        tau = kd_cfg.get("kp_tau", 1.0)
        return torch.sigmoid((diff - threshold) / max(tau, 1e-6))
    if mode == "diff_topk":
        top_ratio = kd_cfg.get("mask_ratio", 0.6)
        return _topk_mask(diff, top_ratio)
    return None


def load_sigma_maps(name_batches, sigma_root, sigma_filename="sigma.npy", missing="error"):
    if name_batches is None:
        raise ValueError("sigma_map mode requires batch['name'] for lookup.")
    sigma_list = []
    for names in name_batches:
        frame_names = names if isinstance(names, (list, tuple)) else [names]
        for name in frame_names:
            base = os.path.splitext(os.path.basename(str(name)))[0]
            sigma_path = os.path.join(sigma_root, base, sigma_filename)
            if not os.path.exists(sigma_path):
                if missing == "error":
                    raise FileNotFoundError(f"Sigma map not found: {sigma_path}")
                if missing == "warn":
                    print(f"[sigma_map] missing: {sigma_path}, fallback to ones")
                    sigma = np.ones((1, 1), dtype=np.float32)
                else:
                    sigma = np.ones((1, 1), dtype=np.float32)
            else:
                sigma = np.load(sigma_path)
            sigma_list.append(torch.from_numpy(sigma).float())
    return torch.stack(sigma_list, dim=0)
