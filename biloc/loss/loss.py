import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pose_util import qexp_t


def _normalize_importance(weight, eps=1e-6):
    # 将重要性权重归一化到 [0, 1]，按样本维度分别归一化
    if weight is None:
        return None
    dims = list(range(1, weight.dim()))
    w_min = weight.amin(dim=dims, keepdim=True)
    w_max = weight.amax(dim=dims, keepdim=True)
    return (weight - w_min) / (w_max - w_min + eps)


def _topk_mask(weight, top_ratio):
    # 对每个样本取 top-k 权重位置，生成二值 mask
    flat = weight.view(weight.size(0), -1)
    k = max(1, int(round(flat.size(1) * top_ratio)))
    topk_idx = torch.topk(flat, k=k, dim=1, largest=True).indices
    mask = torch.zeros_like(flat)
    mask.scatter_(1, topk_idx, 1.0)
    return mask.view_as(weight)


class EntropyLoss(nn.Module):
    # 参考 IDa-Det 的 entropy-loss：在 mask 约束范围内对特征差异加权
    def __init__(self, init_num=1.000001):
        super().__init__()
        self.sigma = nn.Parameter(self._init_sigma(init_num))

    @staticmethod
    def _init_sigma(init_num):
        return torch.full(size=(1, 1, 1, 1), fill_value=init_num, dtype=torch.float32)

    def forward(self, teacher_feat, student_feat, importance=None):
        # teacher_feat/student_feat: [B, C] 的 1D 特征
        # importance: 与特征同形状的权重/掩码
        if importance is None:
            importance = torch.ones_like(teacher_feat)
        # 归一化系数，避免有效区域过少导致数值偏大
        norm = max(1.0, importance.sum())
        diff = teacher_feat - student_feat
        # 基础 MSE 项
        mse = torch.pow(diff, 2) * importance
        # 类高斯分布的相关项（以 sigma 控制）
        corre = torch.pow(self.sigma, -2) * torch.pow(diff, 2) * importance
        # det 项：鼓励 sigma 不要过小
        det = torch.log(torch.pow(self.sigma, 2))
        return (mse + corre + 1e-3 * det).sum() / norm


# class EntropyLoss(nn.Module):
#     # 将 MSE 替换为 Smooth L1（Huber）
#     def __init__(self, init_num=1.000001, beta=1.0):
#         super().__init__()
#         self.sigma = nn.Parameter(self._init_sigma(init_num))
#         self.beta = beta  # Smooth L1 的转折点（PyTorch 的 smooth_l1_loss 参数）
#
#     @staticmethod
#     def _init_sigma(init_num):
#         return torch.full(size=(1, 1, 1, 1), fill_value=init_num, dtype=torch.float32)
#
#     def forward(self, teacher_feat, student_feat, importance=None):
#         # teacher_feat/student_feat: [B, C] 的 1D 特征
#         # importance: 与特征同形状的权重/掩码
#         if importance is None:
#             importance = torch.ones_like(teacher_feat)
#
#         # 归一化系数，避免有效区域过少导致数值偏大
#         norm = importance.sum().clamp_min(1.0)
#
#         diff = teacher_feat - student_feat  # [B, C]
#
#         # Smooth L1（逐元素，不做reduction）
#         s1 = F.smooth_l1_loss(
#             teacher_feat, student_feat,
#             beta=self.beta,
#             reduction="none"
#         )  # [B, C]
#
#         # 基础 Smooth L1 项（带 importance）
#         base = s1 * importance
#
#         # 类高斯相关项：把原来的 diff^2 换成 Smooth L1 形式，再乘 sigma^{-2}
#         corre = torch.pow(self.sigma, -2) * s1 * importance
#
#         # det 项：鼓励 sigma 不要过小
#         det = torch.log(torch.pow(self.sigma, 2))
#
#         return (base + corre + 1e-3 * det).sum() / norm



class KdWeightedMSELoss(nn.Module):
    # 方案1：软权重加权蒸馏
    def __init__(self, threshold=0.0, tau=1.0, use_sigmoid=True, eps=1e-6):
        super().__init__()
        self.threshold = threshold
        self.tau = tau
        self.use_sigmoid = use_sigmoid
        self.eps = eps

    def forward(self, teacher_feat, student_feat, importance=None):
        # importance 为软权重，来自差异或其它重要性估计
        diff = teacher_feat - student_feat
        if importance is None:
            return torch.mean(diff ** 2)
        if self.use_sigmoid:
            # sigmoid 形成软阈值：重要区域权重更高
            weight = torch.sigmoid((importance - self.threshold) / max(self.tau, self.eps))
        else:
            # 直接归一化重要性到 [0, 1]
            weight = _normalize_importance(importance, eps=self.eps)
        weighted = (diff ** 2) * weight
        return weighted.sum() / (weight.sum() + self.eps)


class KdHardMaskMSELoss(nn.Module):
    # 方案2：阈值化 mask 蒸馏
    def __init__(self, threshold=0.5, top_ratio=None, eps=1e-6):
        super().__init__()
        self.threshold = threshold
        self.top_ratio = top_ratio
        self.eps = eps

    def forward(self, teacher_feat, student_feat, importance=None):
        # hard mask: 重要区域为 1，不重要为 0
        diff = teacher_feat - student_feat
        if importance is None:
            importance = torch.abs(diff.detach())
        if self.top_ratio is not None:
            mask = _topk_mask(importance, self.top_ratio)
        else:
            mask = (importance > self.threshold).float()
        return ((diff ** 2) * mask).sum() / (mask.sum() + self.eps)


class KdTopKSampleLoss(nn.Module):
    # 方案3：Top-K 重采样蒸馏
    def __init__(self, top_ratio=0.3, eps=1e-6):
        super().__init__()
        self.top_ratio = top_ratio
        self.eps = eps

    def forward(self, teacher_feat, student_feat, importance=None):
        # 仅使用 top-k 的位置计算蒸馏损失
        diff = teacher_feat - student_feat
        if importance is None:
            importance = torch.abs(diff.detach())
        mask = _topk_mask(importance, self.top_ratio)
        return ((diff ** 2) * mask).sum() / (mask.sum() + self.eps)


class KdDualBranchLoss(nn.Module):
    # 方案4：全局蒸馏 + 重要区域蒸馏
    def __init__(self, alpha=1.0, threshold=0.0, tau=1.0):
        super().__init__()
        self.alpha = alpha
        self.weighted = KdWeightedMSELoss(threshold=threshold, tau=tau, use_sigmoid=True)

    def forward(self, teacher_feat, student_feat, importance=None):
        # 全局 MSE + 局部加权 MSE，兼顾整体与关键区域
        diff = teacher_feat - student_feat
        global_loss = torch.mean(diff ** 2)
        weighted_loss = self.weighted(teacher_feat, student_feat, importance)
        return global_loss + self.alpha * weighted_loss


class KdAdaptiveTemperatureLoss(nn.Module):
    # 方案5：按重要性调节温度/尺度
    def __init__(self, t_min=0.5, t_max=2.0, eps=1e-6):
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        self.eps = eps

    def forward(self, teacher_feat, student_feat, importance=None):
        # temp 越小，约束越强；temp 越大，约束越弱
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
    # 使用离线 sigma-map 对 1D 特征蒸馏做加权
    def __init__(self, invert=True, normalize=True, eps=1e-6):
        super().__init__()
        self.invert = invert
        self.normalize = normalize
        self.eps = eps

    def forward(self, teacher_feat, student_feat, sigma_map):
        # sigma_map: [B, H, W] 或 [H, W]，这里先规整为 [B, 1, H, W]
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
            # sigma 越小越重要，因此使用 1/sigma
            weight = 1.0 / (weight + self.eps)
        if self.normalize:
            # 归一化到 [0, 1]，避免不同样本尺度差异
            dims = list(range(1, weight.dim()))
            w_min = weight.amin(dim=dims, keepdim=True)
            w_max = weight.amax(dim=dims, keepdim=True)
            weight = (weight - w_min) / (w_max - w_min + self.eps)
        # 将 2D 权重压缩为每个样本一个标量权重
        weight = weight.mean(dim=(1, 2, 3))

        diff = teacher_feat - student_feat
        mse_per = torch.mean(diff ** 2, dim=1)
        return (mse_per * weight).sum() / (weight.sum() + self.eps)


class Embed(nn.Module):
    # 简单线性投影，用于将学生特征映射到教师维度
    def __init__(self, s_dim, t_dim, n=1):
        super().__init__()
        self.proj = nn.Linear(s_dim, t_dim)

    def forward(self, x):
        return self.proj(x)


class ITLoss(nn.Module):
    """Information-theoretic Loss function"""
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
    # 与 ITLoss 兼容的轻量配置对象
    def __init__(self, s_dim, t_dim, n_data, alpha_it):
        self.s_dim = s_dim
        self.t_dim = t_dim
        self.n_data = n_data
        self.alpha_it = alpha_it


class KdITLoss(nn.Module):
    # ITLoss 的蒸馏封装，忽略 importance
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
    # 迭代求解最优传输计划，按论文建议停止梯度
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
    # LCKT 结构蒸馏损失（基于 Sinkhorn 的最优传输 OT）
    # 核心思想：
    # - 把 teacher_feat 看成一组“点”(n 个样本的特征)
    # - 把 student_feat 看成另一组“点”
    # - 用代价矩阵 C(i,j) 衡量 teacher 第 i 个点 与 student 第 j 个点 的不匹配程度
    # - 用 Sinkhorn 求一个传输计划 pi(i,j)（近似最优传输的耦合矩阵）
    # - 最终 loss = <pi, C> + eps * <pi, log pi>（熵正则的 OT 目标）
    #
    # 这里还额外引入了 pose 结构代价 C_pose，用于引导 “谁和谁应该匹配”。

    def __init__(self, eps=0.05, T=1, K=20, alpha=1.0, trans_w=1.0, rot_w=1.0):
        super().__init__()
        self.eps = eps          # Sinkhorn 的熵正则强度（越大越“软”、越平均；越小越接近硬匹配/置换）
        self.T = T              # 可能给 Sinkhorn 内部用的温度/缩放（取决于 sinkhorn_transport 实现）
        self.K = K              # Sinkhorn 迭代次数（越大越收敛但越慢）
        self.alpha = alpha      # 融合代价权重：C = alpha*C_feat + (1-alpha)*C_pose
        self.trans_w = trans_w  # pose 代价里平移项权重
        self.rot_w = rot_w      # pose 代价里旋转项权重

    def forward(self, teacher_feat, student_feat, pose=None):
        # teacher_feat: [n, d]，student_feat: [n, d]（通常 n=batch_size）
        n = teacher_feat.shape[0]

        # 空 batch 直接返回 0（保持设备一致）
        if n == 0:
            return torch.zeros((), device=student_feat.device)

        # u, v 是 OT 的边缘分布（marginals），这里设为均匀分布：每个点质量都是 1/n
        # 表示 teacher 的 n 个点总质量=1，student 的 n 个点总质量=1
        u = torch.ones(n, device=student_feat.device, dtype=student_feat.dtype) / n
        v = torch.ones(n, device=student_feat.device, dtype=student_feat.dtype) / n

        # 1) 特征代价矩阵：C_feat[i,j] 表示 teacher_i 和 student_j 的特征不相似程度
        #    这里用 cosine_cost_matrix（常见定义：1 - cos_sim，越小越像）
        #    得到 [n, n] 的矩阵
        C_feat = cosine_cost_matrix(teacher_feat, student_feat)

        # 2) 姿态代价矩阵：C_pose[i,j] 表示 i 和 j 在 pose 上是否“结构相近”
        #    注意：这里的 pose_cost_matrix(pose) 需要返回 [n, n] 或 None
        #    - 如果 pose=None 或不支持，返回 None => 只用特征代价
        #    - 否则返回 pose 的 pairwise distance（例如平移距离+旋转距离）
        C_pose = pose_cost_matrix(pose, trans_w=self.trans_w, rot_w=self.rot_w)

        if C_pose is None:
            # 没有姿态结构约束，只用特征代价
            C = C_feat
        else:
            # 对 pose 代价做归一化，避免量纲/尺度压过特征项
            # 用 max 归一化（有时也会用 mean/std 或分位数）
            C_pose = C_pose / (C_pose.max() + 1e-12)

            # 融合代价：alpha 越大越偏向特征匹配；越小越偏向 pose 结构匹配
            C = self.alpha * C_feat + (1.0 - self.alpha) * C_pose

        # 3) Sinkhorn 求传输计划 pi（耦合矩阵）
        #    关键点：这里用了 C.detach()
        #    => 求 pi 的过程不反传梯度（把 pi 当常量）
        #    好处：更稳定、更省显存/时间；坏处：不是“严格的可微 OT”，梯度只通过后面的 W 走
        pi = sinkhorn_transport(C.detach(), u, v, self.eps, T=self.T, K=self.K)

        # 4) OT 主项：W = <pi, C> = sum_{i,j} pi_ij * C_ij
        #    因为 pi 被当成常量，所以对 student_feat 的梯度来自 dC/dstudent_feat，
        #    形式上就是一个“加权的匹配损失”（权重=pi）
        W = (pi * C).sum()

        # 5) 熵正则项：H = <pi, log pi>
        #    经典 entropic OT 的目标是 <pi,C> + eps*<pi, log pi>
        #    但注意：由于 pi 来自 C.detach()，这里 H 对 student_feat 也基本没有梯度（几乎是常量项）
        #    所以它更多只是让 loss 数值符合“OT形式”，对训练推动不大（除非你让 pi 可微）
        H = (pi * torch.log(pi + 1e-12)).sum()

        # 最终 loss
        return W + self.eps * H



def ida_mask_1d(teacher_feat, student_feat, top_ratio=0.6):
    # IDA-like mask（1D 特征版）：
    # 在通道维上选出差异最大的 top_ratio 通道，作为蒸馏关注区域
    with torch.no_grad():
        # diff: [B, C]，仅用于选通道，不参与反向传播
        diff = torch.abs(teacher_feat.detach() - student_feat.detach())
        k = max(1, int(round(diff.size(1) * top_ratio)))
        topk_idx = torch.topk(diff, k=k, dim=1, largest=True).indices
        mask = torch.zeros_like(diff)
        mask.scatter_(1, topk_idx, 1.0)
    return mask


def build_importance(teacher_feat, student_feat, kd_cfg):
    # 根据配置构造 importance 权重/掩码
    mode = kd_cfg.get("importance_mode", "ida_mask")
    if mode == "ida_mask":
        top_ratio = kd_cfg.get("mask_ratio", 0.6)
        return ida_mask_1d(teacher_feat, student_feat, top_ratio=top_ratio)
    diff = torch.abs(teacher_feat.detach() - student_feat.detach())
    if mode == "diff_soft":
        # 直接使用差异的归一化值作为软权重
        return _normalize_importance(diff)
    if mode == "diff_sigmoid":
        # 通过 sigmoid 形成软阈值权重
        threshold = kd_cfg.get("kp_threshold", 0.0)
        tau = kd_cfg.get("kp_tau", 1.0)
        return torch.sigmoid((diff - threshold) / max(tau, 1e-6))
    if mode == "diff_topk":
        # 仅保留 top-k 通道作为重要区域,在这里和ida_mask没有区别。都是硬选择通道
        top_ratio = kd_cfg.get("mask_ratio", 0.6)
        return _topk_mask(diff, top_ratio)
    return None


def load_sigma_maps(name_batches, sigma_root, sigma_filename="sigma.npy", missing="error"):
    # 按帧名批量加载离线 sigma-map，返回 [B*N, H, W]
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
