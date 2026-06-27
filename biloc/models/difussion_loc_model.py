"""
@author: Wen Li
@file: diffusion_loc_model.py
@time: 2023/9/18 20:44
pose_diffusion模型基础上进行的修改，没有本质区别
"""
import torch
import torch.nn as nn
import logging
from typing import Dict, Optional
from hydra.utils import instantiate

logger = logging.getLogger(__name__)


class DiffusionLocModel(nn.Module):
    def __init__(
            self,
            IMAGE_FEATURE_EXTRACTOR: Dict,
            DIFFUSER: Dict,
            DENOISER: Dict,
    ):
        """ Initializes a DiffusionLoc model
        Args:
            image_feature_extractor_cfg (Dict):
                Configuration for the image feature extractor.
            diffuser_cfg (Dict):
                Configuration for the diffuser.
            denoiser_cfg (Dict):
                Configuration for the denoiser.
        """

        super().__init__()

        # 根据配置字典实例化图像特征提取器（例如 RangeViT）
        # 使用 hydra.utils.instantiate / omegaconf 的 instantiate
        self.image_feature_extractor = instantiate(
            IMAGE_FEATURE_EXTRACTOR, _recursive_=False
        )

        # 实例化 diffusion 调度器 / 采样器（通常封装前向扩散 + 反向采样）
        self.diffuser = instantiate(DIFFUSER, _recursive_=False)

        # 实例化 denoiser（即扩散模型中的 UNet / Transformer 等噪声预测模型）
        denoiser = instantiate(DENOISER, _recursive_=False)

        # 将 denoiser 挂到 diffuser 上，让 diffuser 在内部调用 denoiser 进行噪声预测
        # 比如 diffuser.model(x_t, t, cond=z) 这样的调用
        self.diffuser.model = denoiser

        # denoiser 输出的 target 维度（比如 pose 的维度：6 维（3 平移 + 3 旋转））
        self.target_dim = denoiser.target_dim

    def forward(
            self,
            image: torch.Tensor,
            pose: Optional[torch.Tensor] = None,
            sampling_timesteps = 10,
            training=True,
    ):
        """
        Forward pass of the PoseDiffusionModel.

        Args:
            image (torch.Tensor):
                Input image tensor, BxNx5xHxW.
                - B: batch size
                - N: 每个 batch 中的帧数 / 视角数 / 序列长度
                - 5: 通道数（例如 range, intensity, height, etc.）
                - H, W: range image 的空间分辨率
            pose (Optional[CamerasBase], optional):
                Camera object. Defaults to None.
                这里作为扩散模型的条件之一，通常是 GT pose（训练时）或 None（采样时）。
            sampling_timesteps (int):
                用于 DDIM 采样时的步数（测试阶段使用）
            training (bool):
                True: 训练阶段，调用 diffuser 的 training forward
                False: 测试 / 推理阶段，进行采样

        Return:
            diffusion_results 或 dict，内含：
                - training=True: diffuser 的输出字典 + pred_pose + pred_mask
                - training=False: {"pred_pose", "pred_mask", "z"}
        """

        # 记录输入 image 的形状，便于后续 reshape
        shapelist = list(image.shape)  # [B, N, C, H, W] 18，3，5，32，512
        batch_size = len(image)        # B 18

        if training:
            # ==============================
            # 训练模式：使用 GT pose 做条件，训练扩散模型
            # ==============================

            # 将 (B, N, C, H, W) 拉平成 (B*N, C, H, W)，方便送入 2D backbone
            # 54，5，32，512
            reshaped_image = image.reshape(shapelist[0] * shapelist[1], *shapelist[2:])
            # 图像特征提取：
            # z: 全局/patch 聚合后的 pose 条件特征   [B*N, C_feat]
            # pred_mask: 对应的 mask / 注意力图    [B*N, 1, H', W'] 或类似形状
            # z：54，384  //  pred_mask: 54,1,32,512
            # z, pred_mask = self.image_feature_extractor(reshaped_image)  # [B*N, C_feat], [B*N, 1, h, w]
            z = self.image_feature_extractor(reshaped_image)  # [B*N, C_feat], [B*N, 1, h, w]

            z_out4distil = z

            # 再把 z reshape 回 (B, N, C_feat)，和扩散模型的条件形状对齐
            # z: 18,3,384
            z = z.reshape(batch_size, shapelist[1], -1)  # [B, N, C_feat]

            # pred_mask: squeeze 掉通道维，reshape 回 (B*N, H, W)，通常是 SOAP mask 或 range map mask
            # pred_mask: 54,32,512
            # pred_mask = pred_mask.squeeze(1).reshape(
            #     shapelist[0] * shapelist[1],
            #     shapelist[-2],
            #     shapelist[-1]
            # )  # [B*N, H, W]，例如 [B*N, 32, 512]

            # 调用 diffuser 的 forward，使用 GT pose 和条件特征 z
            # 典型做法：diffuser(pose, z=z) -> 返回一个包含 loss、x_t、x_0_pred 等的字典
            # 包括diffloss, noise, x_0_pred, x_t, t
            diffusion_results = self.diffuser(pose, z=z)

            # 为统一接口，显式添加最终预测的 pose（通常是 x_0 的预测）
            diffusion_results['pred_pose'] = diffusion_results["x_0_pred"]

            # 同时把 pred_mask 加进去，方便可视化 或 损失设计（如 SOAP）
            # diffusion_results['pred_mask'] = pred_mask

            diffusion_results['z_out4distil'] = z_out4distil

            # diffloss // noise:18,3,6 // x_0_pred:18,3,6 // x_t:18,3,6 // t:16 // pred_pose: 18,3,6 // pred_mask: 54,32,512
            return diffusion_results

        else:
            # ==============================
            # 测试 / 推理模式：不用 GT pose，直接从噪声采样 pose
            # ==============================

            # 同样先把 image 拉成 (B*N, C, H, W)
            reshaped_image = image.reshape(shapelist[0] * shapelist[1], *shapelist[2:])

            # 用图像特征提取器提取条件特征 z 和 mask
            z = self.image_feature_extractor(reshaped_image)

            z_out4distil = z

            # z reshape 回 (B, N, C_feat)
            z = z.reshape(batch_size, shapelist[1], -1)


            # B: batch size, N: 每个样本中帧数 / 视角数
            B, N, _ = z.shape

            # 扩散模型目标的形状：通常为 [B, N, target_dim]，例如 target_dim=6 对应 SE(3) pose
            target_shape = [B, N, self.target_dim]

            # ------------------------------
            # 采样阶段
            # ------------------------------

            # 如果是 DDPM 采样（多步随机采样）：
            # pred_pose, pred_pose_diffusion_samples = self.diffuser.sample(shape=target_shape, z=z)

            # 这里使用 DDIM 采样（确定性/半确定性、步数更少的采样方法）
            pred_pose, _ = self.diffuser.ddim_sample(
                shape=target_shape,
                z=z,
                sampling_timesteps=sampling_timesteps
            )

            # 汇总推理结果
            diffusion_results = {
                "pred_pose": pred_pose,   # [B, N, target_dim]，例如 [B, N, 6]
                "z": z                    # 条件特征 [B, N, C_feat]
            }

            return diffusion_results
