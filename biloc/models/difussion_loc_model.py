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

        super().__init__()

        self.image_feature_extractor = instantiate(
            IMAGE_FEATURE_EXTRACTOR, _recursive_=False
        )

        self.diffuser = instantiate(DIFFUSER, _recursive_=False)

        denoiser = instantiate(DENOISER, _recursive_=False)

        self.diffuser.model = denoiser

        self.target_dim = denoiser.target_dim

    def forward(
            self,
            image: torch.Tensor,
            pose: Optional[torch.Tensor] = None,
            sampling_timesteps = 10,
            training=True,
    ):

        shapelist = list(image.shape)
        batch_size = len(image)        # B 18

        if training:
            # ==============================
            # ==============================

            reshaped_image = image.reshape(shapelist[0] * shapelist[1], *shapelist[2:])
            # z, pred_mask = self.image_feature_extractor(reshaped_image)  # [B*N, C_feat], [B*N, 1, h, w]
            z = self.image_feature_extractor(reshaped_image)  # [B*N, C_feat], [B*N, 1, h, w]

            z_out4distil = z

            # z: 18,3,384
            z = z.reshape(batch_size, shapelist[1], -1)  # [B, N, C_feat]

            # pred_mask: 54,32,512
            # pred_mask = pred_mask.squeeze(1).reshape(
            #     shapelist[0] * shapelist[1],
            #     shapelist[-2],
            #     shapelist[-1]

            diffusion_results = self.diffuser(pose, z=z)

            diffusion_results['pred_pose'] = diffusion_results["x_0_pred"]

            # diffusion_results['pred_mask'] = pred_mask

            diffusion_results['z_out4distil'] = z_out4distil

            # diffloss // noise:18,3,6 // x_0_pred:18,3,6 // x_t:18,3,6 // t:16 // pred_pose: 18,3,6 // pred_mask: 54,32,512
            return diffusion_results

        else:
            # ==============================
            # ==============================

            reshaped_image = image.reshape(shapelist[0] * shapelist[1], *shapelist[2:])

            z = self.image_feature_extractor(reshaped_image)

            z_out4distil = z

            z = z.reshape(batch_size, shapelist[1], -1)


            B, N, _ = z.shape

            target_shape = [B, N, self.target_dim]

            # ------------------------------
            # ------------------------------

            # pred_pose, pred_pose_diffusion_samples = self.diffuser.sample(shape=target_shape, z=z)

            pred_pose, _ = self.diffuser.ddim_sample(
                shape=target_shape,
                z=z,
                sampling_timesteps=sampling_timesteps
            )

            diffusion_results = {
                "pred_pose": pred_pose,
                "z": z
            }

            return diffusion_results
