import torch
import torch.nn as nn
import math
from pytorch3d.renderer import HarmonicEmbedding
# try:
#     from pytorch3d.renderer import HarmonicEmbedding  # 优先用官方
# except ImportError:
#     class HarmonicEmbedding(nn.Module):
#         """
#         简化版 PyTorch3D HarmonicEmbedding，append_input=True，logspace=True。
#         x: (..., D) -> (..., D * (1 + 2 * n_harmonic_functions))
#         """
#
#         def __init__(self, n_harmonic_functions: int = 6,
#                      omega0: float = 0.5,
#                      append_input: bool = True):
#             super().__init__()
#             self.n_harmonic_functions = n_harmonic_functions
#             self.omega0 = omega0
#             self.append_input = append_input
#
#         def forward(self, x: torch.Tensor) -> torch.Tensor:
#             # x: (..., D)
#             freqs = self.omega0 * (2.0 ** torch.arange(
#                 self.n_harmonic_functions, device=x.device, dtype=x.dtype))
#             angles = x[..., None] * freqs  # (..., D, n)
#             sin = torch.sin(angles)
#             cos = torch.cos(angles)
#             parts = [sin, cos]
#             if self.append_input:
#                 parts.insert(0, x[..., None])
#             out = torch.cat(parts, dim=-1).reshape(*x.shape[:-1], -1)
#             return out


class TimeStepEmbedding(nn.Module):
    # learned from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/nn.py
    def __init__(self, dim=256, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        self.linear = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, dim // 2),
        )

        self.out_dim = dim // 2

    def _compute_freqs(self, half):
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )
        return freqs

    def forward(self, timesteps):
        half = self.dim // 2
        freqs = self._compute_freqs(half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )

        output = self.linear(embedding)
        return output


class PoseEmbedding(nn.Module):
    def __init__(self, target_dim, n_harmonic_functions=10, append_input=True):
        super().__init__()

        self._emb_pose = HarmonicEmbedding(
            n_harmonic_functions=n_harmonic_functions, append_input=append_input
        )

        self.out_dim = self._emb_pose.get_output_dim(target_dim)

    def forward(self, pose_encoding):
        e_pose_encoding = self._emb_pose(pose_encoding)
        return e_pose_encoding
