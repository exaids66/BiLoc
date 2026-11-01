import logging
from typing import Dict, List, Optional, Callable
from utils.embedding import TimeStepEmbedding, PoseEmbedding

import torch
import torch.nn as nn

from hydra.utils import instantiate


logger = logging.getLogger(__name__)


class Denoiser(nn.Module):
    def __init__(
        self,
        TRANSFORMER: Dict,
        target_dim: int = 6,  # pose shape
        pivot_cam_onehot: bool = True,
        z_dim: int = 384,
        mlp_hidden_dim: bool = 128,
    ):
        super().__init__()

        self.pivot_cam_onehot = pivot_cam_onehot
        self.target_dim = target_dim

        self.time_embed = TimeStepEmbedding()
        self.pose_embed = PoseEmbedding(target_dim=self.target_dim)

        first_dim = (
            self.time_embed.out_dim
            + self.pose_embed.out_dim
            + z_dim
            + int(self.pivot_cam_onehot)
        )

        d_model = TRANSFORMER.d_model

        self._first = nn.Linear(first_dim, d_model)

        # call TransformerEncoderWrapper() to build a encoder-only transformer
        self._trunk = instantiate(TRANSFORMER, _recursive_=False)

        self._last = MLP(
            d_model,
            [mlp_hidden_dim, self.target_dim],
            norm_layer=nn.LayerNorm,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # B //18
        z: torch.Tensor,  # B x N x dim_z //18.3.384
    ):
        B, N, _ = x.shape

        t_emb = self.time_embed(t)
        # expand t from B x C to B x N x C
        t_emb = t_emb.view(B, 1, t_emb.shape[-1]).expand(-1, N, -1)

        x_emb = self.pose_embed(x)

        if self.pivot_cam_onehot:
            # add the one hot vector identifying the first camera as pivot
            cam_pivot_id = torch.zeros_like(z[..., :1])
            cam_pivot_id[:, 0, ...] = 1.0
            z = torch.cat([z, cam_pivot_id], dim=-1)

        feed_feats = torch.cat([x_emb, t_emb, z], dim=-1)

        input_ = self._first(feed_feats)

        feats_ = self._trunk(input_)

        output = self._last(feats_)

        return output


def TransformerEncoderWrapper(
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    norm_first: bool = True,
    batch_first: bool = True,
):
    encoder_layer = torch.nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=batch_first,
        norm_first=norm_first,
    )

    _trunk = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers)
    return _trunk


class MLP(torch.nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[
            Callable[..., torch.nn.Module]
        ] = None,
        activation_layer: Optional[
            Callable[..., torch.nn.Module]
        ] = torch.nn.ReLU,
        # ] = nn.LeakyReLU,
        inplace: Optional[bool] = True,
        bias: bool = True,
        norm_first: bool = False,
        dropout: float = 0.0,
    ):

        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels

        for hidden_dim in hidden_channels[:-1]:
            if norm_first and norm_layer is not None:
                layers.append(norm_layer(in_dim))

            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))

            if not norm_first and norm_layer is not None:
                layers.append(norm_layer(hidden_dim))

            layers.append(activation_layer(**params))

            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout, **params))

            in_dim = hidden_dim


        if norm_first and norm_layer is not None:
            layers.append(norm_layer(in_dim))

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))

        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
