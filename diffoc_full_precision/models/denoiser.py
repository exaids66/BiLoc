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
        x: torch.Tensor,  # B x N x dim //18，3，6
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

        return output # 18，3，6


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
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional):
            Norm layer that will be stacked on top of the convolution layer.
            If ``None`` this layer wont be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional):
            Activation function which will be stacked on top of the
            normalization layer (if not None), otherwise on top of the
            conv layer. If ``None`` this layer wont be used.
            Default: ``torch.nn.ReLU``
        inplace (bool): Parameter for the activation layer, which can
            optionally do the operation in-place. Default ``True``
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,                     # 输入特征维度（MLP 第一层的输入维度）
        hidden_channels: List[int],          # 每一层 hidden 的维度列表，最后一个元素是输出维度
        norm_layer: Optional[
            Callable[..., torch.nn.Module]
        ] = None,                            # 归一化层类型（如 nn.LayerNorm / nn.BatchNorm1d），为 None 则不使用
        activation_layer: Optional[
            Callable[..., torch.nn.Module]
        ] = torch.nn.ReLU,                   # 激活函数类型（默认 ReLU），传的是 class 而不是实例
        # ] = nn.LeakyReLU,
        inplace: Optional[bool] = True,      # 传给激活函数 / Dropout 的 inplace 参数
        bias: bool = True,                   # Linear 层是否使用 bias
        norm_first: bool = False,            # True: Norm -> Linear；False: Linear -> Norm
        dropout: float = 0.0,                # Dropout 概率，为 0 则不加 Dropout
    ):
        # 这段注释来源：
        # `norm_layer` 的设计参考了 TorchMultimodal 的 MLP 实现

        # 构造传入激活函数/Dropout 的参数字典：
        # 如果 inplace 不是 None，就传 {"inplace": inplace}，
        # 否则传一个空 dict（即不设置 inplace 参数）
        params = {} if inplace is None else {"inplace": inplace}

        # 用来存放顺序堆叠的层
        layers = []
        in_dim = in_channels  # 当前层的输入维度，初始化为整体 MLP 的输入维度

        # 这里处理 hidden_channels 中 **除了最后一个元素** 的所有 hidden_dim
        # 这些对应中间层： [in_dim -> hidden_dim1 -> hidden_dim2 -> ...]
        for hidden_dim in hidden_channels[:-1]:
            # 如果 norm_first=True，并且提供了 norm_layer：
            # 在 Linear 之前先做归一化（Norm -> Linear）
            if norm_first and norm_layer is not None:
                layers.append(norm_layer(in_dim))

            # 添加全连接层：Linear(in_dim -> hidden_dim)
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))

            # 如果 norm_first=False，并且提供了 norm_layer：
            # 在 Linear 之后做归一化（Linear -> Norm）
            if not norm_first and norm_layer is not None:
                layers.append(norm_layer(hidden_dim))

            # 添加激活函数层，例如 ReLU/LeakyReLU：
            # activation_layer 是一个类，这里通过 **params 传入 inplace 参数
            layers.append(activation_layer(**params))

            # 如果设置了 dropout > 0，则在激活后添加 Dropout
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout, **params))

            # 更新下一层的输入维度
            in_dim = hidden_dim

        # 到这里为止，已经构建完所有“中间层”
        # 接下来处理最后一层（hidden_channels[-1]）：

        # 如果 norm_first=True，最后一层前也可以先做一次 Norm
        if norm_first and norm_layer is not None:
            layers.append(norm_layer(in_dim))

        # 最后一层 Linear：从 in_dim -> hidden_channels[-1]
        # 这通常就是整个 MLP 的输出维度
        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))

        # 末尾如果指定了 dropout，也可以再加一次 Dropout
        # 注意：这里没有再加激活，相当于最后一层是“线性输出层”
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout, **params))

        # 调用父类 torch.nn.Sequential 的构造函数，
        # 把 layers 中的模块按顺序展开传进去
        super().__init__(*layers)
