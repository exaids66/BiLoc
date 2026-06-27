import copy
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from hydra.utils import instantiate

from models.utils_quant import BinaryActivation, QuantizeLinear
from utils.embedding import PoseEmbedding, TimeStepEmbedding
from typing import Dict, List, Optional, Callable


logger = logging.getLogger(__name__)


class Denoiser(nn.Module):
    """
    扩散模型中的去噪器，使用二值化 Transformer 作为主干。
    组成：
        - 时间步嵌入：TimeStepEmbedding，将标量 t 映射到高维
        - 位姿嵌入：PoseEmbedding，将 6DoF 位姿编码
        - 视觉特征 z：来自图像特征提取器
        - Transformer 主干：BinarizedTransformerEncoder（由 Hydra 配置注入）
        - 输出 MLP：预测目标位姿残差
    主要输入/输出形状（B=batch, N=相机数/序列长度）：
        x: (B, N, target_dim)    当前带噪位姿
        t: (B,)                  对应扩散时间步
        z: (B, N, z_dim)         图像/上下文特征
        返回: (B, N, target_dim) 去噪后的位姿增量
    """
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

        # 构建二值化 Transformer 编码器（Hydra 会实例化配置里的 BinarizedTransformerEncoder）
        self._trunk = instantiate(TRANSFORMER, _recursive_=False)

        self._last = BinaryMLP(
            d_model, hidden_features=mlp_hidden_dim, out_features=target_dim
        )

    def forward(
        self,
        x: torch.Tensor,  # B x N x dim //18，3，6
        t: torch.Tensor,  # B //18
        z: torch.Tensor,  # B x N x dim_z //18.3.384
    ):
        B, N, _ = x.shape

        # 时间与位姿嵌入
        t_emb = self.time_embed(t)
        t_emb = t_emb.view(B, 1, t_emb.shape[-1]).expand(-1, N, -1)  # 广播到每个相机/节点
        x_emb = self.pose_embed(x)

        if self.pivot_cam_onehot:
            # 为第一个相机追加 one-hot 作为 pivot 标识
            cam_pivot_id = torch.zeros_like(z[..., :1])
            cam_pivot_id[:, 0, ...] = 1.0
            z = torch.cat([z, cam_pivot_id], dim=-1)

        # 拼接特征：位姿嵌入 + 时间嵌入 + 图像特征 (+ pivot one-hot)
        feed_feats = torch.cat([x_emb, t_emb, z], dim=-1)

        input_ = self._first(feed_feats)

        feats_ = self._trunk(input_)

        output = self._last(feats_)

        return output # 18，3，6


class BinarizedMultiheadSelfAttention(nn.Module):
    """Multi-head self-attention with elastic binarization on weights and activations."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        weight_bits: int = 1,
        input_bits: int = 1,
        quant_method: str = "elastic",
        clip_val: float = 2.5,
        learnable_scaling: bool = True,
        symmetric: bool = True,
        weight_layerwise: bool = True,
        input_layerwise: bool = True,
    ):
        super().__init__()
        assert (
            d_model % nhead == 0
        ), f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim**-0.5

        quant_kwargs = dict(
            clip_val=clip_val,
            weight_bits=weight_bits,
            input_bits=input_bits,
            weight_layerwise=weight_layerwise,
            input_layerwise=input_layerwise,
            weight_quant_method=quant_method,
            input_quant_method=quant_method,
            learnable=learnable_scaling,
            symmetric=symmetric,
            bias=True,
        )

        self.q_proj = QuantizeLinear(d_model, d_model, **quant_kwargs)
        self.k_proj = QuantizeLinear(d_model, d_model, **quant_kwargs)
        self.v_proj = QuantizeLinear(d_model, d_model, **quant_kwargs)
        self.out_proj = QuantizeLinear(d_model, d_model, **quant_kwargs)
        self.binary_act = BinaryActivation()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape

        q = self.binary_act(self.q_proj(x))
        k = self.binary_act(self.k_proj(x))
        v = self.binary_act(self.v_proj(x))

        q = q.view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.nhead, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            scores = scores + attn_mask

        if key_padding_mask is not None:
            padding_mask = key_padding_mask[:, None, None, :].to(dtype=scores.dtype)
            scores = scores.masked_fill(padding_mask.bool(), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = (
            context.transpose(1, 2).contiguous().view(B, N, self.d_model)
        )
        context = self.binary_act(context)
        return self.out_proj(context)


class BinarizedTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer using BiT-style two-set binarization."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = True,
        weight_bits: int = 1,
        input_bits: int = 1,
        quant_method: str = "elastic",
        clip_val: float = 2.5,
        learnable_scaling: bool = True,
        symmetric: bool = True,
        weight_layerwise: bool = True,
        input_layerwise: bool = True,
    ):
        super().__init__()

        self.self_attn = BinarizedMultiheadSelfAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            weight_bits=weight_bits,
            input_bits=input_bits,
            quant_method=quant_method,
            clip_val=clip_val,
            learnable_scaling=learnable_scaling,
            symmetric=symmetric,
            weight_layerwise=weight_layerwise,
            input_layerwise=input_layerwise,
        )

        quant_kwargs = dict(
            clip_val=clip_val,
            weight_bits=weight_bits,
            input_bits=input_bits,
            weight_layerwise=weight_layerwise,
            input_layerwise=input_layerwise,
            weight_quant_method=quant_method,
            input_quant_method=quant_method,
            learnable=learnable_scaling,
            symmetric=symmetric,
            bias=True,
        )

        self.linear1 = QuantizeLinear(d_model, dim_feedforward, **quant_kwargs)
        self.linear2 = QuantizeLinear(dim_feedforward, d_model, **quant_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = BinaryActivation()

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask)
            )
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class BinarizedTransformerEncoder(nn.Module):
    """Stacked binarized encoder layers."""

    def __init__(self, encoder_layer: BinarizedTransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
        return output


def TransformerEncoderWrapper(
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    norm_first: bool = True,
    weight_bits: int = 1,
    input_bits: int = 1,
    quant_method: str = "elastic",
    clip_val: float = 2.5,
    learnable_scaling: bool = True,
    symmetric: bool = True,
    weight_layerwise: bool = True,
    input_layerwise: bool = True,
):
    encoder_layer = BinarizedTransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        norm_first=norm_first,
        weight_bits=weight_bits,
        input_bits=input_bits,
        quant_method=quant_method,
        clip_val=clip_val,
        learnable_scaling=learnable_scaling,
        symmetric=symmetric,
        weight_layerwise=weight_layerwise,
        input_layerwise=input_layerwise,
    )
    return BinarizedTransformerEncoder(encoder_layer, num_encoder_layers)


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        input = input[0]
        # indicate_small = (input < -1).float()
        # indicate_big = (input > 1).float()
        indicate_leftmid = ((input >= -1.0) & (input <= 0)).float()
        indicate_rightmid = ((input > 0) & (input <= 1.0)).float()

        grad_input = (indicate_leftmid * (2 + 2*input) + indicate_rightmid * (2 - 2*input)) * grad_output.clone()
        return grad_input


class XnorScale(torch.autograd.Function):
    """
    XNOR style weight binarization: sign(w) * mean(|w|) over input dimension.
    For Linear weights with shape [out_features, in_features], the mean is
    taken over dim=1 and broadcast back.
    """

    @staticmethod
    def forward(ctx, w):
        ctx.save_for_backward(w)
        scale = w.abs().mean(dim=1, keepdim=True)
        return w.sign() * scale

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through on the sign; pass gradients unchanged.
        w, = ctx.saved_tensors
        scale = w.abs().mean(dim=1, keepdim=True)
        grad_w = grad_output * (scale > 0).float()
        return grad_w


def binary_activation(x):
    """Clamp to [-1, 1] then apply sign with STE."""
    x = x.clamp(-1, 1)
    return SignSTE.apply(x)


def xnor_weight(w):
    return XnorScale.apply(w)


class LearnableBias(nn.Module):
    """Per-channel learnable bias that broadcasts over batch/seq."""

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        # Works for (B, C) and (B, N, C) by broadcasting the middle dim.
        if x.dim() == 2:
            return x + self.bias.squeeze(1)
        return x + self.bias


class RPReLU(nn.Module):
    """
    Shifted PReLU used in the binary MLP block:
    bias -> PReLU -> bias.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.move1 = LearnableBias(dim)
        self.move2 = LearnableBias(dim)
        self.prelu = nn.PReLU(num_parameters=dim)

    def forward(self, x):
        x = self.move1(x)
        if x.dim() == 3:
            b, n, c = x.shape
            x = x.reshape(b * n, c)
            x = self.prelu(x)
            x = x.reshape(b, n, c)
        else:
            x = self.prelu(x)
        x = self.move2(x)
        return x


class BinaryLinear(nn.Module):
    """
    Linear layer with binary weights and activations (XNOR style).
    Weights: sign(w) * mean(|w|) along input dim.
    Activations: clamp to [-1, 1] then sign (STE).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        # Flatten last dim as features; supports (B, C) or (B, N, C).
        if x.dim() == 3:
            b, n, c = x.shape
            x_in = x.reshape(b * n, c)
        else:
            x_in = x

        a = binary_activation(x_in)
        w = xnor_weight(self.lin.weight)
        out = F.linear(a, w, self.lin.bias)

        if x.dim() == 3:
            out = out.reshape(b, n, -1)
        return out


def _match_channels(x, target_dim: int):
    """
    Lightweight channel matching used for the residual helper.
    If target_dim is a multiple of input dim, repeat; if input is a multiple
    of target, average groups; otherwise use adaptive average pooling.
    """
    in_dim = x.size(-1)
    if target_dim == in_dim:
        return x
    if target_dim % in_dim == 0:
        repeat = target_dim // in_dim
        return x.repeat_interleave(repeat, dim=-1)
    if in_dim % target_dim == 0:
        group = in_dim // target_dim
        new_shape = x.shape[:-1] + (target_dim, group)
        return x.reshape(new_shape).mean(dim=-1)
    # Fallback: pool along channel dimension.
    if x.dim() == 3:
        b, n, c = x.shape
        pooled = F.adaptive_avg_pool1d(x.reshape(b * n, 1, c), target_dim)
        return pooled.reshape(b, n, target_dim)
    pooled = F.adaptive_avg_pool1d(x.unsqueeze(1), target_dim)
    return pooled.squeeze(1)


class BinaryMLP(nn.Module):
    """
    Two-layer MLP using binary weights/activations.
    Includes per-branch learnable biases, RPReLU activations, and a lightweight
    shortcut that reuses the input features to enrich binary capacity.

    Args:
        in_features: input channel dimension
        hidden_features: hidden dimension (default 4x input)
        out_features: output dimension (default = in_features)
        drop: dropout probability
        use_shortcut: whether to add input-derived shortcut at each stage
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 128,
        out_features: int = 6,
        drop: float = 0.0,
        use_shortcut: bool = True,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.use_shortcut = use_shortcut

        self.move1 = LearnableBias(in_features)
        self.fc1 = BinaryLinear(in_features, hidden_features, bias=False)
        self.norm1 = nn.BatchNorm1d(hidden_features)
        self.act1 = RPReLU(hidden_features)

        self.move2 = LearnableBias(hidden_features)
        self.fc2 = BinaryLinear(hidden_features, out_features, bias=False)
        self.norm2 = nn.BatchNorm1d(out_features)
        self.act2 = RPReLU(out_features)

        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.in_features = in_features

    def _apply_norm(self, norm, x):
        if x.dim() == 3:
            b, n, c = x.shape
            x = norm(x.reshape(b * n, c)).reshape(b, n, c)
        else:
            x = norm(x)
        return x

    def forward(self, x):
        # First binary block
        shortcut1 = _match_channels(x, self.hidden_features) if self.use_shortcut else 0
        x1 = self.fc1(self.move1(x))
        x1 = self._apply_norm(self.norm1, x1)
        x1 = x1 + shortcut1
        x1 = self.act1(x1)
        x1 = self.drop(x1)

        # Second binary block
        shortcut2 = _match_channels(x1, self.out_features) if self.use_shortcut else 0
        x2 = self.fc2(self.move2(x1))
        x2 = self._apply_norm(self.norm2, x2)
        x2 = x2 + shortcut2
        x2 = self.act2(x2)
        x2 = self.drop(x2)
        return x2


