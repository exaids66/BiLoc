# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import logging
import math
import torch.nn.functional as F

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_chn), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class ElasticQuantBinarizerSigned(torch.autograd.Function):
    """
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001).float().to(alpha.device)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=True, init_method='default')
        alpha = torch.where(alpha > eps, alpha, eps)
        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)

        grad_scale = 1.0 / math.sqrt(input.numel()) if not Qp else 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        if ctx.num_bits == 1:
            grad_alpha = ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        else:
            grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                    -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class ElasticQuantBinarizerUnsigned(torch.autograd.Function):
    """
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        Qn = 0
        Qp = 2 ** (num_bits) - 1
        if num_bits == 1:
            input_ = input
        else:
            min_val = input.min().item()
            input_ = input - min_val

        eps = torch.tensor(0.00001).float().to(alpha.device)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=False, init_method='default')
        alpha = torch.where(alpha > eps, alpha, eps)
        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)

        grad_scale = 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input_, alpha)
        ctx.other = grad_scale, Qn, Qp
        q_w = (input_ / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        if num_bits != 1:
            w_q = w_q + min_val
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big   # this is more cpu-friendly than torch.ones(input_.shape)
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None

class AlphaInit(nn.Parameter):
    def __new__(cls, tensor, requires_grad=True):
        # Align signature with nn.Parameter so deepcopy can pass (data, requires_grad)
        return super().__new__(cls, data=tensor, requires_grad=requires_grad)

    def __init__(self, tensor, requires_grad=True):
        super().__init__()
        self.initialized = False

    def _initialize(self, init_tensor):
        assert not self.initialized, 'already initialized.'
        self.data.copy_(init_tensor)
        self.initialized = True

    def initialize_wrapper(self, tensor, num_bits, symmetric, init_method='default'):
        Qp = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** (num_bits) - 1
        if Qp == 0:
            Qp = 1.0
        if init_method == 'default':
            init_val = 2 * tensor.abs().mean() / math.sqrt(Qp) if symmetric \
                else 4 * tensor.abs().mean() / math.sqrt(Qp)
        elif init_method == 'uniform':
            init_val = 1./(2*Qp+1) if symmetric else 1./Qp

        self._initialize(init_val)


class BwnQuantizer(torch.autograd.Function):
    """Binary Weight Network (BWN)
     Ref: https://arxiv.org/abs/1603.05279
     """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param input: tensor to be binarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input)
        if layerwise:
            s = input.size()
            m = input.norm(p=1).div(input.nelement())
            e = input.mean()
            result = (input-e).sign().mul(m.expand(s))
        else:
            n = input[0].nelement()  # W of size axb, return a vector of  ax1
            s = input.size()
            m = input.norm(1, 1, keepdim=True).div(n)
            e = input.mean()
            result = (input-e).sign().mul(m.expand(s))

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None


def act_quant_fn(input, clip_val, num_bits, symmetric, quant_method, layerwise):
    if num_bits == 32:
        return input
    elif quant_method == "bwn" and num_bits == 1:
        quant_fn = BwnQuantizer
    elif quant_method == "elastic" and num_bits >= 1 and symmetric:
        quant_fn = ElasticQuantBinarizerSigned
    elif quant_method == "elastic" and num_bits >= 1 and not symmetric:
        quant_fn = ElasticQuantBinarizerUnsigned
    else:
        raise ValueError("Unknownquant_method")

    input = quant_fn.apply(input, clip_val, num_bits, layerwise)

    return input


def weight_quant_fn(weight,  clip_val,  num_bits,  symmetric, quant_method, layerwise):
    if num_bits == 32:
        return weight
    elif quant_method == "bwn" and num_bits == 1:
        quant_fn = BwnQuantizer
    elif quant_method == "elastic" and num_bits >= 1 and symmetric:
        quant_fn = ElasticQuantBinarizerSigned
    elif quant_method == "elastic" and num_bits >= 1 and not symmetric:
        quant_fn = ElasticQuantBinarizerUnsigned
    else:
        raise ValueError("Unknown quant_method")

    weight = quant_fn.apply(weight, clip_val,  num_bits, layerwise)
    return weight


class QuantizeLinear(nn.Linear):

    def __init__(self, *kargs, clip_val=2.5, weight_bits=8, input_bits=8, learnable=False, symmetric=True,
                 weight_layerwise=True, input_layerwise=True, weight_quant_method="twn", input_quant_method="uniform",
                 **kwargs):
        super(QuantizeLinear, self).__init__(*kargs, **kwargs)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.learnable = learnable
        self.symmetric = symmetric
        self.weight_layerwise = weight_layerwise
        self.input_layerwise = input_layerwise
        self.weight_quant_method = weight_quant_method
        self.input_quant_method = input_quant_method
        self._build_weight_clip_val(weight_quant_method, learnable, init_val=clip_val)
        self._build_input_clip_val(input_quant_method, learnable, init_val=clip_val)
        self.move = LearnableBias(self.weight.shape[1])

    def _build_weight_clip_val(self, quant_method, learnable, init_val):
        if quant_method == 'uniform':
            # init_val = self.weight.mean().item() + 3 * self.weight.std().item()
            self.register_buffer('weight_clip_val', torch.tensor([-init_val, init_val]))
            if learnable:
                self.weight_clip_val = nn.Parameter(self.weight_clip_val)
        elif quant_method == 'elastic':
            assert learnable, 'Elastic method must use leranable step size!'
            self.weight_clip_val = AlphaInit(torch.tensor(1.0)) # stepsize will be initialized in the first quantization
        else:
            self.register_buffer('weight_clip_val', None)

    def _build_input_clip_val(self, quant_method, learnable, init_val):
        if quant_method == 'uniform':
            self.register_buffer('input_clip_val', torch.tensor([-init_val, init_val]))
            if learnable:
                self.input_clip_val = nn.Parameter(self.input_clip_val)
        elif quant_method == 'elastic' or quant_method == 'bwn':
            assert learnable, 'Elastic method must use leranable step size!'
            self.input_clip_val = AlphaInit(torch.tensor(1.0))  # stepsize will be initialized in the first quantization
        else:
            self.register_buffer('input_clip_val', None)

    def forward(self, input):
        # quantize weight
        weight = weight_quant_fn(self.weight, self.weight_clip_val, num_bits=self.weight_bits, symmetric=self.symmetric,
                                 quant_method=self.weight_quant_method, layerwise=self.weight_layerwise)
        # quantize input
        input = self.move(input)
        input = act_quant_fn(input, self.input_clip_val, num_bits=self.input_bits, symmetric=self.symmetric,
                             quant_method=self.input_quant_method, layerwise=self.input_layerwise)
        out = nn.functional.linear(input, weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


class QuantizeEmbedding(nn.Embedding):

    def __init__(self, *kargs, clip_val=2.5, weight_bits=8, learnable=False, symmetric=True,
                 embed_layerwise=False, weight_quant_method="twn", **kwargs):
        super(QuantizeEmbedding, self).__init__(*kargs, **kwargs)
        self.weight_bits = weight_bits
        self.learnable = learnable
        self.symmetric = symmetric
        self.embed_layerwise = embed_layerwise
        self.weight_quant_method = weight_quant_method
        self._build_embed_clip_val(weight_quant_method, learnable, init_val=clip_val)

    def _build_embed_clip_val(self, quant_method, learnable, init_val):
        if quant_method == 'uniform':
            self.register_buffer('embed_clip_val', torch.tensor([-init_val, init_val]))
            if learnable:
                self.embed_clip_val = nn.Parameter(self.embed_clip_val)
        elif quant_method == 'elastic':
            assert learnable, 'Elastic method must use leranable step size!'
            self.embed_clip_val = AlphaInit(torch.tensor(1.0)) # stepsize will be initialized in the first quantization
        else:
            self.register_buffer('embed_clip_val', None)

    def forward(self, input):
        weight = weight_quant_fn(self.weight, self.embed_clip_val, num_bits=self.weight_bits, symmetric=self.symmetric,
                                 quant_method=self.weight_quant_method, layerwise=self.embed_layerwise)

        out = nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

        return out

class BNNLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True, precision='bnn', order=2):
        super(BNNLinear, self).__init__(in_features, out_features, bias=True)

        self.channel_threshold = torch.nn.Parameter(torch.zeros(1, self.weight.shape[1]), requires_grad=True)

        self.init_scale = False

        self.precision = precision
        self.bnn_mode = 'bnn'

        self.binary_act = True

    def forward(self, input, bnn_mode='bnn'):

        if 'full' in [self.precision, self.bnn_mode, bnn_mode]:
            return F.linear(input, self.weight, self.bias)

        if self.binary_act:
            x = input + self.channel_threshold
            out_forward = torch.sign(input)
            mask1 = x < -1
            mask2 = x < 0
            mask3 = x < 1
            out1 = (-1) * mask1.type(torch.float32) + (x * x + 2 * x) * (1 - mask1.type(torch.float32))
            out2 = out1 * mask2.type(torch.float32) + (-x * x + 2 * x) * (1 - mask2.type(torch.float32))
            out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
            input = out_forward.detach() - out3.detach() + out3

        real_weights = self.weight
        scaling_factor = torch.mean(abs(real_weights), dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        output = F.linear(input, binary_weights)

        return output

class BinaryActivation(nn.Module):
    """
    二值激活函数：
    - 前向：sign(x) ∈ {-1, +1}
    - 反向：使用分段二次多项式作为平滑近似，提供更平滑的梯度
    区间：
        x < -1        → -1
        -1 <= x < 0   → x^2 + 2x
        0  <= x < 1   → -x^2 + 2x
        x >= 1        → 1
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向二值输出：sign
        out_forward = torch.sign(x)

        # 三个区间 mask
        mask1 = (x < -1).to(x.dtype)  # x < -1
        mask2 = (x <  0).to(x.dtype)  # x < 0
        mask3 = (x <  1).to(x.dtype)  # x < 1

        # 分段多项式构造平滑函数 out3
        # 第一段：x < -1 → -1，其余用 x^2 + 2x
        out1 = (-1.0) * mask1 + (x * x + 2.0 * x) * (1.0 - mask1)

        # 第二段：x < 0 → 保持 out1，其余改为 -x^2 + 2x
        out2 = out1 * mask2 + (-x * x + 2.0 * x) * (1.0 - mask2)

        # 第三段：x < 1 → 保持 out2，其余固定为 1
        out3 = out2 * mask3 + 1.0 * (1.0 - mask3)

        # STE：前向数值等于 sign(x)，梯度来自 out3
        return out_forward.detach() - out3.detach() + out3
