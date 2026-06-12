import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

import commons
import modules
from modules import LayerNorm, LayerNorm_legacy
from einops import rearrange

from typing import Optional, Tuple
#from flash_attn import flash_attn_func
#from flash_attn_custom import flash_attn_func

import zipformer as zf

class SwiGLU(nn.Module):
    def __init__(self, w1, w2, w3) -> None:
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def forward(self, x):
        x1 = F.linear(x, self.w1.weight)
        x2 = F.linear(x, self.w2.weight)
        hidden = F.silu(x1) * x2
        return F.linear(hidden, self.w3.weight)

class RotaryPositionalEmbeddings(nn.Module):
    """
    ## RoPE module
    Rotary encoding transforms pairs of features by rotating in the 2D plane.
    That is, it organizes the $d$ features as $\frac{d}{2}$ pairs.
    Each pair can be considered a coordinate in a 2D plane, and the encoding will rotate it
    by an angle depending on the position of the token.
    """
    def __init__(self, d: int, base: int = 10_000):
        r"""
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()
        self.base = base
        self.d = int(d)
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        r"""
        Cache $\cos$ and $\sin$ values
        """
        # Return if cache is already built
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        # Get sequence length
        seq_len = x.shape[0]

        # $\Theta = {\theta_i = 10000^{-\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)

        # Concatenate so that for row $m$ we have
        # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # Cache them
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: torch.Tensor):
        # $\frac{d}{2}$
        d_2 = self.d // 2

        # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        #return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[b h t d]`
        """
        # Cache $\cos$ and $\sin$ values
        #x = rearrange(x, "b h t d -> t b h d")

        x3d = (x.ndim == 3)
        if x3d:
            x = rearrange(x, "b t d -> t b 1 d")
        else:
            x = rearrange(x, "b h t d -> t b h d")

        self._build_cache(x)

        # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
        x_rope, x_pass = x[..., : self.d], x[..., self.d :]

        # Calculate
        # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        neg_half_x = self._neg_half(x_rope)
        x_rope = (x_rope * self.cos_cached[: x.shape[0]]) + (neg_half_x * self.sin_cached[: x.shape[0]])
        torch.cat((x_rope, x_pass), dim=-1)

        if x3d:
            return rearrange(x, "t b 1 d -> b t d")
        else:
            return rearrange(x, "t b h d -> b h t d")

        #return rearrange(torch.cat((x_rope, x_pass), dim=-1), "t b h d -> b h t d")


class Encoder(nn.Module):  # backward compatible vits2 encoder
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=4, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        encoder_layer = zf.Zipformer2EncoderLayer(
            embed_dim=hidden_channels,
            pos_dim=192,
            num_heads=n_heads,
            query_head_dim=24,
            pos_head_dim=4,
            value_head_dim=12,
            feedforward_dim=512,
            cnn_module_kernel=31,
            dropout=p_dropout,
        )
        self.enc_zf = zf.Zipformer2Encoder(
            encoder_layer,
            4,
            embed_dim=-1,
            time_embed_dim=-1,
            pos_dim=192,
            warmup_begin=4000.0 * (0 + 1) / (0 + 1),
            warmup_end=4000.0 * (0 + 2) / (0 + 1),
            final_layerdrop_rate=0.035 * (1 ** 0.5),
        )

    def forward(self, x, x_length, x_mask, g=None):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1) #-!
        attn_mask = (attn_mask.squeeze(1) == 0)

        x = x * x_mask
        x = x.transpose(1, 2)#-! tp
        x_mask = x_mask.transpose(1, 2) #-! tp

        x = self.enc_zf(src=x.transpose(0, 1), attn_mask=attn_mask)
        x = x.transpose(0, 1)

        x = x * x_mask
        return x.transpose(1, 2)


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads, p_dropout=0., window_size=None, heads_share=True,
                 block_length=None, proximal_bias=False, proximal_init=False):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        self.query_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)
        self.key_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))

        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert t_s == t_t, "Local attention is only available for self-attention."
                block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
                scores = scores.masked_fill(block_mask == 0, -1e4)

        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _attention_bias_proximal(self, length):
        """
        Bias for self-attention to encourage attention to close positions.
        Args:
          length: an integer scalar.
        Returns:
          a Tensor with shape [1, 1, length, length]
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., activation=None,
                 causal=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x


class Depthwise_Separable_Conv1D(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            padding_mode='zeros',  # TODO: refine this type
            device=None,
            dtype=None
    ):
        super().__init__()
        self.depth_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                    groups=in_channels, stride=stride, padding=padding, dilation=dilation, bias=bias,
                                    padding_mode=padding_mode, device=device, dtype=dtype)
        self.point_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias,
                                    device=device, dtype=dtype)

    def forward(self, input):
        return self.point_conv(self.depth_conv(input))

    def weight_norm(self):
        self.depth_conv = weight_norm(self.depth_conv, name='weight')
        self.point_conv = weight_norm(self.point_conv, name='weight')

    def remove_weight_norm(self):
        self.depth_conv = remove_weight_norm(self.depth_conv, name='weight')
        self.point_conv = remove_weight_norm(self.point_conv, name='weight')


class Depthwise_Separable_TransposeConv1D(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            output_padding=0,
            bias=True,
            dilation=1,
            padding_mode='zeros',  # TODO: refine this type
            device=None,
            dtype=None
    ):
        super().__init__()
        self.depth_conv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                             groups=in_channels, stride=stride, output_padding=output_padding,
                                             padding=padding, dilation=dilation, bias=bias, padding_mode=padding_mode,
                                             device=device, dtype=dtype)
        self.point_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias,
                                    device=device, dtype=dtype)

    def forward(self, input):
        return self.point_conv(self.depth_conv(input))

    def weight_norm(self):
        self.depth_conv = weight_norm(self.depth_conv, name='weight')
        self.point_conv = weight_norm(self.point_conv, name='weight')

    def remove_weight_norm(self):
        remove_weight_norm(self.depth_conv, name='weight')
        remove_weight_norm(self.point_conv, name='weight')


def weight_norm_modules(module, name='weight', dim=0):
    if isinstance(module, Depthwise_Separable_Conv1D) or isinstance(module, Depthwise_Separable_TransposeConv1D):
        module.weight_norm()
        return module
    else:
        return weight_norm(module, name, dim)


def remove_weight_norm_modules(module, name='weight'):
    if isinstance(module, Depthwise_Separable_Conv1D) or isinstance(module, Depthwise_Separable_TransposeConv1D):
        module.remove_weight_norm()
    else:
        remove_weight_norm(module, name)


class ConvReluNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.conv_layers.append(torch.nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm_legacy(hidden_channels))
        self.relu_drop = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
            )
            self.norm_layers.append(LayerNorm_legacy(hidden_channels))
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask