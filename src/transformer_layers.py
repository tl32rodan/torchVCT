# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

"""Transformer layers."""

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath


class WindowMultiHeadAttention(nn.Module):
    r""" 
    Args:
        d_model (int): Number of input channels.
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self, d_model, num_heads,  attn_drop=0., proj_drop=0., name = None,):
        super().__init__()
        self._d_model = d_model
        self._num_heads = num_heads
        if d_model % num_heads != 0:
            raise ValueError(
                f"Size of hidden units ({d_model}) not divisible by number "
                f"of head ({num_heads}).")
        head_dim = d_model // num_heads
        self._attn_scale = head_dim**(-0.5)

        # 1. relative bias isn't used
        # 2. in google implementation, kernel initialization use N(std=0.2)
        self.Lq = nn.Linear(d_model, d_model, bias=True)
        self.Lk = nn.Linear(d_model, d_model, bias=True)
        self.Lv = nn.Linear(d_model, d_model, bias=True)


        self._attn_drop = nn.Dropout(attn_drop)
        self._proj = nn.Linear(d_model, d_model)
        self._proj_drop = nn.Dropout(proj_drop)
        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self._softmax = nn.Softmax(dim=-1)

    def forward(self, v, k, q, mask=None, training=True):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b, seq_len_q, c = q.shape
        assert c == self._d_model, (c, self._d_model)
        seq_len_kv = v.shape[-2]
        blowup = seq_len_kv // seq_len_q

        q, k, v = self.Lq(q), self.Lk(k), self.Lv(v)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self._attn_scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            if mask.shape[-2:] != (seq_len_q, seq_len_q):
                raise ValueError(f"Invalid mask shape: {mask.shape}.")

            # tile_pattern = [1] * mask.shape.rank
            # tile_pattern[-1] = blowup
            # attn = attn + tf.tile(mask, tile_pattern) * -1e6
            attn = attn + mask.repeat(1, blowup) * -1e6
            # raise "Here need to be double checked !! （mask format)" # check

        attn = self._softmax(attn)

        if mask is not None:
            # We use the mask again, to be double sure that no masked dimension
            # affects the output.
            keep = 1 - mask
            attn *= mask.repeat(1, blowup)
            # raise "Here need to be double checked !! (mask format)" # check

        attn = self._attn_drop(attn)


        features = (attn @ v).transpose(1, 2).reshape(b, seq_len_q, c)

        features = self._proj(features)
        features = self._proj_drop(features)
        assert features.shape == (b, seq_len_q, c)
        return features, attn

    # def extra_repr(self) -> str:
    #     return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    # def flops(self, N):
    #     # calculate flops for 1 window with token length of N
    #     flops = 0
    #     # qkv = self.qkv(x)
    #     flops += N * self.dim * 3 * self.dim
    #     # attn = (q @ k.transpose(-2, -1))
    #     flops += self.num_heads * N * (self.dim // self.num_heads) * N
    #     #  x = (attn @ v)
    #     flops += self.num_heads * N * N * (self.dim // self.num_heads)
    #     # x = self.proj(x)
    #     flops += N * self.dim * self.dim
    #     return flops
class StochasticDepth(nn.Module):
    """Creates a stochastic depth layer."""

    def __init__(self, stochastic_depth_drop_rate):
        """Initializes a stochastic depth layer.
        Args:
            stochastic_depth_drop_rate: A `float` of drop rate.
        Returns:
            A output `tf.Tensor` of which should have the same shape as input.
        """
        super().__init__()
        self._drop_rate = stochastic_depth_drop_rate

    def forward(self, inputs, training):
        if not training or self._drop_rate == 0.:
            return inputs
        keep_prob = 1.0 - self._drop_rate

        batch_size = inputs.shape[0]
        random_tensor = keep_prob
        # random_tensor += tf.random.uniform(
        #     [batch_size] + [1] * (len(inputs.shape) - 1), dtype=inputs.dtype)
        random_tensor += torch.rand(
            [batch_size] + [1] * (len(inputs.shape) - 1), dtype=inputs.dtype)
        binary_tensor = torch.floor(random_tensor)
        # output = tf.math.divide(inputs, keep_prob) * binary_tensor # check
        output = torch.div(inputs, keep_prob) * binary_tensor
        return output

class MLP(nn.Module):
    def __init__(self, input_channel=192, expansion_rate=4., act_name=nn.GELU, dropout_rate=0.):
        super().__init__()
        self._expansion_rate = expansion_rate
        self._act_name = act_name
        self._dropout_rate = dropout_rate
        # out_features = out_features or in_features
        # hidden_features = hidden_features or in_features
        n_channel = input_channel
        print("sdadas", n_channel, n_channel*expansion_rate)
        self.fc1 = nn.Linear(n_channel, n_channel*expansion_rate)
        self.act = act_name()
        self.fc2 = nn.Linear(n_channel*expansion_rate, n_channel)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def create_look_ahead_mask(size):
    """Creates a lookahead mask for autoregressive masking."""
    mask = np.triu(np.ones((size, size), np.float32), 1)
    return torch.from_numpy(mask)


class TransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        d_model (int): Number of input channels.
        seq_len (int): sequence length.
        num_heads (int): Number of attention heads.

        mlp_expansion (float): Ratio of mlp hidden dim to embedding dim.

        drop_out_rate (float, optional): Dropout rate. Default: 0.0
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.0
        mlp_act (nn.Module, optional): Activation layer. Default: nn.GELU
    """
    def __init__(self, d_model=192, seq_len=16, num_head=4, mlp_expansion=4, drop_out_rate=0.,  drop_path_rate=0., mlp_act=nn.GELU, style = "decoder"):
        super().__init__()
        self.style = style
        if self.style == "decoder":
            self.look_ahead_mask = create_look_ahead_mask(seq_len)
        elif self.style == "encoder":
            self.look_ahead_mask = None
        else:
            raise ValueError(f"Invalid style: {style}")

        self._norm1a = nn.LayerNorm(d_model, eps=1e-5)
        self._norm1b = nn.LayerNorm(d_model, eps=1e-5)
        self._attn1 = WindowMultiHeadAttention(d_model,  num_head, attn_drop=drop_out_rate, proj_drop=drop_out_rate)
        self._attn2 = WindowMultiHeadAttention(d_model,  num_head, attn_drop=drop_out_rate, proj_drop=drop_out_rate)

        # check here
        self._drop_path = StochasticDepth(drop_path_rate) 

        # check
        self._mlp1 = MLP(input_channel=d_model, expansion_rate=mlp_expansion, act_name=mlp_act, dropout_rate=drop_out_rate)
        self._mlp2 = MLP(input_channel=d_model, expansion_rate=mlp_expansion, act_name=mlp_act, dropout_rate=drop_out_rate)


        self._norm2a = nn.LayerNorm(d_model, eps=1e-5)
        self._norm2b = nn.LayerNorm(d_model, eps=1e-5)



    def forward(self, features, enc_output, training):
        shortcut = features
        x = self._norm1a(features)
        # x = x.view(B, H, W, C)

        features, _ = self._attn1(v=features, k=features, q=features, mask=self.look_ahead_mask, training=training)
        assert features.shape == shortcut.shape
        features = shortcut + self._drop_path(features, training)

        features = features + self._drop_path(self._mlp1(self._norm1b(features)), training)

        # Second Block ---
        shortcut = features
        features = self._norm2a(features)
        # Unmasked "lookup" into enc_output, no need for mask.
        features, _ = self._attn2(  # pytype: disable=wrong-arg-types  # dynamic-method-lookup
            v=enc_output if enc_output is not None else features,
            k=enc_output if enc_output is not None else features,
            q=features,
            mask=None,
            training=training)
        features = shortcut + self._drop_path(features, training)
        output = features + self._drop_path(self._mlp2(self._norm2b(features)), training)

        return output

    # def extra_repr(self) -> str:
    #     return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
    #            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    # def flops(self):
    #     flops = 0
    #     H, W = self.input_resolution
    #     # norm1
    #     flops += self.dim * H * W
    #     # W-MSA/SW-MSA
    #     nW = H * W / self.window_size / self.window_size
    #     flops += nW * self.attn.flops(self.window_size * self.window_size)
    #     # mlp
    #     flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
    #     # norm2
    #     flops += self.dim * H * W
    #     return flops




class Transformer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        is_decoder (bool): whelther the block is decoder
        num_layers (int): Number of blocks.
        d_model (int): Number of input channels.
        seq_len (int): Sequence length
        num_heads (int): Number of attention heads.
        mlp_expansion (float): Ratio of mlp hidden dim to embedding dim.
        drop_out (float, optional): Dropout rate. Default: 0.0
    """
    def __init__(self, is_decoder, num_layers = 4, d_model = 192, seq_len = 16, num_head = 4,
        mlp_expansion = 4, drop_out = 0.1, name = None):

        super().__init__()
        self.is_decoder = is_decoder

        # # build layers
        self.layers = nn.ModuleList([
          TransformerBlock(d_model=d_model,
                            seq_len=seq_len,
                            num_head=num_head,
                            mlp_expansion=mlp_expansion,
                            drop_out_rate=drop_out,
                            drop_path_rate=drop_out,
                            style="decoder" if is_decoder else "encoder",)
            for i in range(num_layers)])


    def forward(self, latent, enc_output, training):
        """Forward pass.
        For decoder, this predicts distribution of `latent` given `enc_output`.
        We assume that `latent` has already been embedded in a d_model-dimensional
        space.
        Args:
        latent: (B', seq_len, C) latent.
        enc_output: (B', seq_len_enc, C) result of concatenated encode output.
        training: Whether we are training.
        Returns:
        Decoder output of shape (B', seq_len, C).
        """
        assert len(latent.shape) == 3, latent.shape
        if enc_output is not None:
            assert latent.shape[-1] == enc_output.shape[-1], (latent.shape, enc_output.shape)
            
        for layer in self.layers:
            latent = layer(features=latent, enc_output=enc_output, training=training)

        return latent

        # def extra_repr(self) -> str:
        #     return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

        # def flops(self):
        #     flops = 0
        #     for blk in self.blocks:
        #         flops += blk.flops()
        #     if self.downsample is not None:
        #         flops += self.downsample.flops()
        #     return flops


class EncoderSection(Transformer):
    """N-layer encoder."""

    def __init__(
        self,
        num_layers,
        d_model,
        mlp_expansion,
        num_head,
        drop_out,
        name = None,
    ):
        super().__init__(
            is_decoder=False,
            num_layers=num_layers,
            d_model=d_model,
            seq_len=0,
            num_head=num_head,
            mlp_expansion=mlp_expansion,
            drop_out=drop_out,
            name=name)

    def forward(self, latent, training):
        return super(latent, None, training)