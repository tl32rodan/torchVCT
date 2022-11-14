import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda import amp

from compressai.layers import AttentionBlock as SimpleAttention
from compressai.models.utils import conv, deconv


def conv3x3(in_channels, out_channels):
    return conv(in_channels, out_channels, 3, 1)

def conv1x1(in_channels, out_channels):
    return conv(in_channels, out_channels, 3, 1)

class ResidualBlock(nn.Module):
    """Simple residual unit.

    Reference PyTorch code from CompressAI:
    https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/layers/layers.py#L193

    """

    def __init__(self, N = 64):
        super().__init__()

        self.conv = nn.Sequential(
            conv1x1(N, N // 2),
            nn.ReLU(inplace=True),
            conv3x3(N // 2, N // 2),
            nn.ReLU(inplace=True),
            conv1x1(N // 2, N),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv(x)
        out += identity
        out = self.relu(out)
        return out


class ElicAnalysis(nn.Module):
    """Analysis transform from ELIC.
    
    Can be configured to match the analysis transform from the "Devil's in the
    Details" paper or a combination of ELIC + Devil.
    
    ELIC = https://arxiv.org/abs/2203.10886
    Devil = https://arxiv.org/abs/2203.08450
    
    Note that the paper uses channels = [192, 192, 192, 320].
    """

    def __init__(self,
                 num_residual_blocks = 3,
                 channels = (128, 160, 192, 192),
                 input_channels = None,
                 output_channels = None,
                 **kwargs):
        super().__init__(**kwargs)
        if len(channels) != 4:
            raise ValueError(f"ELIC uses 4 conv layers (not {channels}).")
        if input_channels) is None:
            raise ValueError(f"input_channels should be specified.")
        if output_channels is not None and output_channels != channels[-1]:
            raise ValueError("output_channels specified but does not match channels: "
                           f"{output_channels} vs. {channels}")

        self._output_depth = channels[-1]

        # Keep activation separate from conv layer for clarity and symmetry.
        conv = functools.partial(
            build_conv, kernel_size=5, strides=2, act=None, up_or_down="down")

        convs = [conv(input_channels=cin, output_channels=cout) for cin, cout in zip([input_channels]+channels[:-1], channels)]

        def build_act(N):
            return [ResidualBlock(N) for _ in range(num_residual_blocks)]

        blocks = [
            convs[0],
            *build_act(N=channels[0]),
            convs[1],
            *build_act(N=channels[1]),
            SimpleAttention(),
            convs[2],
            *build_act(N=channels[2]),
            convs[3],
            SimpleAttention(),
        ]
        blocks = list(filter(None, blocks))  # remove None elements
        self._transform = nn.Sequential(blocks)

    def forward(self, x, training = None):
        del training
        return self._transform(x)

    @property
    def output_depth(self):
        return self._output_depth

    def compute_output_shape(self, input_shape):
        shape = list(torch.size(input_shape))
        h, w = shape[-3], shape[-2]
        shape[-3:] = [h // 16, w // 16, self.output_depth]
        return torch.Size(shape)


class ElicSynthesis(nn.Module):
    """Synthesis transform from ELIC.
    ELIC = https://arxiv.org/abs/2203.10886
    """

    def __init__(self,
                 num_residual_blocks = 3,
                 channels = (192, 160, 128, 3),
                 input_channels = None,
                 output_channels = None,
                 **kwargs):
        super().__init__(**kwargs)
        if input_channels) is None:
            raise ValueError(f"input_channels should be specified.")
        if len(channels) != 4:
            raise ValueError(f"ELIC uses 4 conv layers (not {channels}).")
        if output_channels is not None and output_channels != channels[-1]:
            raise ValueError("output_channels specified but does not match channels: "
                           f"{output_channels} vs. {channels}")

        self._output_depth = channels[-1]

        # Keep activation separate from conv layer for clarity and because
        # second conv is followed by attention, not an activation.
        conv = functools.partial(
            build_conv, kernel_size=5, strides=2, act=None, up_or_down="up")

        convs = [conv(input_channels=cin, output_channels=cout) for cin, cout in zip([input_channels] + channels[:-1], channels)]

        def build_act(N):
            return [ResidualBlock(N) for _ in range(num_residual_blocks)]

        blocks = [
            SimpleAttention(),
            convs[0],
            *build_act(N=channels[0]),
            convs[1],
            SimpleAttention(),
            *build_act(N=channels[1]),
            convs[2],
            *build_act(N=channels[2]),
            convs[3],
        ]
        blocks = list(filter(None, blocks))  # remove None elements
        self._transform = torch.Sequential(blocks)

    def forward(self, x, training = None):
        del training  # Unused.
        return self._transform(x)

    @property
    def output_depth(self):
        return self._output_depth

    def compute_output_shape(self, input_shape):
        shape = list(torch.size(input_shape))
        h, w = shape[-3], shape[-2]
        shape[-3:] = [h * 16, w * 16, self.output_depth]
        return torch.Size(shape)


def build_conv(
               input_channels = 0,
               output_channels = 0,
               kernel_size = 3,
               strides = 1,
               act = None,
               up_or_down = "down",
               name = None):
    """Builds either an upsampling or downsampling conv layer."""
    layer_cls = dict(
        up=conv,
        down=deconv)[up_or_down]
    return layer_cls(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=kernel_size,
        strides=strides,
    )
