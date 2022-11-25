import tensorflow as tf
import torch
import torch.nn.functional as F 
import math
import numpy as np
import torch.nn as nn

def get_num_patches(height, width, stride=8):
    # Initial pad to get all strides in.
    height_padded = math.ceil(height / stride) * stride
    width_padded = math.ceil(width / stride) * stride
    # Calculate number of patches in the height and width dimensions.
    n_h = height_padded // stride
    n_w = width_padded // stride
    return (n_h, n_w), (height_padded, width_padded)

import torch.nn as nn
def _shift_to_the_right(x, pad = None):
    """Returns essentially [pad, x[:-1]] but using the right dimensions."""
    *dims, _, c = x.shape
    expected_pad_shape = (*dims, 1, c)
    if pad is None:
        pad = torch.zeros(expected_pad_shape, dtype=x.dtype)
    elif pad.shape != expected_pad_shape:
        raise ValueError(f"Invalid shape: {pad.shape} != {expected_pad_shape}")
    print("origin", x.shape)
    print("new", x[Ellipsis, :-1, :].shape)
    print("pad: ", pad.shape)
    return torch.concat((pad, x[Ellipsis, :-1, :]), axis=-2)
class StartSym(nn.Module):
    """Helper to learn a "zero" symbol, i.e., the first symbol to feed."""

    def __init__(self, num_channels):
        super().__init__()

        mask = torch.FloatTensor(num_channels).uniform_(-3, 3)
        mask = nn.Parameter(mask, requires_grad=True)
        self.register_parameter('sym', mask)


    def forward(self, x):
        """Prefixes `x` with the learned start symbol."""
        b, _, c = x.shape
        print(self.sym * torch.ones((b, 1, c)))
        return _shift_to_the_right(x, self.sym * torch.ones((b, 1, c))) # check

class StartSym_tf(tf.keras.layers.Layer):
  """Helper to learn a "zero" symbol, i.e., the first symbol to feed."""

  def __init__(self, num_channels):
    super().__init__()

    def initializer(shape, dtype):
      return tf.random.uniform(shape, -3, 3, dtype, seed=42)

    self.sym = self.add_weight(
        shape=(num_channels,),
        initializer=initializer,
        trainable=True,
        name="sym",
    )
    print(self.sym)
    print(self.sym.shape)

  def call(self, x):
    """Prefixes `x` with the learned start symbol."""
    b, _, c = x.shape
    print(self.sym * tf.ones((b, 1, c)))
    print((self.sym * tf.ones((b, 1, c))).shape)
    return _shift_to_the_right(x, self.sym * tf.ones((b, 1, c)))


class LearnedPosition(nn.Module):
    """Single learned positional encoding."""

    def __init__(self, name, seq_len, d_model,):
        super().__init__()
        mask = torch.normal(mean=torch.as_tensor([0.]*(seq_len*d_model)), 
                                std = torch.as_tensor([0.02]*(seq_len*d_model))).reshape(seq_len, d_model)
        mask = nn.Parameter(mask, requires_grad=True)
        self.register_parameter('_emb', mask)
        # self._emb = self.add_weight(
        #     initializer=tf.random_normal_initializer(stddev=0.02),
        #     trainable=True,
        #     dtype=tf.float32,
        #     shape=[seq_len, d_model],
        #     name=name)
        self._seq_len = seq_len
        self._d_model = d_model

    def forward(self, tensor):
        """Adds positional encodings to `tensor`."""
        expected = (self._seq_len, self._d_model)
        if tensor.shape[-2:] != expected:
            raise ValueError(f"Invalid shape, {tensor.shape[-2:]} != {expected}")
        return tensor + self._emb
if __name__ == "__main__":

    # size = 1
    # stride = 1
    # padding = "SAME"
    # image = tf.random.normal((2, 16, 16, 5))
    # channels = int(image.shape[-1])
    # print(image)
    # print(image.shape)

    # kernel = tf.reshape(
    # tf.eye(size * size * channels, dtype=image.dtype),
    # (size, size, channels, channels * size * size))
    # print(kernel)
    # print(kernel.shape)

    # out = tf.nn.conv2d(image, kernel, strides=stride, padding=padding)
    # print(out.shape)

    # stride = 8
    # patch_size=16
    # h = 13 
    # w = 16
    # x = tf.reshape(tf.range(h * w), (1, h, w, 1))
    # x = tf.concat([x for _ in range(16)], -1) 

    # if patch_size < stride:
    #   raise ValueError("`patch_size` must be greater than `stride`!")
    # # Additionally pad to handle patch_size > stride.
    # missing = patch_size - stride
    # if missing % 2 != 0:
    #   raise ValueError("Can only handle even missing pixels.")

    # _, height, width, _ = x.shape
    # print("original: ",x.shape)
    # (n_h, n_w), (height_padded, width_padded) = get_num_patches(height, width, stride)
    # out = tf.pad(x, [
    #     [0, 0],
    #     [missing // 2, height_padded - height + missing // 2],
    #     [missing // 2, width_padded - width + missing // 2],
    #     [0, 0],
    # ], "REFLECT")
    # print(out.shape)
    # print((n_h, n_w), (height_padded, width_padded))

    # shape = [3,3]
    dtype = torch.float64
    # import random
    # random.seed(32)
    # a = torch.FloatTensor([3,3]).uniform_(-3, 3).to(dtype)
    # b = torch.tensor(a, requires_grad=True)
    # print(a)
    # print(a.shape)
    # print(a.dtype)
    # print(a.requires_grad)
    # print(b)
    # print(b.shape)
    # print(b.dtype)
    # print(b.requires_grad)   
    # a = np.array((1,2,3))
    # a = torch.from_numpy(a)
    # print(a)
    # print(a.requires_grad)
    # model = StartSym(10)
    # print(list(model.named_parameters()))
    # model = StartSym_tf(10)
    # a = model(tf.ones((3,2,10)))
    # seq_len = 3
    # d_model = 5
    # a = torch.normal(mean=torch.as_tensor([0.]*(seq_len*d_model)), std = torch.as_tensor([0.02]*(seq_len*d_model))).reshape(seq_len, d_model)
    # # a = torch.normal(mean=(1. , 2.), std = (0.01, 0.03))
    # print(a)
    # print(a.shape)
    # model = LearnedPosition("test1", 3, 5)
    # a = model(torch.ones((1,3,5)))
    # print(list(model.named_parameters()))
    # shape = (1,2,3)
    # N = 1
    # for i in shape:
    #     N *= i
    # a = torch.ones((N, 5)).reshape(*shape, 5)
    # print(a.shape)
    dim = (64, 64)
    dim = (1,) + tuple(dim)
    print(dim)
    N = 1
    for i in dim:
        N *= i
    print(N)


