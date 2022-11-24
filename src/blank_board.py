import tensorflow as tf
import torch
import torch.nn.functional as F 
import math

def get_num_patches(height, width, stride=8):
    # Initial pad to get all strides in.
    height_padded = math.ceil(height / stride) * stride
    width_padded = math.ceil(width / stride) * stride
    # Calculate number of patches in the height and width dimensions.
    n_h = height_padded // stride
    n_w = width_padded // stride
    return (n_h, n_w), (height_padded, width_padded)
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

    stride = 8
    patch_size=16
    h = 13 
    w = 16
    x = tf.reshape(tf.range(h * w), (1, h, w, 1))
    x = tf.concat([x for _ in range(16)], -1) 

    if patch_size < stride:
      raise ValueError("`patch_size` must be greater than `stride`!")
    # Additionally pad to handle patch_size > stride.
    missing = patch_size - stride
    if missing % 2 != 0:
      raise ValueError("Can only handle even missing pixels.")

    _, height, width, _ = x.shape
    print("original: ",x.shape)
    (n_h, n_w), (height_padded, width_padded) = get_num_patches(height, width, stride)
    out = tf.pad(x, [
        [0, 0],
        [missing // 2, height_padded - height + missing // 2],
        [missing // 2, width_padded - width + missing // 2],
        [0, 0],
    ], "REFLECT")
    print(out.shape)
    print((n_h, n_w), (height_padded, width_padded))


