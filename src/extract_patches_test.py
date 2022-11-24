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

"""Tests for extract_patches."""


import extract_patches
import unittest

import math
import torch.nn.functional as F
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch', default=2, type=int, help='patch height')
    parser.add_argument('--stride', default=1, type=int, )

    parser.add_argument('--img_h', default=4, type=int, help='image height')
    parser.add_argument('--img_w', default=4, type=int, help='image weight')

    parser.add_argument('--ep', action="store_true", help='test Transformer only')
    args = parser.parse_args()
    return args

def extract_patches_tf(image, size=2, stride=1, rate=1, padding= "SAME"):
    B, H, W, C = image.shape
    image = torch.permute(image, (0, 3, 1, 2)) # B, C, H, W

    kernel_range_horizental = (size-1)*rate + 1 
    kernel_range_vertical = (size-1)*rate + 1 
    pad_row = 0
    pad_col = 0
    if padding == "SAME":
        # Do tensorflow "SAME" Padding 
        pad_row = kernel_range_horizental - 1
        pad_col = kernel_range_vertical - 1
        image = F.pad(image, (0, pad_row, 0, pad_col))
        # print("image",image[0,:,:,0])
        # print("padding ffijjf", pad_row, pad_col)
    elif padding == "VALID":
        pass
    else:
        raise NotImplementedError("some other padding format hasn't been check")
    
    H = (H + pad_col -rate*(kernel_range_vertical-1)-1)//stride + 1
    W = (W + pad_row -rate*(kernel_range_horizental-1)-1)//stride + 1
    print("kernel: ", kernel_range_vertical, kernel_range_horizental)
    print("pad: ", pad_col, pad_row)
    print(H, W)
    patches = F.unfold(image, kernel_size=(size,size), dilation=(rate,rate), stride=(stride,stride))
    print("1 ",patches.shape )
    patches = patches.permute((0, 2, 1))
    # print("2 ",patches.shape )
    patches = patches.reshape((B, H, W, C, -1))
    # print("3 ",patches.shape )
    patches = patches.permute((0, 1, 2, 4, 3)).reshape((B, H, W, -1))
    return patches
    

        




def extract_patches_tf_old(image, size, stride = 1, padding = "SAME"):
    """Plain tf patch extraction."""
    return tf.image.extract_patches(
        image,
        [1, size, size, 1],
        [1, stride, stride, 1],
        [1] * 4,
        padding,
    )


# class ExtractPatchesTest(unittest.TestCase, parameterized.TestCase):
class ExtractPatchesTest(unittest.TestCase):
    # @parameterized.product(
    #     size=(1, 2, 3, 4),
    #     stride=(1, 2),
    #     padding=("SAME", "VALID"),
    #     )
    def test_extract_patches_conv2d(self, size, stride, padding):
        # image = tf.random.normal((2, 16, 16, 5))
        image = torch.normal(mean=0, std=1, size=(2, 16, 16, 5))
        output = extract_patches.extract_patches_conv2d(image, size=size, stride=stride, padding=padding)
        expected = extract_patches_tf(image, size=size, stride=stride, padding=padding)
        diff = output-expected
        # print("out", output.shape)
        # print("expect: ", expected.shape)
        # print(output[0,:,:,0])
        # print(expected[0,:,:,0])
        assert sum(sum(sum(sum(diff)))).item() < 1e-5, f"error: {sum(sum(sum(sum(diff)))).item()}"
        # self.assertAllClose(output, expected)
        # raise "Latter check"

        if size == stride:
            print("========= Sub Test non_padding ========= (ExtractPatchesTest)")
            expected_non_overlapping = extract_patches.extract_patches_nonoverlapping(image, window_size=stride, pad=False)
            diff = output-expected_non_overlapping
            assert sum(sum(sum(sum(diff)))).item() < 1e-7, "error"
            # self.assertAllClose(output, expected_non_overlapping)
            # print(expected_non_overlapping.shape)
            # raise "Latter check"


class WindowPartitionTest(unittest.TestCase):

    def test_non_overlapping(self):
        print("========= Sub Test non_padding ========= (WindowPartitionTest)")
        # image = tf.random.normal((2, 16, 16, 4))
        image = torch.normal(mean=0, std=1, size=(2, 16, 16, 4))
        patches = extract_patches.window_partition(image, 4, pad=False)
        unpatched = extract_patches.unwindow(patches, 4, unpad=None)
        assert sum(sum(sum(sum(image-unpatched)))).item() < 1e-7, "patch error"
        print("No")

        print("=========== Sub Test padding =========== (WindowPartitionTest)")
        # image = tf.random.normal((2, 14, 14, 4))
        image = torch.normal(mean=0, std=1, size=(2, 14, 14, 4))
        patches = extract_patches.window_partition(image, 4, pad=True)
        unpatched = extract_patches.unwindow(patches, 4, unpad=(14, 14))
        # print("AS",image.shape)
        assert sum(sum(sum(sum(image-unpatched)))).item() < 1e-7, "patch error"


if __name__ == "__main__":
    args = parse_args()

    for size in [1,2,3,4]:
        for stride in [1,2]:
            for padding in ["VALID", "SAME"]:
                print("+++++++++++++++", size, stride, padding, "+++++++++++++++")
                if args.ep:
                    import tensorflow as tf
                    image = tf.random.normal((2, args.img_h, args.img_w, 3))
                    ans1 = extract_patches_tf_old(image, size=size, stride = stride, padding = padding)
                    print("original_tensor_format", ans1.shape)

                    
                    t = tf.identity(image).numpy()
                    torch_image = torch.from_numpy(t)

                    ans2 = extract_patches_tf(torch_image,  size=size, stride=stride, rate=1, padding = padding)
                    print("new_tensor_format", ans2.shape)
                    print("There differences: ",sum(sum(sum(sum(ans1-ans2)))))

                windows_testbench = WindowPartitionTest()
                windows_testbench.test_non_overlapping()

                extract_patch_testbench = ExtractPatchesTest()
                a = extract_patch_testbench.test_extract_patches_conv2d(size, stride, padding)
    

