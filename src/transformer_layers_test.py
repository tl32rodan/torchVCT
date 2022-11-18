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

"""Tests for transformer_layers."""



import transformer_layers
import torch
import unittest
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', action="store_true", help='test encoder only')
    parser.add_argument('--decoder', action="store_true", help='test decoder only')
    parser.add_argument('--transformer', action="store_true", help='test Transformer only')
    args = parser.parse_args()
    return args


class TransformerLayersTest(unittest.TestCase):
    def test_make_transformer_block_encoder(self):
        for training in (True, False):
            Case = "Training" if training else "Testing"
            print("======== ", Case, " stage for encoder ========")

            block = transformer_layers.TransformerBlock(d_model=8, seq_len=4, style="encoder")
            self.assertIsNone(block.look_ahead_mask, "The mask is not None")

            inp = torch.ones((16, 4, 8))
            otp = block(inp, enc_output=None, training=training)


    def test_make_transformer_block_decoder(self):
        for training in (True, False):
            Case = "Training" if training else "Testing"
            print("======== ", Case, " stage for decoder ========")
            block = transformer_layers.TransformerBlock(d_model=8, seq_len=4, style="decoder")
            import numpy as np
            expected_look_ahead = torch.from_numpy(np.array([
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ], dtype=np.float32))
            assert sum(sum(block.look_ahead_mask - expected_look_ahead)) < 1e-7, "Mask got some problem"

            inp = torch.ones((16, 4, 8))
            enc_output = torch.ones((16, 17, 8))
            otp = block(inp, enc_output=enc_output, training=training)
            self.assertEqual(otp.shape, inp.shape)

    def test_transformer(self):
        for training in (True, False):
            Case = "Training" if training else "Testing"
            print("======== ", Case, " stage for Transformer ========")

            t = transformer_layers.Transformer(is_decoder=True, num_layers=2, d_model=8, num_head=2, seq_len=16)
            enc_output = torch.ones((3, 16, 8))
            otp = t(torch.ones((3, 16, 8)), enc_output=enc_output, training=training)
            self.assertEqual(otp.shape, (3, 16, 8))

if __name__ == "__main__":
    args = parse_args()
    TestBench = TransformerLayersTest()
    if args.encoder:
        TestBench.test_make_transformer_block_encoder()
    elif args.decoder:
        TestBench.test_make_transformer_block_decoder()
    else:
        TestBench.test_transformer()