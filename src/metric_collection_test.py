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

"""Tests for metric_scope."""
from absl.testing import parameterized
import numpy as np
import unittest
# import tensorflow as tf
import torch
import metric_collection
from assert_fnc import *

Metrics = metric_collection.Metrics

TEST_IMAGE = torch.as_tensor([[[[0], [64]], [[128], [255]]]],
                                  dtype=torch.uint8)

TEST_IMAGE_FLOAT = torch.as_tensor(
    [[[[-0.01], [0.25]], [[0.501], [1.01]]]]).to(torch.float32)

def nested():
    metrics = Metrics.make()
    metrics.record_scalar("mse", torch.as_tensor(1.5))
    metrics.record_image("reconstruction", TEST_IMAGE)
    return metrics

def nested2():
    metrics = Metrics.make()
    metrics.record_scalar("mse2", 2.0)
    metrics.record_scalar("mse3", np.float32(3.0))
    return metrics

# class MetricCollectionTest(parameterized.TestCase, tf.test.TestCase):
class MetricCollectionTest(unittest.TestCase):
    # @parameterized.parameters([
    #     (("foo", "bar"), "foo/bar"),
    #     (("foo/", "bar"), "foo/bar"),
    #     (("/foo", "/bar"), "foo/bar"),
    #     (("/foo/", "/bar/"), "foo/bar"),
    #     (("foo/bar", "baz"), "foo/bar/baz"),
    #     (("", "baz"), "baz"),
    # ])
    def test_join(self, components, expected_result):
        self.assertEqual(metric_collection.join(*components),
                            expected_result)

    def test_collection(self):
        metrics = Metrics.make()
        metrics.record_scalar("global_step", 1)
        metrics_nested = nested()
        metrics_nested2 = nested2()
        metrics.merge("nested", metrics_nested)
        metrics.merge("nested2", metrics_nested2)
        images = metrics.images
        self.assertDictEqual(metrics.scalars_float, {
            "global_step": 1,
            "nested/mse": 1.5,
            "nested2/mse2": 2.0,
            "nested2/mse3": 3.0,
        })
        assertAllClose(images["nested/reconstruction"], TEST_IMAGE)

    def test_merge_single_argument(self):
        metrics = Metrics.make()
        metrics.record_scalar("global_step", 1)
        metrics_nested = nested()
        metrics_nested2 = nested2()
        metrics.merge(metrics_nested)
        metrics.merge(metrics_nested2)
        images = metrics.images
        assertDictEqual(metrics.scalars_float, {
            "global_step": 1,
            "mse": 1.5,
            "mse2": 2.0,
            "mse3": 3.0,
        })
        assertAllClose(images["reconstruction"], TEST_IMAGE)

    def test_works_like_a_tuple(self):
        metrics = Metrics.make()
        metrics.record_scalar("global_step", 1)
        metrics.record_image("reconstruction", TEST_IMAGE)
        scalars, images = metrics
        assertLen(metrics, 2)  # Namely: scalars, images.
        self.assertEqual(tuple(metrics), (scalars, images))

        new_metrics = Metrics(scalars, images)
        self.assertEqual(metrics, new_metrics)

    def test_collection_prevents_overwrite(self):
        metrics = Metrics.make()
        metrics.record_scalar("global_step", 1)
        with self.assertRaisesRegex(ValueError, "Duplicate value for key"):
            metrics.record_scalar("global_step", 1)

    def test_collection_merge_prevents_overwrite(self):
        metrics = Metrics.make()
        metrics.record_scalar("my_key", 1)

    def test_merge_checks_for_duplicates(self):
        metrics = Metrics.make()
        metrics.record_scalar("my_key", 1)

        # This is OK, as we can distinguish `my_key` and `my_key/foo`.
        metrics.merge("my_key", Metrics({"foo": 1}, {}))

        # This is OK, as `my_key/bar` is new.
        metrics.merge("my_key", Metrics({"bar": 1}, {}))

        # Not OK, we now already have `my_key/foo`.
        with self.assertRaisesRegex(ValueError, "Duplicate value for "):
            metrics.record_scalar(metric_collection.join("my_key", "foo"), 1)

        # Another prefix is fine.
        metrics.record_scalar(metric_collection.join("my_key2", "foo"), 1)

    def test_reduce_scalars(self):
        metrics1 = Metrics.make()
        metrics1.record_scalar("foo", 1)
        metrics1.record_image("foo", TEST_IMAGE)

        metrics2 = Metrics.make()
        metrics2.record_scalar("bar", 3)

        metrics3 = Metrics.make()
        metrics3.record_scalar("foo", 2)

        metrics_sum = Metrics.reduce([
            metrics1,
            metrics2,
            metrics3,
        ], scalar_reduce_fn=sum)
        self.assertEqual(metrics_sum.scalars, {
            "foo": 1 + 2,
            "bar": 3
        })
        assertEmpty(metrics_sum.images)

    def test_record_image_checks_rank(self):
        metrics = Metrics.make()
        with self.assertRaisesRegex(ValueError, "rank"):
            metrics.record_image("my_image", TEST_IMAGE_FLOAT[0])

    def test_record_image_float32(self):
        metrics = Metrics.make()
        metrics.record_image("my_image", TEST_IMAGE_FLOAT)
        assertAllClose(metrics.images["my_image"], TEST_IMAGE)

    def test_record_image_float32_clips_and_rounds(self):
        metrics = Metrics.make()
        my_image = torch.as_tensor(
            [[[[0.0], [123.45 / 255.0]], [[123.55 / 255.0], [1.1]]]]).to(torch.float32)
        expected = torch.as_tensor([[[[0], [123]], [[124], [255]]]]).to(torch.uint8)
        metrics.record_image("my_image", my_image)
        assertAllClose(metrics.images["my_image"], expected)

    def test_disable_recording(self):
        with metric_collection.disable_recording():
            metrics = Metrics.make()
            metrics.record_scalar("global_step", 1)
            metrics_nested = nested()
            metrics_nested2 = nested2()
            metrics.merge("nested", metrics_nested)
            metrics.merge("nested2", metrics_nested2)
        assertEmpty(metrics.scalars)
        assertEmpty(metrics.images)


if __name__ == "__main__":
    # tf.test.main()
    testbench = MetricCollectionTest()

    for input, output in [
        (("foo", "bar"), "foo/bar"),
        (("foo/", "bar"), "foo/bar"),
        (("/foo", "/bar"), "foo/bar"),
        (("/foo/", "/bar/"), "foo/bar"),
        (("foo/bar", "baz"), "foo/bar/baz"),
        (("", "baz"), "baz"),
    ]:
        testbench.test_join(input, output)
    testbench.test_collection()
    testbench.test_merge_single_argument()
    testbench.test_works_like_a_tuple()
    testbench.test_collection_prevents_overwrite()
    testbench.test_collection_merge_prevents_overwrite()
    testbench.test_merge_checks_for_duplicates()
    testbench.test_reduce_scalars()
    testbench.test_record_image_checks_rank()
    testbench.test_record_image_float32()
    testbench.test_record_image_float32_clips_and_rounds()
    testbench.test_disable_recording()
    print("No Error")


    