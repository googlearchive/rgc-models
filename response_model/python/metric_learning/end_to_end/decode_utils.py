# Copyright 2018 Google LLC
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
# ==============================================================================
""" Utils for decoding the embedding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def conv2d(input_, kernel_size, stride, num_outputs, scope,
           activation_fn=tf.nn.relu, reuse_variables=False):
  """Same-padded convolution with mirror padding instead of zero-padding.

  This function expects `kernel_size` to be odd.

  Args:
    input_: 4-D Tensor input.
    kernel_size: int (odd-valued) representing the kernel size.
    stride: int representing the strides.
    num_outputs: int. Number of output feature maps.
    scope: str. Scope under which to operate.
    activation_fn: activation function.
    reuse_variables: boolean indicating if reuse variables.

  Returns:
    4-D Tensor output.

  Raises:
    ValueError: if `kernel_size` is even.
  """
  if isinstance(kernel_size, int):
    kernel_size_x, kernel_size_y = kernel_size, kernel_size
  else:
    if not isinstance(kernel_size, (tuple, list)):
      raise TypeError('kernel_size is expected to be tuple or a list.')
    if len(kernel_size) != 2:
      raise TypeError('kernel_size is expected to be of length 2.')
    kernel_size_x, kernel_size_y = kernel_size
  if kernel_size_x % 2 == 0 or kernel_size_y % 2 == 0:
    raise ValueError('kernel_size is expected to be odd.')
  padding_x = kernel_size_x // 2
  padding_y = kernel_size_y // 2
  padded_input = tf.pad(
      input_, [[0, 0], [padding_x, padding_y],
               [padding_x, padding_y], [0, 0]], mode='REFLECT')
  return slim.conv2d(
      padded_input,
      padding='VALID',
      kernel_size=kernel_size,
      stride=stride,
      num_outputs=num_outputs,
      activation_fn=activation_fn,
      scope=scope,
      reuse=reuse_variables)


def upsampling(input_, kernel_size, stride, num_outputs, scope,
               activation_fn=tf.nn.softplus, reuse_variables=False):
  """A smooth replacement of a same-padded transposed convolution.

  This function first computes a nearest-neighbor upsampling of the input by a
  factor of `stride`, then applies a mirror-padded, same-padded convolution.

  It expects `kernel_size` to be odd.

  Args:
    input_: 4-D Tensor input.
    kernel_size: int (odd-valued) representing the kernel size.
    stride: int representing the strides.
    num_outputs: int. Number of output feature maps.
    scope: str. Scope under which to operate.
    activation_fn: activation function.
    reuse_variables: boolean indicating if reuse variables.

  Returns:
    4-D Tensor output.

  Raises:
    ValueError: if `kernel_size` is even.
  """
  if kernel_size % 2 == 0:
    raise ValueError('kernel_size is expected to be odd.')
  with tf.variable_scope(scope):
    if input_.get_shape().is_fully_defined():
      _, height, width, _ = [s.value for s in input_.get_shape()]
    else:
      shape = tf.shape(input_)
      height, width = shape[1], shape[2]
    upsampled_input = tf.image.resize_nearest_neighbor(
        input_, [stride * height, stride * width])
    return conv2d(upsampled_input, kernel_size, 1, num_outputs, 'conv',
                  activation_fn=activation_fn, reuse_variables=reuse_variables)

