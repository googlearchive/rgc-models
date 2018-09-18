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
"""Upsampling code"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.google as tf
import tensorflow.contrib.slim as slim


def conv2d(input_, kernel_size, stride, num_outputs, scope,
           activation_fn=tf.nn.relu):
  """Same-padded convolution with mirror padding instead of zero-padding.

  This function expects `kernel_size` to be odd.

  Args:
    input_: 4-D Tensor input.
    kernel_size: int (odd-valued) representing the kernel size.
    stride: int representing the strides.
    num_outputs: int. Number of output feature maps.
    scope: str. Scope under which to operate.
    activation_fn: activation function.

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
      input_, [[0, 0], [padding_x, padding_y], [padding_x, padding_y], [0, 0]],
      mode='REFLECT')
  return slim.conv2d(
      padded_input,
      padding='VALID',
      kernel_size=kernel_size,
      stride=stride,
      num_outputs=num_outputs,
      activation_fn=activation_fn,
      scope=scope)


def upsampling(input_,
               kernel_size,
               stride,
               num_outputs,
               scope,
               activation_fn=tf.nn.relu,
               tpu_compatible=False):
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
    tpu_compatible: bool. Whether to use a nearest neighbor upsampling
      compatible with TPU or the default tf.image implementation.

  Returns:
    4-D Tensor output.

  Raises:
    ValueError: if `kernel_size` is even.
  """
  if kernel_size % 2 == 0:
    raise ValueError('kernel_size is expected to be odd.')
  with tf.variable_scope(scope):
    if tpu_compatible:
      upsampled_input = upsampling_tpu_compatible(input_, stride)
    else:
      if input_.get_shape().is_fully_defined():
        _, height, width, _ = [s.value for s in input_.get_shape()]
      else:
        shape = tf.shape(input_)
        height, width = shape[1], shape[2]
      upsampled_input = tf.image.resize_nearest_neighbor(
          input_, [stride * height, stride * width])
    return conv2d(upsampled_input, kernel_size, 1, num_outputs, 'conv',
                  activation_fn=activation_fn)


def upsampling_tpu_compatible(data, scale):
  """Nearest neighbor upsampling TPU-compatible implementation.

  This implementation is TPU compatible as opposed to
  tf.image.resize_nearest_neighbor().

  Args:
    data: A 4D float32 tensor of shape [batch, height, width, channels].
    scale: An integer multiple to scale resolution of input data.

  Returns:
    A 4D float32 tensor of shape [batch, height*scale, width*scale, channels].
  """
  with tf.name_scope('upsampling_tpu_compatible'):
    if data.get_shape().is_fully_defined():
      bs, height, width, _ = [s.value for s in data.get_shape()]
    else:
      shape = tf.shape(data)
      bs, height, width = shape[0], shape[1], shape[2]
    channels = data.get_shape().as_list()[3]
    # Use reshape to quickly upsample the input. The nearest pixel is selected
    # implicitly via broadcasting.
    data = tf.reshape(data, [bs, height, 1, width, 1, channels]) * tf.ones(
        [1, 1, scale, 1, scale, 1], dtype=data.dtype)
    return tf.reshape(data, [bs, height * scale, width * scale, channels])


def residual_block(input_, kernel_size, scope, activation_fn=tf.nn.relu):
  """A residual block made of two mirror-padded, same-padded convolutions.

  This function expects `kernel_size` to be odd.

  Args:
    input_: 4-D Tensor, the input.
    kernel_size: int (odd-valued) representing the kernel size.
    scope: str, scope under which to operate.
    activation_fn: activation function.

  Returns:
    4-D Tensor, the output.

  Raises:
    ValueError: if `kernel_size` is even.
  """
  if kernel_size % 2 == 0:
    raise ValueError('kernel_size is expected to be odd.')
  with tf.variable_scope(scope):
    num_outputs = input_.get_shape()[-1].value
    h_1 = conv2d(input_, kernel_size, 1, num_outputs, 'conv1', activation_fn)
    h_2 = conv2d(h_1, kernel_size, 1, num_outputs, 'conv2', None)
    return input_ + h_2:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

