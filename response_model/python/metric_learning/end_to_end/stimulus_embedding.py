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
r"""Different stimulus embeddings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import retina.prosthesis.end_to_end.upsampling as decode_utils
slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS


def convolutional_encode(layers, batch_norm, stim_in,
                         is_training, reuse_variables=False):
  """Embed stimulus using multiple layers of convolutions.

  Each convolutional layer has convolution, batch normalization and soft-plus.
  Args :
    layers : string description of multiple layers.
             Format - (a1, b1, c1, a2, b2, c2 .. ),
               where (a, b, c) = (filter width, num of filters, stride)
    batch_norm : (boolean) if batch norm applied after each convolution.
    stim_in : stimulus input.
    is_training : (boolean) if training or testing for (batch norm).
    reuse_variables : if using previously-defined variables.
                      Useful for embedding mutliple responses using same graph.

  Returns :
    net : stimulus embedding
  """

  n_layers = int(len(layers)/3)
  tf.logging.info('Number of layers: %d' % n_layers)

  # Set normalization
  if batch_norm:
    normalizer_fn = slim.batch_norm
  else:
    normalizer_fn = None

  activation_fn = tf.nn.softplus
  tf.logging.info('Logistic activation')

  # Use slim to define multiple layers of convolution.
  net = stim_in
  layer_collection = [net]
  for ilayer in range(n_layers):
    tf.logging.info('Building stimulus embedding layer: %d, %d, %d'
                    % (int(layers[ilayer*3 + 1]), int(layers[ilayer*3]),
                       int(layers[ilayer*3 + 2])))
    net = slim.conv2d(net, int(layers[ilayer*3 + 1]),
                      int(layers[ilayer*3]),
                      stride=int(layers[ilayer*3 + 2]),
                      scope='stim_layer_wt_%d' % ilayer,
                      reuse=reuse_variables,
                      normalizer_fn=normalizer_fn,
                      activation_fn=activation_fn,
                      normalizer_params={'is_training': is_training})
    layer_collection += [net]
  return net, layer_collection

def convolutional_encode2(layers, batch_norm, stim_in,
                         is_training, reuse_variables=False):
  """Embed stimulus using multiple layers of convolutions.

  Each convolutional layer has convolution, batch normalization and soft-plus.
  Args :
    layers : string description of multiple layers.
             Format - (a1, b1, c1, a2, b2, c2 .. ),
               where (a, b, c) = (filter width, num of filters, stride)
    batch_norm : (boolean) if batch norm applied after each convolution.
    stim_in : stimulus input.
    is_training : (boolean) if training or testing for (batch norm).
    reuse_variables : if using previously-defined variables.
                      Useful for embedding mutliple responses using same graph.

  Returns :
    net : stimulus embedding
  """

  n_layers = int(len(layers)/3)
  tf.logging.info('Number of layers: %d' % n_layers)

  # Set normalization
  if batch_norm:
    normalizer_fn = slim.batch_norm
  else:
    normalizer_fn = None

  tf.logging.info('Logistic activation')

  # Use slim to define multiple layers of convolution.
  net = stim_in
  layer_collection = [net]
  for ilayer in range(n_layers):
    if ilayer == n_layers - 1:
      activation_fn = None
    else:
      activation_fn = tf.nn.softplus

    tf.logging.info('Building stimulus embedding layer: %d, %d, %d'
                    % (int(layers[ilayer*3 + 1]), int(layers[ilayer*3]),
                       int(layers[ilayer*3 + 2])))
    net = slim.conv2d(net, int(layers[ilayer*3 + 1]),
                      int(layers[ilayer*3]),
                      stride=int(layers[ilayer*3 + 2]),
                      scope='stim_layer_wt_%d' % ilayer,
                      reuse=reuse_variables,
                      normalizer_fn=normalizer_fn,
                      activation_fn=activation_fn,
                      normalizer_params={'is_training': is_training})
    layer_collection += [net]
  return net, layer_collection


def residual_encode(layers, batch_norm, stim_in,
                         is_training, reuse_variables=False, scope_prefix='stim', input_channels=30):
  """Embed stimulus using multiple layers of convolutions.

  Each convolutional layer has convolution, batch normalization and soft-plus.
  Args :
    layers : string description of multiple layers.
             Format - (a1, b1, c1, a2, b2, c2 .. ),
               where (a, b, c) = (filter width, num of filters, stride)
    batch_norm : (boolean) if batch norm applied after each convolution.
    stim_in : stimulus input.
    is_training : (boolean) if training or testing for (batch norm).
    reuse_variables : if using previously-defined variables.
                      Useful for embedding mutliple responses using same graph.

  Returns :
    net : stimulus embedding
  """

  n_layers = int(len(layers)/3)
  tf.logging.info('Number of layers: %d' % n_layers)

  # Set normalization
  if batch_norm:
    normalizer_fn = slim.batch_norm
  else:
    normalizer_fn = None

  tf.logging.info('Logistic activation')

  # Use slim to define multiple layers of convolution.
  net = stim_in
  layer_collection = [net]
  prev_output_sz = input_channels
  for ilayer in range(n_layers):
    if (ilayer == n_layers - 1) or (ilayer == 0):
      activation_fn = None
      activation_str = 'None'
    else:
      activation_fn = tf.nn.softplus
      activation_str = 'Softplus'

    tf.logging.info('Layer: %d Building %s embedding layer: %d, %d, %d, activation %s'
                    % (ilayer, scope_prefix,
                       int(layers[ilayer*3 + 1]), int(layers[ilayer*3]),
                       int(layers[ilayer*3 + 2]), activation_str))
    previous_net = net
    net = slim.conv2d(net, int(layers[ilayer*3 + 1]),
                      int(layers[ilayer*3]),
                      stride=int(layers[ilayer*3 + 2]),
                      scope='%s_wt_%d' % (scope_prefix, ilayer),
                      reuse=reuse_variables,
                      normalizer_fn=normalizer_fn,
                      activation_fn=activation_fn,
                      normalizer_params={'is_training': is_training})

    # Residual layers only for 1st and last layers
    if (prev_output_sz == int(layers[ilayer*3 + 1])) and (int(layers[ilayer*3 + 2]) == 1):

      tf.logging.info('Layer: %d Building %s embedding layer: %d, %d, %d, linear'
                    % (ilayer, scope_prefix,
                       int(layers[ilayer*3 + 1]), int(layers[ilayer*3]),
                       int(layers[ilayer*3 + 2])))
      net = slim.conv2d(net, int(layers[ilayer*3 + 1]),
                        int(layers[ilayer*3]),
                        stride=int(layers[ilayer*3 + 2]),
                        scope='%s_wt_%d_extra' % (scope_prefix, ilayer),
                        reuse=reuse_variables,
                        normalizer_fn=normalizer_fn,
                        activation_fn=None,
                        normalizer_params={'is_training': is_training})

      tf.logging.info('Layer: %d Shortcut added' % ilayer)
      net = net + previous_net

    prev_output_sz = int(layers[ilayer*3 + 1])
    layer_collection += [net]

  return net, layer_collection


def convolutional_decode(input_, layers, scope='stim_decode',
                         reuse_variables=False,
                         is_training=True, first_layer_nl=tf.tanh):
  """Decoding stimulus from embeddedings.

  Each convolutional layer has convolution, batch normalization and soft-plus.
  Args :
    input_ : Embedded stimulus or responeses to decode stimulus from.
    layers : string description of deconvolution layers (in reverse order).
             Format - (a1, b1, c1, a2, b2, c2 .. ),
               where (a, b, c) = (filter width, num of filters, stride)
    scope : scope of variables in network.
    reuse_variables : if using previously-defined variables.
                      Useful for embedding mutliple responses using same graph.

  Returns :
    net : decoded stimulus
  """
  n_layers = int(len(layers)/3)
  net = input_
  layer_collection = [net]
  for ilayer in range(n_layers - 1, -1, -1):  # go in reverse order
    tf.logging.info('Building decoding layer: %d, %d, %d' %
                    (int(layers[ilayer*3 + 1]), int(layers[ilayer*3]),
                     int(layers[ilayer*3 + 2])))

    # TODO(bhaishahster) add normalizer (batch norm) layer in decoding.
    stride = int(layers[ilayer*3 + 2])
    num_outputs = int(layers[ilayer*3 + 1])
    kernel_size = int(layers[ilayer*3])

    if ilayer == 0:
      activation_fn = first_layer_nl
    else:
      activation_fn = tf.nn.softplus

    with slim.arg_scope([slim.conv2d], reuse=reuse_variables,
                        normalizer_fn=slim.batch_norm):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = decode_utils.upsampling(net, kernel_size, stride, num_outputs,
                                      scope + '_%d' % ilayer,
                                      activation_fn=activation_fn)
        layer_collection += [net]

  return net, layer_collection
