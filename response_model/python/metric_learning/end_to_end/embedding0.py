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
"""Embed stimulus, response and EIs"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
FLAGS = flags.FLAGS

def embed_responses(embedx, embedy, ei_embedding, is_training=True):
  """Embed responses."""

  # Embed responses
  responses_tf = tf.placeholder(tf.float32)  # Batchsz x n_cells
  ei_embedding_2d = tf.reshape(ei_embedding, [embedx*embedy, -1])
  responses_embedding_2d = tf.matmul(responses_tf, tf.transpose(ei_embedding_2d))
  responses_embedding = tf.reshape(responses_embedding_2d, [-1, embedx, embedy])

  return responses_embedding, responses_tf


def embed_ei(embedx, embedy, eix, eiy, n_elec, ei_embedding_matrix, is_training=True):
  """Embed EIs."""

  # EI -> receptive fields
  ei_tf = tf.placeholder(tf.float32, shape = [n_elec, None]) # n_elec x # cells
  ei_embed_tf = tf.constant(ei_embedding_matrix.astype(np.float32), name='ei_embedding')  # eix x eiy x n_elec
  ei_embed_2d_tf = tf.reshape(ei_embed_tf, [eix * eiy, n_elec])
  ei_embed = tf.matmul(ei_embed_2d_tf, ei_tf)
  ei_embed_3d = tf.reshape(ei_embed, [eix, eiy, -1])

  # make a embed using slim
  net = tf.expand_dims(tf.transpose(ei_embed_3d, [2, 0, 1]), 3)

  # Get RF map from EI
  n_repeats = 3
  output_size = 5
  conv_sz = 5
  with tf.name_scope('ei_model'):
    # pass EI through a few layers of convolutions
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.005),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training}
                        ):
      net =  slim.repeat(net, n_repeats, slim.conv2d, output_size,
                         [conv_sz, conv_sz], scope='conv_ei')


    # Rotate the image.
    rotation_global_ei = tf.Variable(0., name='rotate_ei')  # probably better to rotate the stimulus.
    # Letting TF-Slim know about the additional variable.
    slim.add_model_variable(rotation_global_ei)
    net = tf.contrib.image.rotate(net, rotation_global_ei)

    # scale image
    net = tf.image.resize_images(net, [embedx, embedy])
    ei_embedding = tf.transpose(tf.reduce_sum(net, 3), [1, 2, 0]) # embedx x embedy x n_cells

  return ei_embedding, ei_tf


def embed_stimulus(embedx, embedy, stimx, stimy, stim_history=30, is_training=True):
  """Embed stimulus"""

  # Embed stimulus
  stim_tf = tf.placeholder(tf.float32, shape = [None, stimx, stimy, stim_history])
  net = stim_tf
  n_repeats = 3
  output_size = 5
  conv_sz = 5
  with tf.name_scope('stim_model'):
    # pass EI through a few layers of convolutions
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.005),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training}):
      net =  slim.repeat(net, n_repeats, slim.conv2d, output_size,
                         [conv_sz, conv_sz], scope='conv_stim')
      net = slim.conv2d(net, 1, [conv_sz, conv_sz], scope='conv_stim_final')

      # Rotate stimulus
      rotation_global_stim = tf.Variable(0., name='rotate_stim')  # probably better to rotate the stimulus.
      # Letting TF-Slim know about the additional variable.
      slim.add_model_variable(rotation_global_stim)
      net = tf.contrib.image.rotate(net, rotation_global_stim)

      # Scale stimulus
      net = tf.image.resize_images(net, [embedx, embedy])

      stimulus_embedding = tf.reduce_sum(net, 3)

    return stimulus_embedding, stim_tf



if __name__ == '__main__':
  app.run(main)
