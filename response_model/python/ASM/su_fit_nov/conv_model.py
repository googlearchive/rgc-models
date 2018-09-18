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
"""Convolutional model of RGC responses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

# Import module
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


FLAGS = flags.FLAGS
rng = np.random


def convolutional_1layer(stim3d, resp, tms_train, tms_test,
                         dim_filters=3, num_filters=1, strides=1, depth=1,
                         lr=0.1, num_steps_max=100000,
                         eps=1e-9):
  """Convolutional model."""

  # TODO(bhaishahster) : Convergence guarantees
  # TODO(bhaishahster) : deep / multiple filters ?
  # TODO(bhaishahster) : Implementation from Lane's work / Vintch et al.

  with tf.Graph().as_default():
    with tf.Session() as sess:
      # stimulus response inputs
      resp = np.expand_dims(resp, 1)
      stim_dim = [stim3d.shape[1], stim3d.shape[2]]
      resp_tf = tf.placeholder(tf.float32, name='resp')
      stim3d_tf = tf.placeholder(tf.float32, name='stim')
      stim4d_tf = tf.expand_dims(stim3d_tf, 3)

      # convolution
      w = tf.Variable(np.random.randn(dim_filters, dim_filters, 1,
                                      num_filters).astype(np.float32),
                      name='filters')

      out_height = np.int(np.ceil(float(stim_dim[0]) / float(strides)))
      out_width = np.int(np.ceil(float(stim_dim[1]) / float(strides)))
      num_wts = np.int(out_height*out_width*num_filters)
      print(num_wts)
      a = tf.Variable((0.1+np.random.rand(num_wts, 1)).astype(np.float32))

      output = tf.nn.conv2d(stim4d_tf, w, strides=[1, strides, strides, 1],
                            padding='SAME')

      output2d = tf.nn.softplus(tf.reshape(output,
                                           [-1, out_height *
                                            out_width * num_filters]))

      firing_rate = tf.matmul(output2d, tf.abs(a))
      loss = (tf.reduce_mean(firing_rate) -
              tf.reduce_mean(resp_tf*tf.log(firing_rate)))
      update_model = tf.train.AdagradOptimizer(lr).minimize(loss,
                                                            var_list=[w, a])

      feed_dict_test = {stim3d_tf: stim3d[tms_test, :],
                        resp_tf: resp[tms_test, :]}
      sess.run(tf.global_variables_initializer())

      loss_prev_train = np.inf
      loss_train_log = []
      loss_test_log = []

      for iiter in np.arange(num_steps_max):

        feed_dict_train = {stim3d_tf: stim3d[tms_train, :],
                           resp_tf: resp[tms_train, :]}
        _, loss_np_train, _ = sess.run([update_model, loss, firing_rate],
                                       feed_dict=feed_dict_train)

        if np.abs(loss_prev_train - loss_np_train) < eps:
          break

        loss_prev_train = np.copy(loss_np_train)
        loss_np_test = sess.run(loss, feed_dict=feed_dict_test)

        loss_train_log += [loss_np_train]
        loss_test_log += [loss_np_test]

        if iiter % 100 == 99:
          print(iiter, loss_np_train, loss_np_test)

      model_params = [sess.run(w), sess.run(a)]
  return loss_train_log, loss_test_log, model_params


def convolutional(stim3d, resp, tms_train, tms_test,
                  layers,
                  lr=0.1, num_steps_max=100000,
                  eps=1e-9):
  """Convolutional model."""

  # TODO(bhaishahster) : Convergence guarantees
  # TODO(bhaishahster) : deep / multiple filters ?
  # TODO(bhaishahster) : Implementation from Lane's work / Vintch et al.

  layers = layers.split(',')
  print(layers)

  with tf.Graph().as_default():
    with tf.Session() as sess:
      # stimulus response inputs
      resp = np.expand_dims(resp, 1)
      stim_dim = [stim3d.shape[1], stim3d.shape[2]]
      resp_tf = tf.placeholder(tf.float32, name='resp')
      stim3d_tf = tf.placeholder(tf.float32, name='stim',
                                 shape=[None] + stim_dim)
      stim4d_tf = tf.expand_dims(stim3d_tf, 3)

      # convolution
      convolved_stim = embed_network(stim4d_tf, layers)

      # final layer - fully connected
      convolved_stim_flat = tf.contrib.layers.flatten(convolved_stim)
      firing_rate = slim.fully_connected(convolved_stim_flat, 1,
                                         activation_fn=tf.nn.softplus,
                                         scope='final_layer')

      loss = (tf.reduce_mean(firing_rate) -
              tf.reduce_mean(resp_tf*tf.log(firing_rate)))
      update_model = tf.train.AdagradOptimizer(lr).minimize(loss)

      feed_dict_test = {stim3d_tf: stim3d[tms_test, :],
                        resp_tf: resp[tms_test, :]}
      sess.run(tf.global_variables_initializer())

      loss_prev_train = np.inf
      loss_train_log = []
      loss_test_log = []

      for iiter in np.arange(num_steps_max):

        feed_dict_train = {stim3d_tf: stim3d[tms_train, :],
                           resp_tf: resp[tms_train, :]}
        _, loss_np_train, _ = sess.run([update_model, loss, firing_rate],
                                       feed_dict=feed_dict_train)

        if np.abs(loss_prev_train - loss_np_train) < eps:
          break

        loss_prev_train = np.copy(loss_np_train)
        loss_np_test = sess.run(loss, feed_dict=feed_dict_test)

        loss_train_log += [loss_np_train]
        loss_test_log += [loss_np_test]

        if iiter % 100 == 99:
          print(iiter, loss_np_train, loss_np_test)

      model_params = sess.run(slim.get_model_variables())
  return loss_train_log, loss_test_log, model_params


def embed_network(input_net, layers, reuse_variables=False):
  """Convolutional embedding."""

  n_layers = int(len(layers)/3)
  tf.logging.info('Number of layers: %d' % n_layers)

  # set normalization and activation functions
  normalizer_fn = None
  activation_fn = tf.nn.softplus
  tf.logging.info('Softplus activation')

  net = input_net
  for ilayer in range(n_layers):
    tf.logging.info('Building layer: %d, %d, %d'
                    % (int(layers[ilayer*3 + 1]), int(layers[ilayer*3]),
                       int(layers[ilayer*3 + 2])))
    net = slim.conv2d(net, int(layers[ilayer*3 + 1]),
                      int(layers[ilayer*3]),
                      stride=int(layers[ilayer*3 + 2]),
                      scope='layer_wt_%d' % ilayer,
                      reuse=reuse_variables,
                      normalizer_fn=normalizer_fn,
                      activation_fn=activation_fn)
  return net

