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
"""Embed each response using a MultilayeredNN using TF.SLIM and L2 distance.

Each response sequence (cell x time) is passed through a NN and
the final output of NN is the embedding of the response vector.
Given tripelts of {anchor, positive, negative}, find a NN that
d(embed(anchor), embed(positive)) < d(embed(anchor, negative)),
where d(.,.) is the euclidean distance.

Test :
--model='mlnn_slim' --layers='10, 10' --logtostderr --lam_l1=0 \
--data_path='/cns/oi-d/home/bhaishahster/metric_learning/examples_pc2017_04_25_1/' \
--data_train='data012_lr_13_new_cells_groupb_with_stimulus_train.mat' \
--data_test='data012_lr_13_new_cells_groupb_with_stimulus_test.mat' \
--save_suffix='_2017_04_25_1_cells_13_new_groupb_train_mutiple_expts_local_struct' \
--triplet_type='a'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from retina.response_model.python.metric_learning.score_fcns import metric
from retina.response_model.python.metric_learning.score_fcns import mlnn_score
import tensorflow.contrib.slim as slim

class MetricMLNNSlim(mlnn_score.MetricMLNN):
  """Euclidean distance in embedded responses using RNN."""

  def _build_mlnn(self, n_cells, layers, time_window, lr,  is_training):
    """Builds the tensorflow graph for embedding responses and compute dist.

    Args :
        n_cells (int) : Number of cells for response.
        layers (strong) : specification of layers
        time_window (int) : The number of continuous time bins per response.
        lr (float) : Learning rate.

    """

    self.layer_sizes = layers.split(',')
    self.layer_sizes = [np.int(i) for i in self.layer_sizes]
    self.num_layers = len(self.layer_sizes)
    tf.logging.info('Number of layers: %d' % self.num_layers)

    self.anchor = tf.placeholder(shape=[None, n_cells, time_window],
                                 dtype=tf.float32)
    self.pos = tf.placeholder(shape=[None, n_cells, time_window],
                              dtype=tf.float32)
    self.neg = tf.placeholder(shape=[None, n_cells, time_window,],
                              dtype=tf.float32)

    def embed_mlnn(response, reuse_variables=False):
      """Gives RNN embedding of response ( bactch x n_cells x time_window)"""

      activation_fn = self._get_activation_fn()
      layers = self.layer_sizes
      n_layers = len(layers)
      tf.logging.info('Number of layers: %d' % n_layers)
      net = tf.reshape(response, [-1, n_cells * time_window])
      for ilayer in range(n_layers):
        tf.logging.info('Building layer: %d'
                        % (layers[ilayer]))
        net = slim.fully_connected(net, int(layers[ilayer]),
                                   scope='layer_%d' % ilayer,
                                   reuse=reuse_variables,
                                   normalizer_fn=slim.batch_norm,
                                   activation_fn=activation_fn)

      return net

    # embed anchor, positive and negative examples.
    with slim.arg_scope([slim.batch_norm], is_training=is_training):
      self.embed_anchor = embed_mlnn(self.anchor, reuse_variables=False)
      self.embed_pos = embed_mlnn(self.pos, reuse_variables=True)
      self.embed_neg = embed_mlnn(self.neg, reuse_variables=True)

    # Loss for each point max(d(anchor, pos)^2 - d(anchor, neg)^2 + 1, 0)
    self.get_loss()

    # train model
    self.train_step = tf.train.AdagradOptimizer(lr).minimize(self.loss)


  def distance_squared(self, embed1, embed2):
    return tf.reduce_sum((embed1 - embed2)**2, 1)

  def get_loss(self):
    d_an = self.distance_squared(self.embed_anchor, self.embed_neg)
    d_ap = self.distance_squared(self.embed_anchor, self.embed_pos)
    d_pn = self.distance_squared(self.embed_pos, self.embed_neg)
    d_n = tf.minimum(d_an, d_pn)  # Error ?

    # Loss for each point max(d(anchor, pos)^2 - d(anchor, neg)^2 + 1, 0)
    self.loss = tf.reduce_sum(tf.nn.relu(d_ap - d_n + 1), 0)

  def _get_activation_fn(self):
    return tf.nn.relu
