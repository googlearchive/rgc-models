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
"""Embed each response using a MultilayeredNN and then take euclidean distance.

Each response sequence (cell x time) is passed through a NN and
the final output of NN is the embedding of the response vector.
Given tripelts of {anchor, positive, negative}, find a NN that
d(embed(anchor), embed(positive)) < d(embed(anchor, negative)),
where d(.,.) is the euclidean distance.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from retina.response_model.python.metric_learning.score_fcns import metric


class MetricMLNN(metric.Metric):
  """Euclidean distance in embedded responses using RNN."""

  def __init__(self, sess, save_folder, file_name, **kwargs):
    """Initialize RNN for learning the response metric.

    Args :
        sess : Tensorflow session to initialize the model with.
        **kwargs : arguments for building RNN.
    """

    tf.logging.info('MLNN score')
    self._build_mlnn(**kwargs)

    self.build_summaries()
    tf.logging.info('Summary operatory made')

    self.sess = sess
    self.initialize_model(save_folder, file_name, sess)
    tf.logging.info('Model initialized')


  def _build_mlnn(self, n_cells, layers, time_window, lr, is_training):
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

    def embed_mlnn(x_in):
      """Gives RNN embedding of x_in."""

      def layer_mlnn(in_layer, in_size, out_size, ilayer):
        w = tf.get_variable('w_%d' % ilayer, shape=[in_size, out_size],
                            initializer=tf.random_normal_initializer())

        b = tf.get_variable('b_%d'% ilayer, shape=[out_size],
                            initializer=tf.random_normal_initializer())

        return tf.nn.relu(tf.matmul(in_layer, w) + b)

      # flatten x into n_cells*time entries
      x_flat = tf.reshape(x_in, [-1, n_cells * time_window])

      # now embed using multilayered NN
      in_size= n_cells
      in_layer = x_flat
      for ilayer in range(self.num_layers):
        out_size = self.layer_sizes[ilayer]
        out_layer = layer_mlnn(in_layer, in_size, out_size, ilayer)
        in_size = out_size
        in_layer = out_layer

      return out_layer

    with tf.variable_scope("MLNN") as scope:
      # embed anchor, positive and negative examples.
      self.embed_anchor = embed_mlnn(self.anchor)

      # This enables reusing the NN for anchor, pos and neg.
      scope.reuse_variables()
      # Check that we are reusing the same RNN using:
      # print(tf.get_variable_scope().reuse)

      self.embed_pos = embed_mlnn(self.pos)
      self.embed_neg = embed_mlnn(self.neg)

    # Loss for each point max(d(anchor, pos)^2 - d(anchor, neg)^2 + 1, 0)
    self.get_loss()

    # train model
    self.train_step = tf.train.AdagradOptimizer(lr).minimize(self.loss)


  def distance_squared(self, embed1, embed2):
    return tf.reduce_sum((embed1 - embed2)**2, 1)

  def get_loss(self):
    # Loss for each point max(d(anchor, pos)^2 - d(anchor, neg)^2 + 1, 0)
    self.loss = tf.reduce_sum(tf.nn.relu(self.distance_squared(self.embed_anchor,
                                                       self.embed_pos) -
                                         self.distance_squared(self.embed_anchor,
                                                       self.embed_neg)
                                      + 1), 0)

  def update(self, triplet_batch):
    """Given a batch of training data, update metric parameters.

    Args :
        triplet_batch : List [anchor, positive, negative], each with shape:
                        (batch x cells x time_window)
    Returns :
        loss : Training loss for the batch of data.
    """

    feed_dict = {self.anchor: triplet_batch[0],
                 self.pos: triplet_batch[1],
                 self.neg: triplet_batch[2]}
    _, loss = self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)
    return loss

  def get_distance(self, resp1, resp2):
    """Give distances between pairs in two sets of responses.

    The two response arrays are (batch x cells x time_window)
    Return distances between corresponding pairs of responses in resp1, resp2.

    Args:
        resp1 : Binned responses (each : batch x cells x time_window).
        resp2 : same as resp 1.

    Returns :
        distances : evaluated distances of size (batch).
    """
    feed_dict = {self.anchor: resp1}
    embed1 = self.sess.run(self.embed_anchor, feed_dict=feed_dict)

    feed_dict = {self.anchor: resp2}
    embed2 = self.sess.run(self.embed_anchor, feed_dict=feed_dict)

    return np.sqrt(np.sum((embed1-embed2)**2, 1))

  def get_embedding(self, resp):
    """Give embedding of the response using RNN.

       Args :
           resp (float32) : Binned responses shape = [batch, # cells,
                              time_window].

       Returns :
           embedding (float32) : embedding of each example in resp
                                   shape = [batch, #hidden layer size].
    """

    feed_dict = {self.anchor: resp}
    embedding = self.sess.run(self.embed_anchor, feed_dict=feed_dict)
    return embedding

  def build_summaries(self):
    """Add some summaries."""

    # Loss summary.
    tf.summary.scalar('loss', self.loss)

    merged = tf.summary.merge_all()
    self.summary_op = merged
    tf.logging.info('summary OP set')


  # TODO(bhaishahster) : Add function to initialize the graph one saved earlier

  def get_parameters(self):
    """Return parameters of the metric."""

    variables = tf.global_variables()
    params = {}
    for ivar in variables:
      # print(ivar.name)
      # print(self.sess.run(ivar))
      params.update({ivar.name: self.sess.run(ivar)})

    return params
