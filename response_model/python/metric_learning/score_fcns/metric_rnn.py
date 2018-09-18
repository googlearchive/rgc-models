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
"""Embed each response using a RNN and then take euclidean distance.

Each response sequence (cell x time) is passed through a RNN and
the final output of RNN is the embedding of the response vector.
Given tripelts of {anchor, positive, negative}, find a RNN such that
d(embed(anchor), embed(positive)) < d(embed(anchor, negative)),
where d(.,.) is the euclidean distance.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from retina.response_model.python.metric_learning.score_fcns import metric


class MetricRNN(metric.Metric):
  """Euclidean distance in embedded responses using RNN."""

  def __init__(self, sess, **kwargs):
    """Initialize RNN for learning the response metric.

    Args :
        sess : Tensorflow session to initialize the model with.
        **kwargs : arguments for building RNN.
    """

    outputs = self._build_rnn(**kwargs)
    x_anc, x_pos, x_neg, train, loss, embed_anc, embed_pos, embed_neg = outputs

    self.anchor = x_anc
    self.pos = x_pos
    self.neg = x_neg
    self.embed_anchor = embed_anc
    self.embed_pos = embed_pos
    self.embed_neg = embed_neg
    self.loss = loss
    self.train_step = train

    self.sess = sess

    self.build_summaries()
    tf.logging.info('Summary operatory made')

    self.sess.run(tf.initialize_all_variables())

  def _build_rnn(self, n_cells, hidden_layer_size, num_layers,
                 batch_size, time_window, lr):
    """Builds the tensorflow graph for embedding responses and compute dist.

    Args :
        n_cells (int) : Number of cells for response.
        hidden_layer_size (int) : Size of hidden layer in each RNN cell.
        num_layers (int) : Number of RNN layers stacked on top of each other.
        batch_size (int) : Batch size for training RNN.
        time_window (int) : The number of continuous time bins per response.
        lr (float) : Learning rate.

    Returns :
        x_anchor (float32): Tensorflow placeholder for batch of anchors
                                    (size: batch_size, n_cells, time_window).
        x_pos (float32): Same as 'x_anchor', but for positive examples.
        x_neg (float32): Same as 'x_anchor', but for negative examples.
        train_step (float32): Tensorflow Op to update variables
                                using a training batch.
        loss (float32) : Total loss across batch.
        embed_anchor (float32) : RNN embedding of anchors.
                       (size: batch x hidden_layer_size)
        embed_pos (float32) : RNN embedding of positive examples.
        embed_neg (float32) : RNN embedding of negative examples.
    """

    x_anchor = tf.placeholder(shape=[batch_size, n_cells, time_window],
                              dtype=tf.float32)
    x_pos = tf.placeholder(shape=[batch_size, n_cells, time_window],
                           dtype=tf.float32)
    x_neg = tf.placeholder(shape=[batch_size, n_cells, time_window,],
                           dtype=tf.float32)

    def embed_rnn(x_in):
      """Gives RNN embedding of x_in."""
      x_in_flat = tf.reshape(x_in, [batch_size*time_window, n_cells])
      w_in = tf.get_variable("w_in", shape=[n_cells, hidden_layer_size],
                             initializer=tf.random_normal_initializer())
      b_in = tf.get_variable("b_in", shape=[hidden_layer_size],
                             initializer=tf.random_normal_initializer())
      x_ld_flat = tf.matmul(x_in_flat, w_in) + b_in
      x_ld = tf.reshape(x_ld_flat, [batch_size, time_window, hidden_layer_size])

      # stacked LSTM cells
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size,
                                               forget_bias=0.0,
                                               state_is_tuple=True)
      cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers,
                                         state_is_tuple=True)

      # compute output
      initial_state = cell.zero_state(batch_size, tf.float32)
      outputs, _ = tf.nn.dynamic_rnn(cell, x_ld,
                                     initial_state=initial_state,
                                     time_major=False)

      # reconstruct and compute loss on last output only
      outputs_unp = tf.unpack(tf.transpose(outputs, [1, 0, 2]))
      embedding = outputs_unp[-1]

      return embedding

    def distance_squared(embed1, embed2):
      return tf.reduce_sum((embed1 - embed2)**2, 1)

    with tf.variable_scope("RNN") as scope:

      # embed anchor, positive and negative examples.
      embed_anchor = embed_rnn(tf.transpose(x_anchor, [0, 2, 1]))
      # This enables reusing the RNN for anchor, pos and neg.
      scope.reuse_variables()
      # Check that we are reusing the same RNN using:
      # print(tf.get_variable_scope().reuse)
      embed_pos = embed_rnn(tf.transpose(x_pos, [0, 2, 1]))
      embed_neg = embed_rnn(tf.transpose(x_neg, [0, 2, 1]))

      # Loss for each point max(d(anchor, pos)^2 - d(anchor, neg)^2 + 1, 0)
      loss = tf.reduce_sum(tf.nn.relu(distance_squared(embed_anchor,
                                                       embed_pos) -
                                      distance_squared(embed_anchor,
                                                       embed_neg)
                                      + 1), 0)
      # train model
      train_step = tf.train.AdagradOptimizer(lr).minimize(loss)

    return (x_anchor, x_pos, x_neg, train_step, loss,
            embed_anchor, embed_pos, embed_neg)

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
