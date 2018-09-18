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
r"""Learn a distance function of form d(x,y) = (x-y)'A(x-y), where A=A'.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from retina.response_model.python.metric_learning.score_fcns import metric

class QuadraticScore(metric.Metric):
  """Quadratic distance of form d(x,y) = (x-y)'A(x-y), with A=A'."""

  def __init__(self, sess, save_folder, file_name, **kwargs):
    """Initialize the tensorflow model to learn the metric."""

    tf.logging.info('Building graph for quadratic metric')
    self._build_graph(**kwargs)

    self.build_summaries()
    tf.logging.info('Summary operator made')

    self.sess = sess
    self.initialize_model(save_folder, file_name, sess)
    tf.logging.info('Model initialized')

  def _build_graph(self, n_cells, time_window, lr, lam_l1):

    # declare variables
    self.dim = n_cells*time_window
    SCALEA = 0.1
    A = tf.Variable(SCALEA * np.eye(self.dim).astype(np.float32), name='A')
    self.A_symm = (A + tf.transpose(A))/2

    # placeholders for anchor, pos and neg
    self.anchor = tf.placeholder(dtype=tf.float32,
                                 shape=[None, n_cells, time_window],
                                 name='anchor')
    self.pos = tf.placeholder(dtype=tf.float32,
                              shape=[None, n_cells, time_window], name='pos')
    self.neg = tf.placeholder(dtype=tf.float32,
                              shape=[None, n_cells, time_window], name='neg')

    self.score_anchor_pos = self.get_score(self.anchor, self.pos)
    self.score_anchor_neg = self.get_score(self.anchor, self.neg)

    self.loss = (tf.reduce_sum(tf.nn.relu(self.score_anchor_pos -
                                          self.score_anchor_neg + 1)) +
                 lam_l1*tf.reduce_sum(tf.abs(self.A_symm)))

    self.train_step = tf.train.AdagradOptimizer(lr).minimize(self.loss)

  def get_score(self, anchor, pos):
    """Setup tensorflow graph for distance between pairs of responses.

    The two response arrays are (batch x cells x time_window) return distances
    between corresponding pairs of responses in anchor and pos.

    First, we flatten each response array into size (batch x dim),
    where dim = cells*time_window.
    Then, we compute distance(i) = (ai-pi)'A(ai-pi),
    where ai and pi are the ith row of flattened anchor and pos.

    Args:
        anchor : Response tensor (each : batch x cells x time_window).
        pos : same as anchor.

    Returns:
        distances : evaluated distances of size (batch).
    """

    anchor_flat = tf.reshape(anchor, [-1, self.dim])
    pos_flat = tf.reshape(pos, [-1, self.dim])
    delta_anchor_pos = anchor_flat - pos_flat
    score_anchor_pos = tf.reduce_sum(tf.multiply((delta_anchor_pos),
                                                 tf.matmul((delta_anchor_pos),
                                                           self.A_symm)), 1)
    return score_anchor_pos

  def get_parameters(self):
    """Return insightful parameters of the score function."""
    return self.sess.run(self.A_symm)

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

  def get_distance(self, anchor_in, pos_in):
    """Compute distance between pairs of responses.

    The two response arrays are (batch x cells x time_window) return distances
    between corresponding pairs of responses in anchor and pos.
    The distances are evaluated using tensorflow model setup.

    Args:
        anchor : Binned responses (each : batch x cells x time_window).
        pos : same as anchor.

    Returns:
        distances : evaluated distances of size (batch).
    """
    feed_dict = {self.anchor: anchor_in, self.pos: pos_in}
    distances = self.sess.run(self.score_anchor_pos, feed_dict=feed_dict)
    return distances

  def get_expected_score(self, firing_rate, response):
    """Get expected distance between a fixed response & samples of firing rate.

       Let r ~ Poisson(firing rate).
       Hence, E[d(r, y)] = E[(r-y)'A(r-y)] = d(E[r], y) + variance term.
       This expected distance is calculated for each pair of firing_rate and
       response in the input.

       Args:
         firing_rate : Poisson response probabilities
                       (batch x cells x time_window).
         response : Binned responses (batch x cells x time_window).

       Returns:
         total_distance : evaluated distances of size (batch).
    """

    firing_rate_flat = tf.reshape(firing_rate, [-1, self.dim])
    response_flat = tf.reshape(response, [-1, self.dim])

    mean = firing_rate_flat
    mean_sq = tf.pow(firing_rate_flat, 2)

    # variance term.
    mean_sq_contrib = tf.reduce_sum(mean_sq * tf.diag_part(self.A_symm), 1)
    # mean term.
    diff_mean_response = mean - response_flat
    expected_score = tf.reduce_sum(tf.multiply(diff_mean_response,
                                               tf.matmul(diff_mean_response,
                                                         self.A_symm)), 1)
    # total expected distance.
    total_distance = mean_sq_contrib + expected_score
    return total_distance

  def build_summaries(self):
    """Add loss summary operator."""

    # Loss summary.
    tf.summary.scalar('loss', self.loss)

    merged = tf.summary.merge_all()
    self.summary_op = merged
    tf.logging.info('summary op set')
