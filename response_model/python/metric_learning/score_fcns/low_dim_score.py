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
"""Learn a distance function of form d(x,y) = Dxy, where D=A'A (A tall)

--model='low_dim_score' --logtostderr --score_mat_dim=20 \
--data_path='/cns/oi-d/home/bhaishahster/metric_learning/examples_pc2017_04_25_1/' \
--data_train='data012_lr_13_new_cells_groupb_with_stimulus_train.mat' \
--data_test='data012_lr_13_new_cells_groupb_with_stimulus_test.mat' \
--save_suffix='_2017_04_25_1_cells_13_new_groupb_train_mutiple_expts' --gfs_user='foam-brain-gpu' \
--triplet_type='a'

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from retina.response_model.python.metric_learning.score_fcns import metric


class LowDimScore(metric.Metric):
  """Distance d(x,y) = Dxy, where D=A'A (A tall)."""

  def __init__(self, sess, save_folder, file_name, **kwargs):
    """Initialize the tensorflow model to learn the metric."""

    tf.logging.info('Building graph for low dimensional score metric')
    self._build_graph(**kwargs)

    self.build_summaries()
    tf.logging.info('Summary operator made')

    self.sess = sess
    self.initialize_model(save_folder, file_name, sess)
    tf.logging.info('Model initialized')

  def _build_graph(self, n_cells, time_window, lr, dim_lr=10):

    # declare variables

    self.dim = n_cells*time_window
    self.weights = tf.expand_dims((2**np.arange(self.dim)).astype(np.float32), 1)

    SCALEA = 0.1
    self.A = tf.Variable(SCALEA * np.ones((2**self.dim, dim_lr)).astype(np.float32), name='A')

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

    self.loss = (tf.reduce_sum(tf.nn.relu(- self.score_anchor_pos +
                                          self.score_anchor_neg + 1)))

    self.train_step = tf.train.AdagradOptimizer(lr).minimize(self.loss)

  def get_score(self, anchor, pos):
    """Setup tensorflow graph for distance between pairs of responses.

    The two response arrays are (batch x cells x time_window) return distances
    between corresponding pairs of responses in anchor and pos.

    First, we flatten each response array into size (batch x dim),
    where dim = cells*time_window.
    We find index of each response (anchor/postivie) in 2^dim space.
    Then, we compute
    distance(i) = sum(A(index(anchor(i)), :) * A(index(pos(i)), :), 1)

    Args:
        anchor : Response tensor (each : batch x cells x time_window).
        pos : same as anchor.

    Returns:
        distances : evaluated distances of size (batch).
    """

    anchor = tf.cast(anchor > 0, tf.float32)
    pos = tf.cast(pos > 0, tf.float32)

    anchor_flat = tf.reshape(anchor, [-1, self.dim])
    pos_flat = tf.reshape(pos, [-1, self.dim])

    anchor_idx = tf.squeeze(tf.matmul(anchor_flat, self.weights), 1)
    pos_idx = tf.squeeze(tf.matmul(pos_flat, self.weights), 1)

    # Make all indices to 0/1

    anchor_idx = tf.cast(anchor_idx, tf.int32)
    d_anchor = tf.gather(self.A, anchor_idx)

    pos_idx = tf.cast(pos_idx, tf.int32)
    d_pos = tf.gather(self.A, pos_idx)

    distances = tf.reduce_sum(d_anchor * d_pos, 1)
    return distances

  def get_parameters(self):
    """Return insightful parameters of the score function."""
    return self.sess.run(self.A)

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
    distances = - self.sess.run(self.score_anchor_pos, feed_dict=feed_dict)
    return distances

  def build_summaries(self):
    """Add loss summary operator."""

    # Loss summary.
    tf.summary.scalar('loss', self.loss)

    merged = tf.summary.merge_all()
    self.summary_op = merged
    tf.logging.info('summary op set')
