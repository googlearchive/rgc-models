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
"""Learn a score function of form s(x,y) = (x-y)'A(x-y) with Aij=0 if cells far.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import retina.response_model.python.metric_learning.score_fcns.quadratic_score as quad


class QuadraticScoreConstrained(quad.QuadraticScore):
  """Score of form x'Ax, with A constrained."""

  def _build_graph(self, n_cells, time_window, lr, lam_l1,
                   cell_centers, neighbor_threshold):
    """Build tensorflow graph.

    Args :
      n_cells : number of cells (int)
      time_window : number of time bins in each response vector (int)
      lr : learning rate (float)
      lam_l1 : regularization on entires of A (float)
      cell_centers : centers of cells (n_cells x 2)
      neighbor_threshold : Aij=0 if
                           distance(cell(i), cell(j)) > neighbor_threshold.
    """

    # make projection operator to make Aij=0 for cells far away
    cell_distances = np.sqrt(np.sum((np.expand_dims(cell_centers, 1) -
                                     np.expand_dims(cell_centers, 0))**2, 2))
    self.cell_pairs = np.double(cell_distances <
                                neighbor_threshold).astype(np.float32)

    # declare variables
    self.dim = n_cells*time_window
    A = tf.Variable(np.random.randn(self.dim, self.dim).astype(np.float32) *
                    self.cell_pairs,
                    name='A')  # initialize Aij to be zero
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

    # project A to space where distance between cells is high.
    self.project_A = tf.assign(A, tf.multiply(A, self.cell_pairs))

  def update(self, triplet_batch):
    """Given a batch of training data, update metric parameters.

    Args :
        triplet_batch : List [anchor, positive, negative], each with shape:
                        (batch x cells x time_window)
    Returns :
        loss : Training loss for the batch of data.
    """

    # project A onto relevant contraint set
    self.sess.run(self.project_A)

    # make gradient step.
    feed_dict = {self.anchor: triplet_batch[0],
                 self.pos: triplet_batch[1],
                 self.neg: triplet_batch[2]}
    _, loss = self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)

    return loss
