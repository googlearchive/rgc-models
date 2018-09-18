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
r"""Distance s(x,y) = (x-y)'A(x-y) with A>=0, all negatives for each anchor.

Example of how to run :
metric_learn \
--model='quadratic_psd_neg_batch' --logtostderr --lam_l1=0 \
--data_path='/cns/oi-d/home/bhaishahster/metric_learning/examples_pc2017_04_25_1/' \
--data_train='data012_lr_13_new_cells_groupb_with_stimulus_train.mat' \
--data_test='data012_lr_13_new_cells_groupb_with_stimulus_test.mat' \
--save_suffix='_2017_04_25_1_cells_13_new_groupb_train_mutiple_expts_local_struct' \
--triplet_type='batch' --beta=10 \
--batch_size_train=1000 --batch_size_train=100 --learning_rate=0.01 --max_iter=40000
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import retina.response_model.python.metric_learning.score_fcns.quadratic_score_psd as quad_psd


class QuadraticScorePSDNegBatch(quad_psd.QuadraticScorePSD):
  """Score of form x'Ax, with A P.S.D."""

  def _build_graph(self, n_cells, time_window, lr, lam_l1, beta):
    """Build tensorflow graph.

    Args :
      n_cells : number of cells (int)
      time_window : number of time bins in each response vector (int)
      lr : learning rate (float)
      lam_l1 : regularization on entires of A (float)
    """

    # declare variables
    self.dim = n_cells*time_window
    self.lam_l1 = lam_l1
    self.beta = beta
    self.A_symm = tf.Variable(0.001 * np.eye(self.dim).astype(np.float32),
                    name='A_symm') # initialize Aij to be zero

    # placeholders for anchor, pos and neg
    self.anchor = tf.placeholder(dtype=tf.float32,
                                 shape=[None, n_cells, time_window],
                                 name='anchor')
    self.pos = tf.placeholder(dtype=tf.float32,
                              shape=[None, n_cells, time_window], name='pos')
    self.neg = tf.placeholder(dtype=tf.float32,
                              shape=[None, n_cells, time_window], name='neg')

    self.score_anchor_pos = self.get_score(self.anchor, self.pos)
    self.score_anchor_neg = self.get_score_all_pairs(self.anchor, self.neg)

    self.get_loss()
    self.train_step = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    # do projection to A_symm symmetric
    project_A_symm = tf.assign(self.A_symm,
                               (self.A_symm + tf.transpose(self.A_symm))/2)

    # remove negative eigen vectors.
    with tf.control_dependencies([project_A_symm]):
      e, v = tf.self_adjoint_eig(self.A_symm)
      e_pos = tf.nn.relu(e)
      A_symm_new = tf.matmul(tf.matmul(v, tf.diag(e_pos)), tf.transpose(v))
      project_A_PSD = tf.assign(self.A_symm, A_symm_new)

    # combine all projection operators into one.
    self.project_A = tf.group(project_A_symm, project_A_PSD)


  def get_loss(self):
    """Get loss."""
    d_ap = self.score_anchor_pos
    pairwise_an = self.score_anchor_neg
    '''
    self.loss = (tf.reduce_sum(self.beta *
                               tf.reduce_logsumexp(tf.expand_dims(d_ap /
                                                                  self.beta,
                                                                  1) -
                                                   pairwise_an / self.beta, 1),
                               0) +
                 self.lam_l1*tf.reduce_sum(tf.abs(self.A_symm)))
    '''

    # Loss changed on Oct 21, 2017
    beta = self.beta
    difference = (tf.expand_dims(d_ap/beta, 1) - pairwise_an/beta) # postives x negatives
    # log-sum-exp loss
    # log(\sum_j(exp(d+ - dj-)))
    # loss = tf.reduce_sum(beta * tf.reduce_logsumexp(difference, 1), 0)

    # # log(1 + \sum_j(exp(d+ - dj-)))
    difference_padded = tf.pad(difference, [[0, 0], [0, 1]])
    loss = (tf.reduce_sum(beta * tf.reduce_logsumexp(difference_padded, 1), 0) +
            self.lam_l1*tf.reduce_sum(tf.abs(self.A_symm)))
    self.loss = loss

  def get_score_all_pairs(self, anchor, neg):
    """Setup tensorflow graph for distance between anchor and all negativess.

    The two response arrays are (batch x cells x time_window) return distances
    between corresponding pairs of responses in anchor and negs.

    First, we flatten each response array into size (batch x dim),
    where dim = cells*time_window.
    Then, we compute distance(i, j) = (ai-nj)'A(ai-nj),
    where ai and ni are the ith & jth row of flattened anchor and neg.

    Args:
        anchor : Response tensor (each : batch x cells x time_window).
        neg : Responses that are negatives for all anchors.

    Returns:
        distances : evaluated distances of size (batch).
    """

    anchor_flat = tf.reshape(anchor, [-1, self.dim])
    neg_flat = tf.reshape(neg, [-1, self.dim])

    anchor_expand = tf.expand_dims(anchor_flat, 1)
    neg_expand = tf.expand_dims(neg_flat, 0)
    delta_anchor_neg = anchor_expand - neg_expand  # anchors x negatives x dim
    # A_symm_3d = tf.tile(tf.expand_dims(self.A_symm, 0),
    #                       [tf.shape(delta_anchor_neg)[0], 1, 1])

    delta_anchor_neg_2d = tf.reshape(delta_anchor_neg,
                                     [tf.shape(delta_anchor_neg)[0] *
                                      tf.shape(delta_anchor_neg)[1],
                                      -1])
    score_anchor_neg = tf.reduce_sum(tf.multiply((delta_anchor_neg_2d),
                                                 tf.matmul(delta_anchor_neg_2d,
                                                           self.A_symm)), 1)
    score_anchor_neg_2d = tf.reshape(score_anchor_neg,
                                     [tf.shape(delta_anchor_neg)[0] ,
                                      tf.shape(delta_anchor_neg)[1]])
    return score_anchor_neg_2d

