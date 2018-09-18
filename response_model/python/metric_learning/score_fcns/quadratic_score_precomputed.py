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
"""Learn a precomputed score function of form s(x,y) = (x-y)'A(x-y).

Example of how to run :
metric_eval --model='quadratic_psd' --logtostderr --lam_l1=0 --data_path='/cns/oi-d/home/bhaishahster/metric_learning/examples_pc2017_04_25_1/' --data_train='data012_lr_15_cells_groupb.mat' --data_test='data012_lr_15_cells_groupb_with_stimulus.mat' --save_suffix='_2017_04_25_1_cells_15_groupb' --gfs_user='foam-brain-gpu' --triplet_type='a'

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import scipy.io as sio
from tensorflow.python.platform import gfile
import retina.response_model.python.metric_learning.score_fcns.quadratic_score as quad

class QuadraticScorePrecomputed(quad.QuadraticScore):
  """Score of form x'Ax, with A constrained."""

  def _build_graph(self, n_cells, time_window, precomputed_a_loc):

    # declare variables
    self.dim = n_cells*time_window

    tf.logging.info('A loading from %s' % precomputed_a_loc)
    data_a = sio.loadmat(gfile.Open(precomputed_a_loc))
    A_loaded = data_a['A']


    self.A_symm = tf.Variable(A_loaded.astype(np.float32),
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
    self.score_anchor_neg = self.get_score(self.anchor, self.neg)


