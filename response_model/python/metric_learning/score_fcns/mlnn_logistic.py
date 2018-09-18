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
"""Embed each response using a MultilayeredNN and use softplus triplet loss.

Each response sequence (cell x time) is passed through a NN and
the final output of NN is the embedding of the response vector.
Given tripelts of {anchor, positive, negative}, find a NN that
d(embed(anchor), embed(positive)) < d(embed(anchor, negative)),
where d(.,.) is the euclidean distance.

Test :
--model='mlnn_logistic' --layers='10, 10' --logtostderr --lam_l1=0 \
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
from retina.response_model.python.metric_learning.score_fcns import mlnn_slim
import tensorflow.contrib.slim as slim

class MetricMLNNLogistic(mlnn_slim.MetricMLNNSlim):
  """Euclidean distance in embedded responses using RNN."""

  def get_loss(self):

    d_an = self.distance_squared(self.embed_anchor, self.embed_neg)
    d_ap = self.distance_squared(self.embed_anchor, self.embed_pos)

    # Loss for each point max(d(anchor, pos)^2 - d(anchor, neg)^2 + 1, 0)
    self.loss = tf.reduce_sum(tf.nn.softplus(d_ap - d_an + 1), 0)
