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
"""Embed each response using a MultilayeredNN, take euclidean distance and symmetrize.

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
from retina.response_model.python.metric_learning.score_fcns import mlnn_score

class MetricMLNNSymm(mlnn_score.MetricMLNN):
  """Euclidean distance in embedded responses using RNN."""

  def distance_squared(self, embed1, embed2):
    return tf.reduce_sum((embed1 - embed2)**2, 1)

  def get_loss(self):

    d_an = self.distance_squared(self.embed_anchor, self.embed_neg)
    d_ap = self.distance_squared(self.embed_anchor, self.embed_pos)
    d_pn = self.distance_squared(self.embed_pos, self.embed_neg)
    d_n = tf.minimum(d_an, d_pn)  # Error ?

    # Loss for each point max(d(anchor, pos)^2 - d(anchor, neg)^2 + 1, 0)
    self.loss = tf.reduce_sum(tf.nn.relu(d_ap - d_n + 1), 0)
