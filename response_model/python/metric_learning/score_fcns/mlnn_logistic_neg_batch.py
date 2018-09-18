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
"""Embed each response using a MultilayeredNN and use softplus loss & activation.

Each response sequence (cell x time) is passed through a NN and
the final output of NN is the embedding of the response vector.
Given tripelts of {anchor, positive, negative}, find a NN that
d(embed(anchor), embed(positive)) < d(embed(anchor, negative)),
where d(.,.) is the euclidean distance.

Test :
--model='mlnn_logistic_neg_batch' --layers='15, 15' --logtostderr --lam_l1=0 \
--data_path='/cns/oi-d/home/bhaishahster/metric_learning/examples_pc2017_04_25_1/' \
--data_train='data012_lr_13_new_cells_groupb_with_stimulus_train.mat' \
--data_test='data012_lr_13_new_cells_groupb_with_stimulus_test.mat' \
--save_suffix='_2017_04_25_1_cells_13_new_groupb_train_mutiple_expts_local_struct' \
--triplet_type='batch'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from retina.response_model.python.metric_learning.score_fcns import metric
from retina.response_model.python.metric_learning.score_fcns import mlnn_slim
import tensorflow.contrib.slim as slim

class MetricMLNNLogisticAllNegBatch(mlnn_slim.MetricMLNNSlim):
  """Euclidean distance in embedded responses using RNN."""

  def __init__(self, sess, save_folder, file_name, beta, **kwargs):
    """Initialize MLNN.

    Args :
        sess : Tensorflow session to initialize the model with.
        **kwargs : arguments for building RNN.
    """

    self.beta = beta
    tf.logging.info('Setting beta to %.5f' % beta)
    super(MetricMLNNLogisticAllNegBatch, self).__init__(sess, save_folder,
                                                        file_name, **kwargs)

  def get_loss(self):

    # from IPython import embed; embed()

    d_ap = self.distance_squared(self.embed_anchor, self.embed_pos)
    # d_an = self.distance_squared(self.embed_anchor, self.embed_neg)
    pairwise_an = tf.reduce_sum((tf.expand_dims(self.embed_anchor, 1) -
                                 tf.expand_dims(self.embed_neg, 0))**2, 2)
    self.loss = tf.reduce_sum(self.beta * tf.reduce_logsumexp(tf.expand_dims(d_ap / self.beta, 1) -
                                                  pairwise_an / self.beta, 1), 0)

  def _get_activation_fn(self):
    return tf.nn.softplus
