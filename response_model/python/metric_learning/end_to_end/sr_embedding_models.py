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
r"""Make a convolutional stimulus-response embedding graph in tensorflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
import retina.response_model.python.metric_learning.end_to_end.losses as losses
import retina.response_model.python.metric_learning.end_to_end.response_embedding as resp
import retina.response_model.python.metric_learning.end_to_end.stimulus_embedding as stim


FLAGS = tf.app.flags.FLAGS


def convolutional_embedding(model_id, sess, is_training, dimx, dimy):
  """Convolutional embedding of stimulus and response.

  Args:
    model_id : The variant of convolutional model to train.
    sess : Tensorflow session.
    is_training : Either training or evaluation mode.
    dimx: X dimension of the stimulus.
    dimy: Y dimension of the stimulus.

  Returns:
    sr_graph : Container of the embedding parameters and losses.
  """

  # Embed stimulus.
  stim_tf = tf.placeholder(tf.float32,
                           shape=[None, dimx,
                                  dimy, 30])  # batch x X x Y x time_window
  stim_embed, stim_layers = stim.convolutional_encode(
      FLAGS.stim_layers.split(','), FLAGS.batch_norm, stim_tf, is_training,
      reuse_variables=False)

  # Embed responses.
  layers = FLAGS.resp_layers
  # format: window x filters x stride
  # NOTE: final layer - filters=1, stride =1 throughout
  batch_norm = FLAGS.batch_norm

  anchor_model = resp.Convolutional(time_window=1,
                                    layers=layers,
                                    batch_norm=batch_norm,
                                    is_training=is_training,
                                    reuse_variables=False,
                                    num_cell_types=2,
                                    dimx=dimx, dimy=dimy)

  # Use same model to embed negative responses using reuse_variables=True.
  neg_model = resp.Convolutional(time_window=1,
                                 layers=layers,
                                 batch_norm=batch_norm,
                                 is_training=is_training,
                                 reuse_variables=True,
                                 num_cell_types=2,
                                 dimx=dimx, dimy=dimy)

  # Define loss.
  if model_id == 'convolutional_embedding':
    op = losses.log_sum_exp(stim_embed, anchor_model.responses_embed,
                            neg_model.responses_embed, FLAGS.beta)
    loss, accuracy_tf, d_s_r_pos, d_pairwise_s_rneg = op

  # Define the training op.
  train_op = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(loss)

  # Store everything in a graph.
  sr_graph = collections.namedtuple('SR_Graph', 'sess train_op anchor_model'
                                    ' neg_model d_s_r_pos d_pairwise_s_rneg '
                                    'loss accuracy_tf stim_tf stim_embed')
  sr_graph = sr_graph(sess=sess, train_op=train_op, anchor_model=anchor_model,
                      neg_model=neg_model,
                      d_s_r_pos=d_s_r_pos,
                      d_pairwise_s_rneg=d_pairwise_s_rneg,
                      loss=loss, accuracy_tf=accuracy_tf, stim_tf=stim_tf,
                      stim_embed=stim_embed)

  return sr_graph


