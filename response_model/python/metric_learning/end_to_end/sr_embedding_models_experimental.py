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


def convolutional_embedding_experimental(model_id, sess, is_training, dimx, dimy):
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
  stim_embed, stim_layers = stim.convolutional_encode2(
      FLAGS.stim_layers.split(','), FLAGS.batch_norm, stim_tf, is_training,
      reuse_variables=False)

  # Embed responses.
  layers = FLAGS.resp_layers
  # format: window x filters x stride
  # NOTE: final layer - filters=1, stride =1 throughout
  batch_norm = FLAGS.batch_norm

  anchor_model = resp.Convolutional2(time_window=1,
                                    layers=layers,
                                    batch_norm=batch_norm,
                                    is_training=is_training,
                                    reuse_variables=False,
                                    num_cell_types=2,
                                    dimx=dimx, dimy=dimy)

  # Use same model to embed negative responses using reuse_variables=True.
  neg_model = resp.Convolutional2(time_window=1,
                                 layers=layers,
                                 batch_norm=batch_norm,
                                 is_training=is_training,
                                 reuse_variables=True,
                                 num_cell_types=2,
                                 dimx=dimx, dimy=dimy)


  if model_id == 'convolutional_embedding_gauss_expt':
    # add variance layer for stimulus and variance
    slim = tf.contrib.slim
    stim_layers_params = FLAGS.stim_layers.split(',')
    resp_layers_params = FLAGS.resp_layers.split(',')

    with slim.arg_scope([slim.conv2d],
                        normalizer_fn=slim.batch_norm,
                        activation_fn=tf.nn.softplus):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):

        # Variance of stimulus embedding.
        print('Adding stimulus variance layer_expt', stim_layers_params[-3:])
        with slim.arg_scope([slim.conv2d], scope='stim_variance',
                            num_outputs=int(stim_layers_params[-2]),
                            kernel_size=int(stim_layers_params[-3]),
                            stride=int(stim_layers_params[-1])):
          stim_variance = slim.conv2d(stim_layers[-2], reuse=False)

        # Variance of response embedding.
        print('Adding response variance layer_expt', stim_layers_params[-3:])
        with slim.arg_scope([slim.conv2d], scope='resp_variance',
                            num_outputs=int(resp_layers_params[-2]),
                            kernel_size=int(resp_layers_params[-3]),
                            stride=int(resp_layers_params[-1])):
          anchor_model.variance = slim.conv2d(anchor_model.layer_collection[-2],
                                              reuse=False)

          neg_model.variance = slim.conv2d(neg_model.layer_collection[-2],
                                           reuse=True)

  # Define loss.
  if model_id == 'convolutional_embedding_expt':
    op = losses.log_sum_exp(stim_embed, anchor_model.responses_embed,
                            neg_model.responses_embed, FLAGS.beta)
    loss, accuracy_tf, d_s_r_pos, d_pairwise_s_rneg = op

  if model_id == 'convolutional_embedding_margin_expt':
    # Hamming margin - use normalized across retinas with reduce_mean.
    margin_fn = tf.reduce_mean(tf.abs(anchor_model.responses_tf -
                                      neg_model.responses_tf))
    op = losses.log_sum_exp_margin(stim_embed, anchor_model.responses_embed,
                                   neg_model.responses_embed,
                                   FLAGS.beta, margin_fn)
    loss, accuracy_tf, d_s_r_pos, d_pairwise_s_rneg = op

  if model_id == 'convolutional_embedding_inner_product_expt':
    # Use d(s, r) = - phi(s).psi(r).
    op = losses.log_sum_exp_inner_product(stim_embed,
                                          anchor_model.responses_embed,
                                          neg_model.responses_embed, FLAGS.beta)
    loss, accuracy_tf, d_s_r_pos, d_pairwise_s_rneg = op

  if model_id == 'convolutional_embedding_gauss_expt':
    # Use d(s, r) = KL divergence between the gaussians
    loss_fn = losses.log_sum_exp_kl_divergence
    op = loss_fn(stim_embed, anchor_model.responses_embed,
                 neg_model.responses_embed,
                 stim_variance, anchor_model.variance, neg_model.variance,
                 FLAGS.beta)
    loss, accuracy_tf, d_s_r_pos, d_pairwise_s_rneg = op

  if model_id == 'convolutional_embedding_kernel_expt':
    import numpy as np
    kernel = np.ones((10, 10)).astype(np.float32)  # TODO(bhaishahster) : - make it 10?
    op = losses.log_sum_exp_kernel(stim_embed, anchor_model.responses_embed,
                                   neg_model.responses_embed, FLAGS.beta,
                                   kernel)
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


def convolutional_autoembedder(sess, is_training, dimx, dimy,
                               loss='log_sum_exp_inner_product'):
  """Convolutional embedding and de-embeding of stimulus and response.

  Args:
    sess : Tensorflow session.
    is_training : Either training or evaluation mode.
    dimx: X dimension of the stimulus.
    dimy: Y dimension of the stimulus.

  Returns:
    sr_graph : Container of the embedding parameters and losses.
  """

  ## Embed stuff
  # Embed stimulus.
  # anchor/pos stimulus
  stim_tf = tf.placeholder(tf.float32,
                           shape=[None, dimx,
                                  dimy, 30])  # batch x X x Y x time_window

  stim_embed, _ = stim.convolutional_encode(FLAGS.stim_layers.split(','),
                                            FLAGS.batch_norm, stim_tf,
                                            is_training, reuse_variables=False)

  # negative stimulus
  stim_neg_tf = tf.placeholder(tf.float32,
                           shape=[None, dimx,
                                  dimy, 30])  # batch x X x Y x time_window

  stim_embed_neg, _ = stim.convolutional_encode(FLAGS.stim_layers.split(','),
                                                FLAGS.batch_norm, stim_neg_tf,
                                                is_training, reuse_variables=True)

  # Embed responses.
  layers = FLAGS.resp_layers
  # format: window x filters x stride
  # NOTE: final layer - filters=1, stride =1 throughout
  batch_norm = FLAGS.batch_norm

  anchor_model = resp.Convolutional2(time_window=1,
                                    layers=layers,
                                    batch_norm=batch_norm,
                                    is_training=is_training,
                                    reuse_variables=False,
                                    num_cell_types=2,
                                    dimx=dimx, dimy=dimy)

  # Use same model to embed negative responses using reuse_variables=True.
  neg_model = resp.Convolutional2(time_window=1,
                                 layers=layers,
                                 batch_norm=batch_norm,
                                 is_training=is_training,
                                 reuse_variables=True,
                                 num_cell_types=2,
                                 dimx=dimx, dimy=dimy)

  # Define embedding loss.
  # Loss for S, R, R triplet
  # changed from losses.log_sum_exp to log_sum_exp_inner_product on march 10
  if loss == 'log_sum_exp_inner_product':
    loss_fcn = losses.log_sum_exp_inner_product
  if loss == 'log_sum_exp':
    loss_fcn = losses.log_sum_exp

  op = loss_fcn(stim_embed, anchor_model.responses_embed,
                          neg_model.responses_embed, FLAGS.beta)
  loss_triplet, accuracy_tf, d_s_r_pos, d_pairwise_s_rneg = op

  # Loss for R, S, S triplet
  op = loss_fcn(anchor_model.responses_embed, stim_embed,
                                        stim_embed_neg, FLAGS.beta)
  loss_triplet_rss, accuracy_tf_rss, d_r_s_pos_rss, d_pairwise_r_sneg_rss = op


  ## Decode stimulus
  # Decode from stimulus.
  decoder_layers = '1, 30, 1, ' + FLAGS.stim_layers
  stim_decode_from_stim, _ = stim.convolutional_decode(stim_embed,
                                                       decoder_layers.split(','
                                                                           ),
                                                       scope='stim_decode',
                                                       reuse_variables=False,
                                                       is_training=is_training)
  # Decode from positive responses
  stim_decode_from_resp, _ = stim.convolutional_decode(anchor_model.responses_embed,
                                                       decoder_layers.split(','
                                                                           ),
                                                       scope='stim_decode',
                                                       reuse_variables=True,
                                                       is_training=is_training)

  bound = FLAGS.valid_cells_boundary
  loss_stim_decode_from_stim = tf.nn.l2_loss(stim_decode_from_stim[:, bound:80-bound, bound:40-bound, :] - stim_tf[:,  bound:80-bound, bound:40-bound, :])
  loss_stim_decode_from_resp = tf.nn.l2_loss(stim_decode_from_resp[:,  bound:80-bound, bound:40-bound, :] - stim_tf[:,  bound:80-bound, bound:40-bound, :])

  ## Decode responses
  # Decode responses from stimulus.
  decoder_layers = '1, 2, 1, ' + FLAGS.resp_layers
  resp_decode_from_stim, _ = stim.convolutional_decode(stim_embed,
                                                       decoder_layers.split(','
                                                                           ),
                                                       scope='resp_decode',
                                                       reuse_variables=False,
                                                       is_training=is_training,
                                                       first_layer_nl=tf.nn.softplus)
  # Decode responses from positive responses
  resp_decode_from_resp, _ = stim.convolutional_decode(anchor_model.responses_embed,
                                                       decoder_layers.split(','
                                                                           ),
                                                       scope='resp_decode',
                                                       reuse_variables=True,
                                                       is_training=is_training,
                                                       first_layer_nl=tf.nn.softplus)
  # Decode responses from arbitrary embedding
  arbitrary_embedding = tf.placeholder(tf.float32, shape=anchor_model.responses_embed.get_shape())
  resp_decode_from_embedding, _ = stim.convolutional_decode(arbitrary_embedding,
                                                       decoder_layers.split(','
                                                                           ),
                                                       scope='resp_decode',
                                                       reuse_variables=True,
                                                       is_training=is_training,
                                                       first_layer_nl=tf.nn.softplus)

  # Find distances between pairs of arbitrary embeddings
  arbitrary_embedding_1 = tf.placeholder(tf.float32, shape=anchor_model.responses_embed.get_shape())
  arbitrary_embedding_2 = tf.placeholder(tf.float32, shape=anchor_model.responses_embed.get_shape())
  op = loss_fcn(arbitrary_embedding_1,
                                        arbitrary_embedding_2,
                                        arbitrary_embedding_2,
                                        FLAGS.beta)  # repeated arbitrary_embedding_2, as it is used both for compnentwise and all pairs.
  loss_arbitray, accuracy_arbitrary, distances_arbitrary, distances_all_pairs_arbitrary = op

  tfd = tf.expand_dims
  # Poisson response prediction loss (changed on march 10)
  loss_resp_decode_from_stim = (tf.reduce_sum((resp_decode_from_stim -
                                              anchor_model.embed_responses_original * tf.log(resp_decode_from_stim)) * tfd(anchor_model.embed_locations_original, 0)) / tf.reduce_sum(anchor_model.embed_locations_original))

  loss_resp_decode_from_resp = (tf.reduce_sum((resp_decode_from_resp -
                                              anchor_model.embed_responses_original * tf.log(resp_decode_from_resp)) * tfd(anchor_model.embed_locations_original, 0)) / tf.reduce_sum(anchor_model.embed_locations_original))


  # L2 response prediction loss
  # loss_resp_decode_from_stim = tf.nn.l2_loss(resp_decode_from_stim[:,  bound:80-bound, bound:40-bound, :] - anchor_model.responses_embed_1[:,  bound:80-bound, bound:40-bound, :])
  # loss_resp_decode_from_resp = tf.nn.l2_loss(resp_decode_from_resp[:,  bound:80-bound, bound:40-bound, :] - anchor_model.responses_embed_1[:,  bound:80-bound, bound:40-bound, :])

  # Define the training op.
  loss = (FLAGS.scale_triplet * loss_triplet + FLAGS.scale_decode * (loss_stim_decode_from_stim + loss_stim_decode_from_resp)
          + FLAGS.scale_encode * (loss_resp_decode_from_stim + loss_resp_decode_from_resp))

  train_op = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(loss)

  # Store everything in a graph.
  sr_graph = collections.namedtuple('SR_Graph', 'sess train_op anchor_model'
                                    ' neg_model d_s_r_pos d_pairwise_s_rneg '
                                    'loss accuracy_tf loss_triplet '
                                    'loss_stim_decode_from_stim '
                                    'loss_stim_decode_from_resp '
                                    'stim_tf stim_embed stim_decode_from_resp '
                                    'stim_decode_from_stim '
                                    'loss_resp_decode_from_stim '
                                    'loss_resp_decode_from_resp '
                                    'resp_decode_from_stim resp_decode_from_resp '
                                    'resp_decode_from_embedding arbitrary_embedding '
                                    'stim_neg_tf stim_embed_neg '
                                    'loss_triplet_rss accuracy_tf_rss '
                                    'd_r_s_pos_rss d_pairwise_r_sneg_rss '
                                    'arbitrary_embedding_1 arbitrary_embedding_2 '
                                    'loss_arbitray '
                                    'accuracy_arbitrary distances_arbitrary '
                                    'distances_all_pairs_arbitrary')

  sr_graph = sr_graph(sess=sess, train_op=train_op, anchor_model=anchor_model,
                      neg_model=neg_model,
                      d_s_r_pos=d_s_r_pos,
                      d_pairwise_s_rneg=d_pairwise_s_rneg,
                      loss=loss, accuracy_tf=accuracy_tf,
                      loss_triplet=loss_triplet,
                      loss_stim_decode_from_stim=loss_stim_decode_from_stim,
                      loss_stim_decode_from_resp=loss_stim_decode_from_resp,
                      stim_tf=stim_tf,
                      stim_embed=stim_embed,
                      stim_decode_from_resp=stim_decode_from_resp,
                      stim_decode_from_stim=stim_decode_from_stim,
                      loss_resp_decode_from_stim=loss_resp_decode_from_stim,
                      loss_resp_decode_from_resp=loss_resp_decode_from_resp,
                      resp_decode_from_stim=resp_decode_from_stim,
                      resp_decode_from_resp=resp_decode_from_resp,
                      resp_decode_from_embedding=resp_decode_from_embedding,
                      arbitrary_embedding=arbitrary_embedding,
                      stim_neg_tf=stim_neg_tf, stim_embed_neg=stim_embed_neg,
                      loss_triplet_rss=loss_triplet_rss,
                      accuracy_tf_rss=accuracy_tf_rss,
                      d_r_s_pos_rss=d_r_s_pos_rss,
                      d_pairwise_r_sneg_rss=d_pairwise_r_sneg_rss,
                      arbitrary_embedding_1=arbitrary_embedding_1,
                      arbitrary_embedding_2=arbitrary_embedding_2,
                      loss_arbitray=loss_arbitray,
                      accuracy_arbitrary=accuracy_arbitrary,
                      distances_arbitrary=distances_arbitrary,
                      distances_all_pairs_arbitrary=distances_all_pairs_arbitrary)

  return sr_graph

def residual_experimental(model_id, sess, is_training, dimx, dimy):
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
  stim_embed, stim_layers = stim.residual_encode(
      FLAGS.stim_layers.split(','), FLAGS.batch_norm, stim_tf, is_training,
      reuse_variables=False, scope_prefix='stim', input_channels=30)

  # Embed responses.
  layers = FLAGS.resp_layers
  # format: window x filters x stride
  # NOTE: final layer - filters=1, stride =1 throughout
  batch_norm = FLAGS.batch_norm

  anchor_model = resp.Residual(time_window=1,
                               layers=layers,
                               batch_norm=batch_norm,
                               is_training=is_training,
                               reuse_variables=False,
                               num_cell_types=2,
                               dimx=dimx, dimy=dimy)

  # Use same model to embed negative responses using reuse_variables=True.
  neg_model = resp.Residual(time_window=1,
                            layers=layers,
                            batch_norm=batch_norm,
                            is_training=is_training,
                            reuse_variables=True,
                            num_cell_types=2,
                            dimx=dimx, dimy=dimy)

  # Define loss.
  if model_id == 'residual':
    op = losses.log_sum_exp(stim_embed, anchor_model.responses_embed,
                            neg_model.responses_embed, FLAGS.beta)
    loss, accuracy_tf, d_s_r_pos, d_pairwise_s_rneg = op

  if model_id == 'residual_inner_product':
    # Use d(s, r) = - phi(s).psi(r).
    op = losses.log_sum_exp_inner_product(stim_embed,
                                          anchor_model.responses_embed,
                                          neg_model.responses_embed,
                                          FLAGS.beta)
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

# def convolutional_autoembedder_match_retina(sess, is_training, dimx, dimy):
#   """Convolutional autoembedding with matching of responses across retinas.
#
#   Args:
#     sess : Tensorflow session.
#     is_training : Either training or evaluation mode.
#     dimx: X dimension of the stimulus.
#     dimy: Y dimension of the stimulus.
#
#   Returns:
#     sr_graph : Container of the embedding parameters and losses.
#   """
#
#   ## Embed stuff
#   # Embed stimulus.
#   # anchor/pos stimulus
#   stim_tf = tf.placeholder(tf.float32,
#                            shape=[None, dimx,
#                                   dimy, 30])  # batch x X x Y x time_window
#
#   stim_embed, _ = stim.convolutional_encode(FLAGS.stim_layers.split(','),
#                                             FLAGS.batch_norm, stim_tf,
#                                             is_training, reuse_variables=False)
#
#   # negative stimulus
#   stim_neg_tf = tf.placeholder(tf.float32,
#                            shape=[None, dimx,
#                                   dimy, 30])  # batch x X x Y x time_window
#
#   stim_embed_neg, _ = stim.convolutional_encode(FLAGS.stim_layers.split(','),
#                                                 FLAGS.batch_norm, stim_neg_tf,
#                                                 is_training, reuse_variables=True)
#
#   # Embed responses.
#   layers = FLAGS.resp_layers
#   # format: window x filters x stride
#   # NOTE: final layer - filters=1, stride =1 throughout
#   batch_norm = FLAGS.batch_norm
#
#   anchor_model = resp.Convolutional2(time_window=1,
#                                     layers=layers,
#                                     batch_norm=batch_norm,
#                                     is_training=is_training,
#                                     reuse_variables=False,
#                                     num_cell_types=2,
#                                     dimx=dimx, dimy=dimy)
#
#   # Use same model to embed negative responses using reuse_variables=True.
#   neg_model = resp.Convolutional2(time_window=1,
#                                  layers=layers,
#                                  batch_norm=batch_norm,
#                                  is_training=is_training,
#                                  reuse_variables=True,
#                                  num_cell_types=2,
#                                  dimx=dimx, dimy=dimy)
#
#   # Define embedding loss.
#   # Loss for S, R, R triplet
#   # changed from losses.log_sum_exp to log_sum_exp_inner_product on march 10
#   op = losses.log_sum_exp_inner_product(stim_embed, anchor_model.responses_embed,
#                           neg_model.responses_embed, FLAGS.beta)
#   loss_triplet, accuracy_tf, d_s_r_pos, d_pairwise_s_rneg = op
#
#   # Loss for R, S, S triplet
#   op = losses.log_sum_exp_inner_product(anchor_model.responses_embed, stim_embed,
#                                         stim_embed_neg, FLAGS.beta)
#   loss_triplet_rss, accuracy_tf_rss, d_r_s_pos_rss, d_pairwise_r_sneg_rss = op
#
#
#   ## Decode stimulus
#   # Decode from stimulus.
#   decoder_layers = '1, 30, 1, ' + FLAGS.stim_layers
#   stim_decode_from_stim, _ = stim.convolutional_decode(stim_embed,
#                                                        decoder_layers.split(','
#                                                                            ),
#                                                        scope='stim_decode',
#                                                        reuse_variables=False,
#                                                        is_training=is_training)
#   # Decode from positive responses
#   stim_decode_from_resp, _ = stim.convolutional_decode(anchor_model.responses_embed,
#                                                        decoder_layers.split(','
#                                                                            ),
#                                                        scope='stim_decode',
#                                                        reuse_variables=True,
#                                                        is_training=is_training)
#
#   bound = FLAGS.valid_cells_boundary
#   loss_stim_decode_from_stim = tf.nn.l2_loss(stim_decode_from_stim[:, bound:80-bound, bound:40-bound, :] - stim_tf[:,  bound:80-bound, bound:40-bound, :])
#   loss_stim_decode_from_resp = tf.nn.l2_loss(stim_decode_from_resp[:,  bound:80-bound, bound:40-bound, :] - stim_tf[:,  bound:80-bound, bound:40-bound, :])
#
#   ## Decode responses
#   # Decode responses from stimulus.
#   decoder_layers = '1, 2, 1, ' + FLAGS.resp_layers
#   resp_decode_from_stim, _ = stim.convolutional_decode(stim_embed,
#                                                        decoder_layers.split(','
#                                                                            ),
#                                                        scope='resp_decode',
#                                                        reuse_variables=False,
#                                                        is_training=is_training,
#                                                        first_layer_nl=tf.nn.softplus)
#   # Decode responses from positive responses
#   resp_decode_from_resp, _ = stim.convolutional_decode(anchor_model.responses_embed,
#                                                        decoder_layers.split(','
#                                                                            ),
#                                                        scope='resp_decode',
#                                                        reuse_variables=True,
#                                                        is_training=is_training,
#                                                        first_layer_nl=tf.nn.softplus)
#   # Decode responses from arbitrary embedding
#   arbitrary_embedding = tf.placeholder(tf.float32, shape=anchor_model.responses_embed.get_shape())
#   resp_decode_from_embedding, _ = stim.convolutional_decode(arbitrary_embedding,
#                                                        decoder_layers.split(','
#                                                                            ),
#                                                        scope='resp_decode',
#                                                        reuse_variables=True,
#                                                        is_training=is_training,
#                                                        first_layer_nl=tf.nn.softplus)
#
#   tfd = tf.expand_dims
#   # Poisson response prediction loss (changed on march 10)
#   loss_resp_decode_from_stim = (tf.reduce_sum((resp_decode_from_stim -
#                                               anchor_model.embed_responses_original * tf.log(resp_decode_from_stim)) * tfd(anchor_model.embed_locations_original, 0)) / tf.reduce_sum(anchor_model.embed_locations_original))
#
#   loss_resp_decode_from_resp = (tf.reduce_sum((resp_decode_from_resp -
#                                               anchor_model.embed_responses_original * tf.log(resp_decode_from_resp)) * tfd(anchor_model.embed_locations_original, 0)) / tf.reduce_sum(anchor_model.embed_locations_original))
#
#
#   # L2 response prediction loss
#   # loss_resp_decode_from_stim = tf.nn.l2_loss(resp_decode_from_stim[:,  bound:80-bound, bound:40-bound, :] - anchor_model.responses_embed_1[:,  bound:80-bound, bound:40-bound, :])
#   # loss_resp_decode_from_resp = tf.nn.l2_loss(resp_decode_from_resp[:,  bound:80-bound, bound:40-bound, :] - anchor_model.responses_embed_1[:,  bound:80-bound, bound:40-bound, :])
#
#   # Define the training op.
#   loss = (FLAGS.scale_triplet * loss_triplet + FLAGS.scale_decode * (loss_stim_decode_from_stim + loss_stim_decode_from_resp)
#           + FLAGS.scale_encode * (loss_resp_decode_from_stim + loss_resp_decode_from_resp))
#
#   train_op = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(loss)
#
#   # Store everything in a graph.
#   sr_graph = collections.namedtuple('SR_Graph', 'sess train_op anchor_model'
#                                     ' neg_model d_s_r_pos d_pairwise_s_rneg '
#                                     'loss accuracy_tf loss_triplet '
#                                     'loss_stim_decode_from_stim '
#                                     'loss_stim_decode_from_resp '
#                                     'stim_tf stim_embed stim_decode_from_resp '
#                                     'stim_decode_from_stim '
#                                     'loss_resp_decode_from_stim '
#                                     'loss_resp_decode_from_resp '
#                                     'resp_decode_from_stim resp_decode_from_resp '
#                                     'resp_decode_from_embedding arbitrary_embedding '
#                                     'stim_neg_tf stim_embed_neg '
#                                     'loss_triplet_rss accuracy_tf_rss '
#                                     'd_r_s_pos_rss d_pairwise_r_sneg_rss')
#   sr_graph = sr_graph(sess=sess, train_op=train_op, anchor_model=anchor_model,
#                       neg_model=neg_model,
#                       d_s_r_pos=d_s_r_pos,
#                       d_pairwise_s_rneg=d_pairwise_s_rneg,
#                       loss=loss, accuracy_tf=accuracy_tf,
#                       loss_triplet=loss_triplet,
#                       loss_stim_decode_from_stim=loss_stim_decode_from_stim,
#                       loss_stim_decode_from_resp=loss_stim_decode_from_resp,
#                       stim_tf=stim_tf,
#                       stim_embed=stim_embed,
#                       stim_decode_from_resp=stim_decode_from_resp,
#                       stim_decode_from_stim=stim_decode_from_stim,
#                       loss_resp_decode_from_stim=loss_resp_decode_from_stim,
#                       loss_resp_decode_from_resp=loss_resp_decode_from_resp,
#                       resp_decode_from_stim=resp_decode_from_stim,
#                       resp_decode_from_resp=resp_decode_from_resp,
#                       resp_decode_from_embedding=resp_decode_from_embedding,
#                       arbitrary_embedding=arbitrary_embedding,
#                       stim_neg_tf=stim_neg_tf, stim_embed_neg=stim_embed_neg,
#                       loss_triplet_rss=loss_triplet_rss,
#                       accuracy_tf_rss=accuracy_tf_rss,
#                       d_r_s_pos_rss=d_r_s_pos_rss,
#                       d_pairwise_r_sneg_rss=d_pairwise_r_sneg_rss)
#
#   return sr_graph
