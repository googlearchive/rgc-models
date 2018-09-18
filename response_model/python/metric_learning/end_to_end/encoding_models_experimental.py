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
"""Make an encoding model that uses 'latent' retina information.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
import numpy as np
import retina.response_model.python.metric_learning.end_to_end.losses as losses
import retina.response_model.python.metric_learning.end_to_end.response_embedding as resp
import retina.response_model.python.metric_learning.end_to_end.stimulus_embedding as stim
slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS


def convolutional_encoder(sess, is_training, dimx, dimy):
  """Find latent representation and use it for predicting responses.

  TODO(bhaishahster): Subsample retinas in training,

  Args:
    sess : Tensorflow session.
    is_training : Either training or evaluation mode.
    dimx: X dimension of the stimulus.
    dimy: Y dimension of the stimulus.

  Returns:
    sr_graph : Container of the embedding parameters and losses.
  """

  # Find latent dimensions
  layers = FLAGS.resp_layers
  latent_dimensionality = np.int(layers.split(',')[-2])
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

  # Get retina parameters from negative responses.
  retina_params = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(neg_model.responses_embed,
                                                               0), 0), 0)

  ## Go from stimulus to response prediction for the population.
  # Get stimulus placeholder.
  stim_tf = tf.placeholder(tf.float32,
                           shape=[None, dimx,
                                  dimy, 30])  # batch x X x Y x time_window

  # Do 1 layer of convolution on stimulus.
  # Set normalization
  if batch_norm:
    normalizer_fn = slim.batch_norm
  else:
    normalizer_fn = None

  if FLAGS.sr_model == 'convolutional_encoder': #or FLAGS.sr_model == 'convolutional_encoder_2'
    pred_fcn = pred_responses_latent

  if FLAGS.sr_model == 'convolutional_encoder_2':
    pred_fcn = pred_responses_latent_2

  response_predicted = pred_fcn(retina_params, stim_tf,
                                latent_dimensionality,
                                normalizer_fn, is_training,
                                reuse_variables=False)

  # Predict responses using arbitrary retina parameters.
  retina_params_arbitrary = tf.placeholder(tf.float32,
                                           shape=[latent_dimensionality],
                                           name='retina_params_arbitrary')

  response_pred_from_arbit_ret_params = pred_fcn(retina_params_arbitrary,
                                                 stim_tf,
                                                 latent_dimensionality,
                                                 normalizer_fn, is_training,
                                                 reuse_variables=True)

  tfd = tf.expand_dims
  loss = (tf.reduce_sum((response_predicted -
                         anchor_model.embed_responses_original *
                         tf.log(response_predicted)) * tfd(anchor_model.embed_locations_original, 0)) / tf.reduce_sum(anchor_model.embed_locations_original))
  loss_regularization = 0.5 * tf.reduce_sum(tf.square(retina_params))

  loss_total = (FLAGS.scale_encode * loss +
          FLAGS.scale_regularization * loss_regularization)

  train_op = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(loss_total)

  # Store everything in a graph.
  sr_graph = collections.namedtuple('SR_Graph', 'sess train_op anchor_model'
                                    ' neg_model'
                                    ' loss'
                                    ' stim_tf fr_predicted retina_params'
                                    ' response_pred_from_arbit_ret_params'
                                    ' retina_params_arbitrary')

  sr_graph = sr_graph(sess, train_op, anchor_model, neg_model, loss,
                      stim_tf, response_predicted, retina_params,
                      response_pred_from_arbit_ret_params, retina_params_arbitrary)

  return sr_graph


def pred_responses_latent(retina_params, stim_tf, latent_dimensionality,
                          normalizer_fn, is_training, reuse_variables=False):
  """Use latent representation and stimulus to predict responses.

  Do outer product on first layer.
  """

  stim_ld = slim.conv2d(stim_tf, latent_dimensionality,
                        [1, 1],
                        stride=1,
                        scope='stim_downsample',
                        normalizer_fn=normalizer_fn,
                        activation_fn=tf.nn.relu,
                        reuse=reuse_variables)

  # Use response embedding to
  stim_ret_outer_prod = tf.expand_dims(stim_ld, 4) * retina_params
  shape = stim_ret_outer_prod.get_shape().as_list()
  stim_ret_merged = tf.reshape(stim_ret_outer_prod,
                               [-1, shape[1],
                                shape[2], latent_dimensionality ** 2])

  response_predicted, _= stim.convolutional_encode(FLAGS.stim_layers.split(','),
                                                   FLAGS.batch_norm,
                                                   stim_ret_merged,
                                                   is_training,
                                                   reuse_variables=reuse_variables)

  return response_predicted


def pred_responses_latent_2(retina_params, stim_tf, latent_dimensionality,
                          normalizer_fn, is_training, reuse_variables=False):
  """Use latent representation and stimulus to predict responses

  Feature weighting using latent dimensionality.
  """

  ##
  layers = FLAGS.stim_layers.split(',')
  n_layers = int(len(layers)/3)
  tf.logging.info('Number of layers: %d' % n_layers)

  ##
  # Take latent vector and get scales, shifts for different layers by a fully connected layer
  n_scales = np.sum([int(layers[ilayer * 3 + 1]) for ilayer in range(n_layers)])
  n_shifts = np.sum([int(layers[ilayer * 3 + 1]) for ilayer in range(n_layers)])
  retina_scales = slim.fully_connected(tf.expand_dims(retina_params, 0),
                                       n_scales,
                                       activation_fn=None,
                                       reuse=reuse_variables,
                                       scope='latent_to_scales')
  retina_scales = retina_scales[0, :]

  retina_shifts = slim.fully_connected(tf.expand_dims(retina_params, 0),
                                       n_scales,
                                       activation_fn=None,
                                       reuse=reuse_variables,
                                       scope='latent_to_shifts')
  retina_shifts = retina_shifts[0, :]

  #if latent_dimensionality != n_scales:
  #  raise ValueError('Latent dimensionality must be same as sum of widths of each layer')


  activation_fn = tf.nn.softplus
  tf.logging.info('Logistic activation')

  # Use slim to define multiple layers of convolution.
  net = stim_tf
  layer_collection = [net]
  latent_idx = 0
  for ilayer in range(n_layers):

    # Convolve
    tf.logging.info('Building stimulus embedding layer: %d, %d, %d'
                    % (int(layers[ilayer*3 + 1]), int(layers[ilayer*3]),
                       int(layers[ilayer*3 + 2])))
    net = slim.conv2d(net, int(layers[ilayer*3 + 1]),
                      int(layers[ilayer*3]),
                      stride=int(layers[ilayer*3 + 2]),
                      scope='stim_layer_wt_%d' % ilayer,
                      reuse=reuse_variables,
                      normalizer_fn=normalizer_fn,
                      activation_fn=None,
                      normalizer_params={'is_training': is_training})

    # Scale using latent dimensionality after each layer.
    #if latent_dimensionality >= latent_idx + 1 + int(layers[ilayer*3 + 1]):
    net = (net * retina_scales[latent_idx:
                              latent_idx + int(layers[ilayer*3 + 1])] +
    retina_shifts[latent_idx: latent_idx + int(layers[ilayer*3 + 1])])


    if ilayer == n_layers - 1:
      net = tf.exp(net)  # Put exponential nonlinearity in last layer.
    else:
      net = activation_fn(net)
    '''
    print('CAREFUL: REMOVED EXPONENT - resp prediction!!!!!!!')
    net = activation_fn(net)
    '''
    latent_idx += int(layers[ilayer*3 + 1])

    layer_collection += [net]

  return net


def embed_ei(ei_tf, normalizer_fn, is_training, reuse_variables=False):

  layers = FLAGS.ei_layers.split(',')
  n_layers = int(len(layers)/3)
  tf.logging.info('Number of layers: %d' % n_layers)


  activation_fn = tf.nn.softplus
  tf.logging.info('Logistic activation')

  # Use slim to define multiple layers of convolution.
  net = ei_tf
  layer_collection = [net]
  latent_idx = 0
  for ilayer in range(n_layers):

    # Convolve
    tf.logging.info('Building stimulus embedding layer: %d, %d, %d'
                    % (int(layers[ilayer*3 + 1]), int(layers[ilayer*3]),
                       int(layers[ilayer*3 + 2])))
    net = slim.conv2d(net, int(layers[ilayer*3 + 1]),
                      int(layers[ilayer*3]),
                      stride=int(layers[ilayer*3 + 2]),
                      scope='ei_layer_wt_%d' % ilayer,
                      reuse=reuse_variables,
                      normalizer_fn=normalizer_fn,
                      activation_fn=None,
                      normalizer_params={'is_training': is_training})

    if ilayer < n_layers - 1:  # No rectification for the last layer.
      net = activation_fn(net)
      #print('REMOVED SOFTPLUS IN FINAL LAYER OF EI EMBEDDING!!')

  # Final EI embedding
  net = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(net, 0), 0), 0)
  return net


def convolutional_encoder_using_retina_id(sess, is_training, dimx, dimy, n_retinas):
  """Find latent representation and use it for predicting responses.

  Use one-hot encoding of retina-id to interpolate responses.
  
  Args:
    sess : Tensorflow session.
    is_training : Either training or evaluation mode.
    dimx: X dimension of the stimulus.
    dimy: Y dimension of the stimulus.
    n_retinas: Number of retinas

  Returns:
    sr_graph : Container of the embedding parameters and losses.

  """

  # Find latent dimensions
  layers = FLAGS.resp_layers
  latent_dimensionality = np.int(layers.split(',')[-2])
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

  # Get retina parameters from negative responses.
  # get 'retina_params' from 1 hot retina-id.
  retina_indicator = tf.placeholder(shape=[n_retinas], dtype=tf.float32)
  a_mat = tf.get_variable('a_mat', shape=[latent_dimensionality,
                                          n_retinas])
  retina_params = tf.matmul(a_mat, tf.expand_dims(retina_indicator, 1))
  retina_params = retina_params[:, 0]

  ## Go from stimulus to response prediction for the population.
  # Get stimulus placeholder.
  stim_tf = tf.placeholder(tf.float32,
                           shape=[None, dimx,
                                  dimy, 30])  # batch x X x Y x time_window

  # Do 1 layer of convolution on stimulus.
  # Set normalization
  if batch_norm:
    normalizer_fn = slim.batch_norm
  else:
    normalizer_fn = None


  pred_fcn = pred_responses_latent_2

  response_predicted = pred_fcn(retina_params, stim_tf,
                                latent_dimensionality,
                                normalizer_fn, is_training,
                                reuse_variables=False)


  # Predict responses using arbitrary retina parameters.
  retina_params_arbitrary = tf.placeholder(tf.float32,
                                           shape=[latent_dimensionality],
                                           name='retina_params_arbitrary')

  response_pred_from_arbit_ret_params = pred_fcn(retina_params_arbitrary,
                                                 stim_tf,
                                                 latent_dimensionality,
                                                 normalizer_fn, is_training,
                                                 reuse_variables=True)

  tfd = tf.expand_dims
  loss = (tf.reduce_sum((response_predicted -
                         anchor_model.embed_responses_original *
                         tf.log(response_predicted)) * tfd(anchor_model.embed_locations_original, 0)) / tf.reduce_sum(anchor_model.embed_locations_original))
  loss_regularization = 0.5 * tf.reduce_sum(tf.square(retina_params))

  loss_total = (FLAGS.scale_encode * loss +
                FLAGS.scale_regularization * loss_regularization)

  loss_arbit_ret_params = (tf.reduce_sum((response_pred_from_arbit_ret_params -
                         anchor_model.embed_responses_original *
                         tf.log(response_pred_from_arbit_ret_params)) * tfd(anchor_model.embed_locations_original, 0)) / tf.reduce_sum(anchor_model.embed_locations_original))

  # Use EIs for blind retina.
  str_ei_embedding = ''
  ei_params = ()
  if FLAGS.use_EIs:
    # Get retina parameters from EIs.
    # get : retina_params_from_ei
    ei_image = tf.placeholder(tf.float32, shape=[None, 64, 32])

     # Set normalization
    if FLAGS.batch_norm_ei:
      normalizer_fn = slim.batch_norm
    else:
      normalizer_fn = None

    retina_params_from_ei = embed_ei(tf.expand_dims(ei_image, 3),
                                     normalizer_fn, is_training,
                                     reuse_variables=False)

    # Predict responses from retina params from EIs.
    if batch_norm:
      normalizer_fn = slim.batch_norm
    else:
      normalizer_fn = None
    response_pred_from_eis = pred_fcn(retina_params_from_ei,
                                      stim_tf,
                                      latent_dimensionality,
                                      normalizer_fn, is_training,
                                      reuse_variables=True)

    loss_from_ei = (tf.reduce_sum((response_pred_from_eis -
                         anchor_model.embed_responses_original *
                         tf.log(response_pred_from_eis)) * tfd(anchor_model.embed_locations_original, 0)) / tf.reduce_sum(anchor_model.embed_locations_original))
    loss_regularization_from_ei = 0.5 * tf.reduce_sum(tf.square(retina_params_from_ei))
    loss_match_embeddings = 0.5 * tf.reduce_sum(tf.square(retina_params - retina_params_from_ei))
    loss_total_ei = (FLAGS.scale_encode_from_ei * loss_from_ei +
                     FLAGS.scale_regularization_from_ei * loss_regularization_from_ei +
                     FLAGS.scale_match_embeddding * loss_match_embeddings)
    loss_total = loss_total + loss_total_ei

    str_ei_embedding = (' ei_image retina_params_from_ei response_pred_from_eis'
                       ' loss_from_ei loss_regularization_from_ei'
                       ' loss_match_embeddings loss_total_ei')
    ei_params = (ei_image, retina_params_from_ei, response_pred_from_eis,
                 loss_from_ei, loss_regularization_from_ei,
                 loss_match_embeddings, loss_total_ei)

  train_op = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(loss_total)

  accuracy_tf = (tf.reduce_sum(anchor_model.embed_responses_original *
                            (response_predicted - tf.reduce_mean(response_predicted, 0)) *
                            tfd(anchor_model.embed_locations_original, 0)) / tf.reduce_sum(anchor_model.embed_locations_original))

  # Store everything in a graph.
  sr_graph = collections.namedtuple('SR_Graph', 'sess train_op anchor_model'
                                    ' loss loss_arbit_ret_params'
                                    ' stim_tf fr_predicted retina_params'
                                    ' response_pred_from_arbit_ret_params'
                                    ' retina_params_arbitrary retina_indicator a_mat accuracy_tf' + str_ei_embedding)

  sr_graph = sr_graph(sess, train_op, anchor_model, loss, loss_arbit_ret_params,
                      stim_tf, response_predicted, retina_params,
                      response_pred_from_arbit_ret_params,
                      retina_params_arbitrary, retina_indicator, a_mat,
                      accuracy_tf, *ei_params)

  return sr_graph
