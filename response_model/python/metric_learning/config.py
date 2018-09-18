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
"""Set global parameters for metric learning and/or evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import tensorflow as tf
import retina.response_model.python.metric_learning.score_fcns.convolutional as conv
import retina.response_model.python.metric_learning.score_fcns.convolutional_prosthesis as conv_prosthesis
# TODO(bhaishahster): add score functions which have todo after them
import retina.response_model.python.metric_learning.score_fcns.hamming_distance as hamming  # TODO(bhaishahster)
import retina.response_model.python.metric_learning.score_fcns.low_dim_score as low_dim_score  # TODO(bhaishahster)
# MLNN models
import retina.response_model.python.metric_learning.score_fcns.mlnn_logistic as mlnn_logistic  # TODO(bhaishahster)
import retina.response_model.python.metric_learning.score_fcns.mlnn_logistic_all as mlnn_logistic_all  # TODO(bhaishahster)
import retina.response_model.python.metric_learning.score_fcns.mlnn_logistic_neg_batch as mlnn_logistic_neg_batch  # TODO(bhaishahster)
import retina.response_model.python.metric_learning.score_fcns.mlnn_score as mlnn_score  # TODO(bhaishahster)
import retina.response_model.python.metric_learning.score_fcns.mlnn_slim as mlnn_slim  # TODO(bhaishahster)
import retina.response_model.python.metric_learning.score_fcns.mlnn_symmetrized_score as mlnn_symmetrized_score  # TODO(bhaishahster)
# MRF
import retina.response_model.python.metric_learning.score_fcns.mrf as mrf  # TODO(bhaishahster)
# Quadratic
import retina.response_model.python.metric_learning.score_fcns.quadratic_score as quad  # TODO(bhaishahster)
import retina.response_model.python.metric_learning.score_fcns.quadratic_score_constrained as quad_constrained  # TODO(bhaishahster)
import retina.response_model.python.metric_learning.score_fcns.quadratic_score_precomputed as quad_precompute  # TODO(bhaishahster)
import retina.response_model.python.metric_learning.score_fcns.quadratic_score_psd as quad_psd  # TODO(bhaishahster)
import retina.response_model.python.metric_learning.score_fcns.quadratic_score_psd_diag as quad_psd_diag  # TODO(bhaishahster)
import retina.response_model.python.metric_learning.score_fcns.quadratic_score_psd_neg_batch as quad_psd_neg_batch  # TODO(bhaishahster)
import retina.response_model.python.metric_learning.score_fcns.quadratic_score_psd_scaledI as quad_psd_scaledI  # TODO(bhaishahster)
flags = tf.app.flags

flags.DEFINE_string('data_path',
                    '/home/bhaishahster/metric_learn',
                    'location of example data')

flags.DEFINE_string('save_suffix', '', 'Identifying suffix in folder name')

flags.DEFINE_float('learning_rate',
                   0.1,
                   'learning_rate for training')

flags.DEFINE_integer('time_window',
                     1,
                     'Number of bins of response')

flags.DEFINE_string('data_train', 'example_long_wn_2rep_ON_OFF_ALL_lr.mat',
                    'Test data')

flags.DEFINE_string('triplet_type', 'a',
                    'The definition of triplets to train with (a/b).'
                    'Please refer to data_util for details.')

flags.DEFINE_string('data_test',
                    'example_wn_30reps_ON_OFF_ALL_lr_with_stimulus.mat',
                    'Test data')

flags.DEFINE_integer('batch_size_test',
                     10000,
                     'Batch size for eval')

flags.DEFINE_integer('batch_size_train',
                     10000,
                     'Batch size for training')

flags.DEFINE_integer('max_iter', 20000,
                     'maximum number of iterations for training')

## Model specific parameters

flags.DEFINE_string('model', 'quadratic', 'type score model to learn')

flags.DEFINE_string('save_folder',
                    '/home/bhaishahster/metric_learn_dec_15/',
                    'where to store model and analysis')

# Quadratic model
flags.DEFINE_float('lam_l1',
                   0.1,
                   'L1 regularization on quadratic model parameters')

# Quadratic model, precomputed A
flags.DEFINE_string('precomputed_a_loc', '',
                    'Location of A for precomputed quadratic metric')

# RNN
flags.DEFINE_integer('hidden_layer_size',
                     10,
                     'dimensionality of each unit in RNN')

flags.DEFINE_integer('num_rnn_layers',
                     2,
                     'number of  layers in RNN')

# MRF model and quadratic constrained
flags.DEFINE_float('neighbor_threshold', 6,
                   'interaction terms for cells with distance'
                   ' below this threshold')

# MLNN model and it's variants
flags.DEFINE_string('layers', '5, 5', 'sizes of layers for a multilayered NN'
                    ' (comma separated)')

# parameter for MLNN with batch negative
flags.DEFINE_float('beta', 1, 'temperature parameter for log-sum-exp')

# Low dimensional distance matrix model
flags.DEFINE_integer('score_mat_dim', 10, 'rank of score matrix, '
                     'where score matrix is dxd and d = 2^(cells x time steps)')

# Convolutional / Convolutional_prostheis metric for the whole population
flags.DEFINE_float('grid_resolution', 0.5,
                   'resolution of grid on which cells will be located.')

flags.DEFINE_string('convolutional_layers', '10, 7, 6',
                    'Model description with format '
                    '(window size , number of filters, stride) for each layer')

flags.DEFINE_float('lam',
                   0.1,
                   'L2 regularization of model parameters')

flags.DEFINE_bool('batch_norm',
                  False,
                  'Do we apply batch-norm regularization between layers?')


def get_filepaths():
  """Return save folder and save filename."""

  def _folder_suffix(name):
    return {
        'quadratic': (flags.FLAGS.model + '_lam_l1_%.3f' % flags.FLAGS.lam_l1),
        'quadratic_psd': (flags.FLAGS.model +
                          '_lam_l1_%.3f' % flags.FLAGS.lam_l1),
        'quadratic_psd_neg_batch': (flags.FLAGS.model +
                                    '_lam_l1_%.3f_beta_%.5f' %
                                    (flags.FLAGS.lam_l1, flags.FLAGS.beta)),
        'quadratic_psd_scaledI': (flags.FLAGS.model +
                                  '_lam_l1_%.3f' % flags.FLAGS.lam_l1),
        'quadratic_psd_diag': (flags.FLAGS.model +
                               '_lam_l1_%.3f' % flags.FLAGS.lam_l1),
        'quadratic_constrained': (flags.FLAGS.model +
                                  '_lam_l1_%.3f_neighborhood_%.3f' %
                                  (flags.FLAGS.lam_l1,
                                   flags.FLAGS.neighbor_threshold)),
        'quadratic_precomputed': (flags.FLAGS.model +
                                  '_a_%s' %
                                  os.path.basename(flags.FLAGS.
                                                   precomputed_a_loc)),
        'RNN': (flags.FLAGS.model + 'hidden_size_%.3f_num_layers_%.3f' %
                (flags.FLAGS.hidden_layer_size,
                 flags.FLAGS.num_rnn_layers)),
        'mlnn': (flags.FLAGS.model + '_layers_%s' % flags.FLAGS.layers),
        'mlnn_symm': (flags.FLAGS.model + '_layers_%s' % flags.FLAGS.layers),
        'mlnn_slim': (flags.FLAGS.model + '_layers_%s' % flags.FLAGS.layers),
        'mlnn_logistic': (flags.FLAGS.model +
                          '_layers_%s' % flags.FLAGS.layers),
        'mlnn_logistic_all': (flags.FLAGS.model +
                              '_layers_%s' % flags.FLAGS.layers),
        'mlnn_logistic_neg_batch': (flags.FLAGS.model + '_layers_%s_beta_%.5f'
                                    % (flags.FLAGS.layers, flags.FLAGS.beta)),
        'mrf': (flags.FLAGS.model + '_lam_l1_%.3f_neighborhood_%.3f' %
                (flags.FLAGS.lam_l1, flags.FLAGS.neighbor_threshold)),
        'hamming': ('hamming'),
        'low_dim_score': (flags.FLAGS.model + '_score_mat_dim_%d' %
                          flags.FLAGS.score_mat_dim),
        'conv': (flags.FLAGS.model + '_grid_res_%.3f_layers_%s'
                 '_lam_%.3f_batch_norm_%d:' %
                 (flags.FLAGS.grid_resolution,
                  flags.FLAGS.convolutional_layers, flags.FLAGS.lam,
                  flags.FLAGS.batch_norm)),
        'conv_prosthesis': (flags.FLAGS.model + '_grid_res_%.3f_layers_%s'
                            '_lam_%.3f_batch_norm_%d:' %
                            (flags.FLAGS.grid_resolution,
                             flags.FLAGS.convolutional_layers,
                             flags.FLAGS.lam,
                             flags.FLAGS.batch_norm))
    }.get(name, -1)  # -1 is default if name not found

  folder_suffix = _folder_suffix(flags.FLAGS.model)

  # add global suffix.
  if flags.FLAGS.triplet_type != 'a':
    folder_suffix += '_triplet_type_%s' % flags.FLAGS.triplet_type

  if flags.FLAGS.triplet_type == 'batch':
    folder_suffix += '_beta_%.3f' % flags.FLAGS.beta

  folder_suffix += flags.FLAGS.save_suffix

  model_savepath = os.path.join(flags.FLAGS.save_folder, folder_suffix)
  model_filename = flags.FLAGS.model

  return model_savepath, model_filename


def get_model(sess, model_savepath, model_filename, data_wn, is_training):
  """Get the score object."""

  FLAGS = tf.app.flags.FLAGS

  if FLAGS.model == 'quadratic':
    score = quad.QuadraticScore(sess, model_savepath,
                                model_filename,
                                n_cells=data_wn.n_cells,
                                time_window=FLAGS.time_window,
                                lr=FLAGS.learning_rate,
                                lam_l1=FLAGS.lam_l1)

  if FLAGS.model == 'quadratic_psd':
    score = quad_psd.QuadraticScorePSD(sess, model_savepath,
                                       model_filename,
                                       n_cells=data_wn.n_cells,
                                       time_window=FLAGS.time_window,
                                       lr=FLAGS.learning_rate,
                                       lam_l1=FLAGS.lam_l1)

  if FLAGS.model == 'quadratic_psd_neg_batch':
    model = quad_psd_neg_batch.QuadraticScorePSDNegBatch
    score = model(sess, model_savepath, model_filename, n_cells=data_wn.n_cells,
                  time_window=FLAGS.time_window, lr=FLAGS.learning_rate,
                  lam_l1=FLAGS.lam_l1, beta=FLAGS.beta)

  if FLAGS.model == 'quadratic_psd_scaledI':
    score = quad_psd_scaledI.QuadraticScorePSDscaledI(sess, model_savepath,
                                                      model_filename,
                                                      n_cells=data_wn.n_cells,
                                                      time_window=
                                                      FLAGS.time_window,
                                                      lr=
                                                      FLAGS.learning_rate,
                                                      lam_l1=FLAGS.lam_l1)

  if FLAGS.model == 'quadratic_psd_diag':
    score = quad_psd_diag.QuadraticScorePSDDiagonal(sess, model_savepath,
                                                    model_filename,
                                                    n_cells=data_wn.n_cells,
                                                    time_window=
                                                    FLAGS.time_window,
                                                    lr=FLAGS.learning_rate,
                                                    lam_l1=FLAGS.lam_l1)

  if FLAGS.model == 'mrf':
    score = mrf.MRFScore(sess, model_savepath, model_filename,
                         n_cells=data_wn.n_cells,
                         time_window=FLAGS.time_window,
                         lr=FLAGS.learning_rate,
                         lam_l1=FLAGS.lam_l1,
                         cell_centers=data_wn.get_centers(),
                         neighbor_threshold=FLAGS.neighbor_threshold)

  if FLAGS.model == 'quadratic_constrained':
    score_class = quad_constrained.QuadraticScoreConstrained
    score = score_class(sess, model_savepath, model_filename,
                        n_cells=data_wn.n_cells,
                        time_window=FLAGS.time_window, lr=FLAGS.learning_rate,
                        lam_l1=FLAGS.lam_l1,
                        cell_centers=data_wn.get_centers(),
                        neighbor_threshold=FLAGS.neighbor_threshold)

  if FLAGS.model == 'quadratic_precomputed':
    score_class = quad_precompute.QuadraticScorePrecomputed
    score = score_class(sess, model_savepath, model_filename,
                        n_cells=data_wn.n_cells,
                        time_window=FLAGS.time_window,
                        precomputed_a_loc=FLAGS.precomputed_a_loc)

  if FLAGS.model == 'mlnn':
    score_class = mlnn_score.MetricMLNN
    score = score_class(sess, model_savepath, model_filename,
                        n_cells=data_wn.n_cells,
                        layers=FLAGS.layers,
                        time_window=FLAGS.time_window, lr=FLAGS.learning_rate,
                        is_training=is_training)

  if FLAGS.model == 'mlnn_slim':
    score_class = mlnn_slim.MetricMLNNSlim
    score = score_class(sess, model_savepath, model_filename,
                        n_cells=data_wn.n_cells,
                        layers=FLAGS.layers,
                        time_window=FLAGS.time_window, lr=FLAGS.learning_rate,
                        is_training=is_training)

  if FLAGS.model == 'mlnn_logistic':
    score_class = mlnn_logistic.MetricMLNNLogistic
    score = score_class(sess, model_savepath, model_filename,
                        n_cells=data_wn.n_cells,
                        layers=FLAGS.layers,
                        time_window=FLAGS.time_window, lr=FLAGS.learning_rate,
                        is_training=is_training)

  if FLAGS.model == 'mlnn_logistic_all':
    score_class = mlnn_logistic_all.MetricMLNNLogisticAll
    score = score_class(sess, model_savepath, model_filename,
                        n_cells=data_wn.n_cells,
                        layers=FLAGS.layers,
                        time_window=FLAGS.time_window, lr=FLAGS.learning_rate,
                        is_training=is_training)

  if FLAGS.model == 'mlnn_logistic_neg_batch':
    score_class = mlnn_logistic_neg_batch.MetricMLNNLogisticAllNegBatch
    score = score_class(sess, model_savepath, model_filename, beta=FLAGS.beta,
                        n_cells=data_wn.n_cells,
                        layers=FLAGS.layers,
                        time_window=FLAGS.time_window, lr=FLAGS.learning_rate,
                        is_training=is_training)


  if FLAGS.model == 'mlnn_symm':
    score_class = mlnn_symmetrized_score.MetricMLNNSymm
    score = score_class(sess, model_savepath, model_filename,
                        n_cells=data_wn.n_cells,
                        layers=FLAGS.layers,
                        time_window=FLAGS.time_window, lr=FLAGS.learning_rate,
                        is_training=is_training)

  if FLAGS.model == 'hamming':
    score = hamming.HammingScore(model_savepath)

  if FLAGS.model == 'low_dim_score':
    score = low_dim_score.LowDimScore(sess, model_savepath, model_filename,
                                      n_cells=data_wn.n_cells,
                                      time_window=FLAGS.time_window,
                                      lr=FLAGS.learning_rate,
                                      dim_lr=FLAGS.score_mat_dim)

  if FLAGS.model == 'conv':
    score = conv.ConvolutionalScore(sess, model_savepath, model_filename,
                                    n_cells=data_wn.n_cells,
                                    time_window=FLAGS.time_window,
                                    lr=FLAGS.learning_rate,
                                    centers=data_wn.get_centers(),
                                    resolution=FLAGS.grid_resolution,
                                    layers=FLAGS.convolutional_layers,
                                    lam=FLAGS.lam,
                                    batch_norm=FLAGS.batch_norm,
                                    is_training=is_training,
                                    cell_type=data_wn.get_cell_type(),
                                    triplet_type=FLAGS.triplet_type,
                                    beta=FLAGS.beta)

  if FLAGS.model == 'conv_prosthesis':
    model = conv_prosthesis.ConvolutionalProsthesisScore
    cell_statistics = [data_wn.get_mean_response()]
    # TODO(bhaishahster) : In future, firing rate given as a part of the dataset.
    score = model(sess, model_savepath, model_filename,
                  n_cells=data_wn.n_cells, time_window=FLAGS.time_window,
                  lr=FLAGS.learning_rate, centers=data_wn.get_centers(),
                  resolution=FLAGS.grid_resolution,
                  layers=FLAGS.convolutional_layers, lam=FLAGS.lam,
                  batch_norm=FLAGS.batch_norm, is_training=is_training,
                  cell_type=data_wn.get_cell_type(),
                  cell_statistics=cell_statistics,
                  triplet_type=FLAGS.triplet_type,
                  beta=FLAGS.beta)

  return score


def get_triplet_fcn(data_wn, batch_size):
  """Get functions for training and testing triplets."""

  FLAGS = tf.app.flags.FLAGS

  tf.logging.info('Triplet type : %s' % FLAGS.triplet_type)

  if FLAGS.triplet_type == 'a' or FLAGS.triplet_type == 'mix':
    # for 'mix' type triplets, we still use triplet type 'a'.
    triplet_fcn = lambda: data_wn.get_triplets(batch_size=batch_size,
                                               time_window=FLAGS.time_window)

  if FLAGS.triplet_type == 'b':
    triplet_fcn = lambda: data_wn.get_tripletsB(batch_size=batch_size,
                                                time_window=FLAGS.time_window)

  if FLAGS.triplet_type == 'c':
    triplet_fcn = lambda: data_wn.get_tripletsC(batch_size=batch_size,
                                                time_window=FLAGS.time_window)

  if FLAGS.triplet_type == 'd':
    triplet_fcn = lambda: data_wn.get_tripletsD(batch_size=batch_size,
                                                time_window=FLAGS.time_window)

  # get positive pairs and a batch of negatives
  if FLAGS.triplet_type == 'batch':
    triplet_fcn = lambda: data_wn.get_triplets_batch(batch_size=batch_size,
                                             time_window=FLAGS.time_window)
  return triplet_fcn

