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
"""Learn a score function modeled as Markov Random Field (MRF).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from retina.response_model.python.metric_learning.score_fcns import metric

class MRFScore(metric.Metric):

  def __init__(self, sess, save_folder, file_name, **kwargs):
    """Initialize MRF score model."""

    tf.logging.info('MRF score')
    self._build_graph(**kwargs)

    self.build_summaries()
    tf.logging.info('Summary operatory made')

    self.sess = sess
    self.initialize_model(save_folder, file_name, sess)
    tf.logging.info('Model initialized')

  def _build_graph(self, n_cells, time_window, lr, lam_l1,
                   cell_centers, neighbor_threshold):

    # placeholders for anchor, pos and neg
    self.anchor = tf.placeholder(dtype=tf.float32,
                                 shape=[None, n_cells, time_window],
                                 name='anchor')
    self.pos = tf.placeholder(dtype=tf.float32,
                              shape=[None, n_cells, time_window],
                              name='pos')
    self.neg = tf.placeholder(dtype=tf.float32,
                              shape=[None, n_cells, time_window],
                              name='neg')
    self.data_dim = n_cells*time_window

    # convert them into feature vector and get output dimensions
    # find nearby cells
    cell_distances = np.sqrt(np.sum((np.expand_dims(cell_centers, 1) -
                             np.expand_dims(cell_centers, 0))**2, 2))
    self.cell_pairs = cell_distances < neighbor_threshold

    # binarize points
    anchor_bin = tf.cast(tf.greater_equal(self.anchor, 1), tf.float32)
    pos_bin = tf.cast(tf.greater_equal(self.pos, 1), tf.float32)
    neg_bin = tf.cast(tf.greater_equal(self.neg, 1), tf.float32)

    # featurize points
    self.phi_anchor, self.feature_dim = self._extract_features(anchor_bin)
    self.phi_pos, _ = self._extract_features(pos_bin)
    self.phi_neg, _ = self._extract_features(neg_bin)

    # declare variables
    # learn weights for confusing a cells response 0/1 -> 0/1
    self.A_x = tf.Variable(np.random.randn(self.feature_dim[0],
                                      2, 2).astype(np.float32),name='A_x')
    # learn weights for confusing a pair of resposnes xx -> xx
    self.A_xx = tf.Variable(np.random.randn(self.feature_dim[2],
                                       3, 3).astype(np.float32),name='A_xx')

    # set train_step, loss, score_anchor_pos
    self.score_anchor_pos = self._get_score(self.phi_anchor, self.phi_pos)
    self.score_anchor_neg = self._get_score(self.phi_anchor, self.phi_neg)

    self.loss = (tf.reduce_sum(tf.nn.relu(self.score_anchor_pos -
                                          self.score_anchor_neg + 1)) +
                 lam_l1*tf.reduce_sum(tf.abs(self.A_x)) +
                 lam_l1*tf.reduce_sum(tf.abs(self.A_xx)))
    self.train_step = tf.train.AdagradOptimizer(lr).minimize(self.loss)

  def _extract_features(self, anchor):
    """Extract features to use in score function.

    Args :
      anchor : input tensor of size (batch x cells x time_window)

    Returns :
      features : input converted into list of features [batch x feature_dim]
      feature_dim : dimension of each output tensor [feature_dim]
    """

    anchor_flat = tf.reshape(anchor, [-1, self.data_dim])
    feature_dim = []
    features0 = 1 - anchor_flat #tf.cast(tf.equal(anchor_flat, 0), tf.float32)
    features1 = anchor_flat #tf.cast(tf.greater_equal(anchor_flat, 1), tf.float32)
    # better than == 1 if bigger bin size.
    feature_dim += [self.data_dim]*2

    # select cell pairs
    select_cell_pairs = np.reshape(self.cell_pairs, [self.data_dim**2])
    pairs_locations = np.where(select_cell_pairs)[0]
    n_pairs = pairs_locations.shape[0]

    # Compute features of type (cell 1 = x)^(cell 2 = y) , for x, y = {0,1}.
    features00_all = tf.reshape(tf.expand_dims(features0, 2) *
                            tf.expand_dims(features0, 1),
                            [-1, self.data_dim**2])
    features00 = tf.transpose(tf.gather(tf.transpose(features00_all),
                                        pairs_locations))

    features01_all = tf.reshape(tf.expand_dims(features0, 2) *
                            tf.expand_dims(features1, 1),
                            [-1, self.data_dim**2])
    features01 = tf.transpose(tf.gather(tf.transpose(features01_all),
                                        pairs_locations))

    features11_all = tf.reshape(tf.expand_dims(features1, 2) *
                            tf.expand_dims(features1, 1),
                            [-1, self.data_dim**2])
    features11 = tf.transpose(tf.gather(tf.transpose(features11_all),
                                        pairs_locations))
    feature_dim += 3 * [n_pairs]

    # Now append all features.
    features_list = [features0, features1, features00, features01, features11]

    return features_list, feature_dim

  def _get_score(self, phi_anchor, phi_pos):
    """Returns the mismatch score of two featurized tensors."""

    anchor_2 = tf.expand_dims(tf.concat(2, [tf.expand_dims(phi_anchor[0], 2),
                                            tf.expand_dims(phi_anchor[1], 2)]),
                              2)
    pos_2 = tf.expand_dims(tf.concat(2, [tf.expand_dims(phi_pos[0], 2),
                                         tf.expand_dims(phi_pos[1], 2)]), 3)
    score_2 = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.mul(self.A_x,
                                                               anchor_2 * pos_2),
                                                        3), 2), 1)
    anchor_3 = tf.expand_dims(tf.concat(2, [tf.expand_dims(phi_anchor[2], 2),
                                            tf.expand_dims(phi_anchor[3], 2),
                                            tf.expand_dims(phi_anchor[4], 2)]), 2)

    pos_3 = tf.expand_dims(tf.concat(2, [tf.expand_dims(phi_pos[2], 2),
                                         tf.expand_dims(phi_pos[3], 2),
                                         tf.expand_dims(phi_pos[4], 2)]), 3)
    score_3 = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.mul(self.A_xx,
                                                               anchor_3 * pos_3),
                                                        3), 2), 1)


    return score_2 + score_3

  def get_parameters(self):
    """Return insightful parameters of the score function."""
    return [self.sess.run(self.A_x), self.sess.run(self.A_xx)]

  def update(self, triplet_batch):
    """Given a batch of training data, update metric parameters.

    Args :
        triplet_batch : List [anchor, positive, negative], each with shape:
                        (batch x cells x time_window)
    Returns :
        loss : Training loss for the batch of data.
    """
    feed_dict = {self.anchor: triplet_batch[0],
                 self.pos: triplet_batch[1],
                 self.neg: triplet_batch[2]}
    _, loss = self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)
    return loss

  def get_distance(self, anchor_in, pos_in):
    """Return mismatch score (similar to distance) for two inputs."""
    feed_dict = {self.anchor: anchor_in, self.pos: pos_in}
    distances = self.sess.run(self.score_anchor_pos, feed_dict=feed_dict)
    return distances

  def get_embedding(self, x):
    """Get embedding of data into the space."""
    pass

  def learn_encoding_model_glm(self, stimulus, response, ttf_in=None, lr=0.001, lam_l1_rf=0.001):
    """Learn GLM encoding model using the metric."""

    from IPython import embed; embed()

    # get paramters
    data_len = stimulus.shape[0]
    n_cells = response.shape[2]
    dimx = stimulus.shape[1]
    dimy = stimulus.shape[2]

    # generate responses using current parameters.
    # stimulus - response constants
    stim_tf = tf.placeholder(dtype=tf.float32, shape=[None, dimx, dimy])
    #tf.constant(stimulus[0:1000,:,:].astype(np.float32)) # ? x dimx x dimy
    resp_tf = tf.placeholder(dtype=tf.float32, shape=[None, n_cells])
    #tf.constant(response[0,0:1000,:].astype(np.float32), name='resp')

    # Compute variables.
    tlen = 30
    if ttf_in is None:
      ttf = tf.Variable(0.1 + 0*np.random.randn(tlen).astype(np.float32),
                        name='ttf')
    else:
      ttf = tf.Variable(ttf_in.astype(np.float32), name='ttf')

    # Time filter.
    stim_inp  = tf.expand_dims(tf.transpose(stim_tf, [1, 0, 2]), 3)
    ttf_filt = tf.expand_dims(tf.expand_dims(tf.expand_dims(ttf, 1), 2), 3)

    stim_time_filtered = tf.nn.conv2d(stim_inp, ttf_filt, strides=[1, 1, 1, 1],
                                      padding="VALID")
    stim_time_filt_reshape = tf.transpose(stim_time_filtered,
                                          [1, 0, 2, 3])

    # Initialize remaining variables
    uninitialized_vars = []
    for var in tf.all_variables():
      try:
        self.sess.run(var)
      except tf.errors.FailedPreconditionError:
        uninitialized_vars.append(var)
    init_new_vars_op = tf.variables_initializer(uninitialized_vars)
    self.sess.run(init_new_vars_op)

    # compute STAs
    if ttf_in is None:
      stas_np = None
    else:
      stas = tf.reshape(tf.matmul(tf.transpose(tf.reshape(stim_time_filt_reshape, [-1, dimx*dimy])), resp_tf ), [dimx, dimy, n_cells])
      batch_sz = data_len - tlen
      end_time = data_len
      feed_dict = {stim_tf: stimulus[end_time-batch_sz-(tlen-1): end_time,:,:].astype(np.float32),
                   resp_tf: response[0, end_time-batch_sz: end_time, :].astype(np.float32)}
      stas_np = self.sess.run(stas, feed_dict=feed_dict)

    # Space filter.
    if stas_np is None:
      RF_all = tf.Variable(0.1 + 0*np.random.randn(dimx, dimy,
                                                 n_cells).astype(np.float32),
                           name='RFs')
    else :
      RF_all = tf.Variable(stas_np.astype(np.float32),
                           name='RFs')

    stim_space_filtered = tf.reduce_sum(tf.reduce_sum(stim_time_filt_reshape * RF_all, 2),
                                        1) # ? x n_cells

    generator_signal = stim_space_filtered
    firing_rate = tf.nn.relu(generator_signal)

    # update parameters.
    
    phi_firing_rate, _ = self._extract_features(tf.expand_dims(firing_rate, 2))
    phi_response, _ = self._extract_features(tf.expand_dims(resp_tf, 2))
    distances = self._get_score(phi_firing_rate, phi_response)

    # distances = tf.reduce_sum(tf.pow(firing_rate - resp_tf, 2), 1)

    loss_encoding = (tf.reduce_sum(distances) +
                     lam_l1_rf*tf.reduce_sum(tf.abs(RF_all)))

    train_step_RF = tf.train.AdamOptimizer(lr).minimize(loss_encoding,
                                                        var_list=[RF_all])

    train_step_ttf = tf.train.AdamOptimizer(lr).minimize(loss_encoding,
                                                         var_list=[ttf])
    # Initialize remaining variables
    uninitialized_vars = []
    for var in tf.all_variables():
      try:
        self.sess.run(var)
      except tf.errors.FailedPreconditionError:
        uninitialized_vars.append(var)
    init_new_vars_op = tf.variables_initializer(uninitialized_vars)
    self.sess.run(init_new_vars_op)


    ## Learning
    # test data
    batch_sz = 1000
    end_time = np.random.randint(batch_sz, data_len)
    feed_dict_test = {stim_tf: stimulus[end_time-batch_sz-(tlen-1): end_time,:,:].astype(np.float32),
                 resp_tf: response[0, end_time-batch_sz: end_time, :].astype(np.float32)}

    # Scale RF and TTF
    # Scale time course to match firing rate of all cells.
    scale_ttf = tf.reduce_mean(resp_tf) / tf.reduce_mean(firing_rate)
    update_ttf = tf.assign(ttf, ttf*scale_ttf)
    batch_sz = 1000
    end_time = np.random.randint(batch_sz, data_len)
    feed_dict = {stim_tf: stimulus[end_time-batch_sz-(tlen-1): end_time,:,:].astype(np.float32),
                 resp_tf: response[0, end_time-batch_sz: end_time, :].astype(np.float32)}
    self.sess.run(update_ttf, feed_dict=feed_dict)
    print('Time course scale updated')


    # Scale RF to match firing rate of individual cells.
    # skipping for now

    # Learn spatial RF
    for iiter in range(100):

      end_time = np.random.randint(batch_sz, data_len)
      feed_dict = {stim_tf: stimulus[end_time-batch_sz-(tlen-1): end_time,:,:].astype(np.float32),
                   resp_tf: response[0, end_time-batch_sz: end_time, :].astype(np.float32)}

      _, loss_encoding_np = self.sess.run([train_step_RF, loss_encoding],
                                          feed_dict=feed_dict)
      # if iiter % 100 == 0:
      loss_encoding_np_test = self.sess.run(loss_encoding, feed_dict=feed_dict)
      print(iiter, end_time, loss_encoding_np, loss_encoding_np_test)

    # Learn temporal part
    for iiter in range(100):

      end_time = np.random.randint(batch_sz, data_len)
      feed_dict = {stim_tf: stimulus[end_time-batch_sz-(tlen-1): end_time,:,:].astype(np.float32),
                   resp_tf: response[0, end_time-batch_sz: end_time, :].astype(np.float32)}

      _, loss_encoding_np = self.sess.run([train_step_ttf, loss_encoding],
                                          feed_dict=feed_dict)
      # if iiter % 100 == 0:
      loss_encoding_np_test = self.sess.run(loss_encoding, feed_dict=feed_dict)

      print(iiter, end_time, loss_encoding_np, loss_encoding_np_test)


      # do some response prediction
      batch_sz = 1000
      end_time = np.random.randint(batch_sz, data_len)
      feed_dict_fr = {stim_tf: stimulus[end_time-batch_sz-(tlen-1): end_time,:,:].astype(np.float32),
                   resp_tf: response[0, end_time-batch_sz: end_time, :].astype(np.float32)}
      fr_np = self.sess.run(firing_rate, feed_dict=feed_dict_fr)

      spks_sample = np.sum(np.random.rand(batch_sz, n_cells) < fr_np)
      spks_rec = np.sum(response[0, end_time-batch_sz: end_time, :].astype(np.float32))
      print('True spks %d, sample spks %d' % (spks_rec, spks_sample))
      plt.plot(fr_np[:,23]);
      plt.show()

  def build_summaries(self):
    """Add some summaries."""

    # Loss summary.
    tf.summary.scalar('loss', self.loss)

    merged = tf.summary.merge_all()
    self.summary_op = merged
    tf.logging.info('summary OP set')
