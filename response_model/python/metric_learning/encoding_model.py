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
"""Learn model of stimulus encoding using a response metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
import tensorflow as tf
import collections

FLAGS = flags.FLAGS


# TODO(bhaishahster) : Learn encoding model, given stimulus, response and metric

'''
class EncodingModelLNP():

  def __init__(self, n_cells, dimx, dimy, ):
    pass

  def learn_encoding_model(self, met, stimulus, response, ttf_in=None,
                            initialize_RF_using_ttf=True,
                            scale_ttf=True, lr=0.1,
                            lam_l1_rf=0):
    pass

  def predict_spikes(self, stimulus_in):
    pass


'''

def learn_encoding_model_ln(sess, met, stimulus, response, ttf_in=None,
                            initialize_RF_using_ttf=True,
                            scale_ttf=True, lr=0.1,
                            lam_l1_rf=0):

  """Learn GLM encoding model using the metric.

  Uses ttf to initialize the RF only if ttf_in is given and
  initialize_RF_using_ttf=True.

  If scale_ttf is True, it scales time course to match firing rate
  in observed data.
  """

  # get paramters
  data_len = stimulus.shape[0]
  n_cells = response.shape[2]
  dimx = stimulus.shape[1]
  dimy = stimulus.shape[2]

  # generate responses using current parameters.
  # stimulus - response placeholders
  stim_tf = tf.placeholder(dtype=tf.float32, shape=[None, dimx, dimy])
  resp_tf = tf.placeholder(dtype=tf.float32, shape=[None, n_cells])

  # Compute variables.
  tlen = 30
  if ttf_in is None:
    ttf = tf.Variable(0.1 + 0*np.random.randn(tlen).astype(np.float32),
                      name='ttf')
  else:
    ttf = tf.Variable(ttf_in.astype(np.float32), name='ttf')

  # Time filter.
  stim_inp = tf.expand_dims(tf.transpose(stim_tf, [1, 0, 2]), 3)
  ttf_filt = tf.expand_dims(tf.expand_dims(tf.expand_dims(ttf, 1), 2), 3)

  stim_time_filtered = tf.nn.conv2d(stim_inp, ttf_filt, strides=[1, 1, 1, 1],
                                    padding='VALID')
  stim_time_filt_reshape = tf.transpose(stim_time_filtered,
                                        [1, 0, 2, 3])

  # Initialize remaining variables
  uninitialized_vars = []
  for var in tf.all_variables():
    try:
      sess.run(var)
    except tf.errors.FailedPreconditionError:
      uninitialized_vars.append(var)
  init_new_vars_op = tf.variables_initializer(uninitialized_vars)
  sess.run(init_new_vars_op)

  # compute STAs
  if ttf_in is None or not initialize_RF_using_ttf :
    stas_np = None
    print('RF will be randomly initialized')
  else:
    stas = tf.reshape(tf.matmul(tf.transpose(tf.reshape(stim_time_filt_reshape
                                                        , [-1, dimx*dimy])),
                                resp_tf), [dimx, dimy, n_cells])
    print('RF will be initialized to STAs computed using given ttf')

    batch_sz = data_len - tlen
    end_time = data_len
    feed_dict = {stim_tf: stimulus[end_time-batch_sz-(tlen-1):
                                   end_time, : , : ].astype(np.float32),
                 resp_tf: response[0, end_time-batch_sz:
                                   end_time, : ].astype(np.float32)}
    stas_np = sess.run(stas, feed_dict=feed_dict)

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

  distances = met.get_expected_score(firing_rate, resp_tf)

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
      sess.run(var)
    except tf.errors.FailedPreconditionError:
      uninitialized_vars.append(var)
  init_new_vars_op = tf.variables_initializer(uninitialized_vars)
  sess.run(init_new_vars_op)


  ## Learning
  # test data
  batch_sz = 1000
  end_time = np.random.randint(batch_sz, data_len)
  feed_dict_test = {stim_tf: stimulus[end_time-batch_sz-(tlen-1): end_time,:,:].astype(np.float32),
                    resp_tf: response[0, end_time-batch_sz: end_time, :].astype(np.float32)}


  # Scale RF and TTF

  # Scale time course to match firing rate of all cells.
  if scale_ttf:
    scale_ttf = tf.reduce_mean(resp_tf) / tf.reduce_mean(firing_rate)
    update_ttf = tf.assign(ttf, ttf*scale_ttf)
    batch_sz = 1000
    end_time = np.random.randint(batch_sz, data_len)
    feed_dict = {stim_tf: stimulus[end_time-batch_sz-(tlen-1): end_time,:,:].astype(np.float32),
                 resp_tf: response[0, end_time-batch_sz: end_time, :].astype(np.float32)}
    sess.run(update_ttf, feed_dict=feed_dict)
    print('Time course scaled to match firing rate')

  # TODO(bhaishahster): Scale RF to match firing rate of individual cells.


  for outer_iter in range(10):
    # Plot test loss
    loss_encoding_np_test = sess.run(loss_encoding, feed_dict=feed_dict_test)
    print(outer_iter, loss_encoding_np_test)

    # Learn spatial RF
    for iiter in range(1000):

      end_time = np.random.randint(batch_sz+1000, data_len)
      feed_dict = {stim_tf: stimulus[end_time-batch_sz-(tlen-1): end_time,:,:].astype(np.float32),
                   resp_tf: response[0, end_time-batch_sz: end_time, :].astype(np.float32)}

      _, loss_encoding_np = sess.run([train_step_RF, loss_encoding],
                                          feed_dict=feed_dict)
      '''
      if iiter % 100 == 0:
        loss_encoding_np_test = sess.run(loss_encoding, feed_dict=feed_dict_test)
        print(iiter, end_time, loss_encoding_np, loss_encoding_np_test)
      '''

    # Learn temporal part
    for iiter in range(1000):

      end_time = np.random.randint(batch_sz+1000, data_len)
      feed_dict = {stim_tf: stimulus[end_time-batch_sz-(tlen-1): end_time,:,:].astype(np.float32),
                   resp_tf: response[0, end_time-batch_sz: end_time, :].astype(np.float32)}

      _, loss_encoding_np = sess.run([train_step_ttf, loss_encoding],
                                     feed_dict=feed_dict)

  # Collect return parameters.
  RF_np = sess.run(RF_all)
  ttf_np = sess.run(ttf)
  encoding_model = collections.namedtuple('encoding_model',
                                          ['stimulus', 'firing_rate'])
  model = encoding_model(stim_tf, firing_rate)

  return RF_np, ttf_np, model

  '''
  # do some response prediction
  batch_sz = 1000
  end_time = np.random.randint(batch_sz, data_len)
  feed_dict_fr = {stim_tf: stimulus[end_time-batch_sz-(tlen-1): end_time,:,:].astype(np.float32),
                 resp_tf: response[0, end_time-batch_sz: end_time, :].astype(np.float32)}
  fr_np = self.sess.run(firing_rate, feed_dict=feed_dict_fr)

  spks_sample = np.sum(np.random.rand(batch_sz, n_cells) < fr_np)
  spks_rec = np.sum(response[0, end_time-batch_sz: end_time, :].astype(np.float32))
  print('True spks %d, sample spks %d' % (spks_rec, spks_sample))
  plt.plot(fr_np[:, 0]);
  plt.show()

  # plot RF
  RF_np = self.sess.run(RF_all)
  plt.figure()
  for icell in range(n_cells):
    plt.subplot(np.ceil(np.sqrt(n_cells)), np.ceil(np.sqrt(n_cells)), icell+1)
    plt.imshow(RF_np[:, :, icell], interpolation='nearest', cmap='gray')

  plt.show()

  # plot ttf
  ttf_np = self.sess.run(ttf)
  plt.plot(ttf_np)
  plt.hold(True)
  plt.plot(ttf_in)
  plt.legend(['Fit', 'Initialized'])
  plt.title('ttf')
  plt.show()
  '''




# TODO(bhaishahster) : Make predictions, given stimulus

