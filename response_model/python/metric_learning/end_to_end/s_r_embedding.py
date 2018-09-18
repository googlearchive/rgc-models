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
r"""Jointly embed stimulus and responses, learn time courses, RF fixed."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import tensorflow as tf
import tensorflow.contrib.slim as slim
from absl import app

import numpy as np, h5py,numpy

# for plotting stuff
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# for embedding stuff
import retina.prosthesis.end_to_end.embedding0 as em
import tensorflow.models.research.tranformer.spatial_transformer as spatial_transformer

# utils
import retina.prosthesis.end_to_end.utils as utils

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=()):

  # load default data
  data = utils.get_data_retina()

  # verify utils
  utils.verify_data(data)

    #########################################################################
  # Compute RF from EI.
  with tf.Session() as sess:

    from IPython import embed; embed()
    # Embed stimulus
    n_cells = data['responses'].shape[1]
    stim_tf = tf.placeholder(tf.float32,
                             shape=[None, data['stimx'],
                                    data['stimy'], 30]) # batch x X x Y x 30
    ttf_tf = tf.Variable(np.ones(30).astype(np.float32))
    filt = tf.expand_dims(tf.expand_dims(tf.expand_dims(ttf_tf, 0), 0), 3)
    stim_time_filt = tf.nn.conv2d(stim_tf, filt,
                                  strides=[1, 1, 1, 1], padding='SAME') # batch x X x Y x 1

    # Embed response using RF
    resp_tf = tf.placeholder(tf.float32, shape=[None, n_cells]) # batch x n_cells
    rf_tf = tf.constant(data['rfs'].astype(np.float32))  # X x Y x n_cells
    rf_tf_flat = tf.reshape(rf_tf,
                            [data['stimx'] * data['stimy'], n_cells])
    resp_decode_flat = tf.matmul(resp_tf, tf.transpose(rf_tf_flat))
    resp_decode = tf.expand_dims(tf.reshape(resp_decode_flat,
                                            [-1, data['stimx'],
                                             data['stimy']]), 3)

    loss = tf.norm(stim_time_filt - resp_decode, ord='euclidean')

    train_op = tf.train.AdagradOptimizer(0.1).minimize(loss, var_list=[ttf_tf])
    sess.run(tf.global_variables_initializer())

    plt.ion()
    prev_loss = []
    loss_plot = []
    for iiter in range(1000000):
      ttf_np = sess.run(ttf_tf)

      # plot learned ttf on each iteration
      plt.clf()
      plt.subplot(1, 2, 1)
      plt.plot(ttf_np)

      # update parameters
      stim_batch, resp_batch = get_sr_batch(data, batch_size=1000,
                                               stim_history=30)
      feed_dict = {stim_tf: stim_batch, resp_tf: resp_batch}
      _, l_np = sess.run([train_op, loss], feed_dict=feed_dict)

      # plot loss curve
      prev_loss += [l_np]
      if len(prev_loss) > 10:
        prev_loss = prev_loss[-10:]
      loss_plot += [np.mean(prev_loss)]
      plt.subplot(1, 2, 2)
      plt.plot(loss_plot)
      plt.show()
      plt.draw()
      plt.pause(0.0001)


def get_sr_batch(data, batch_size=100, stim_history=30):
  """Get a batch of training data."""

  stim = data['stimulus']
  resp = data['responses']

  stim_batch = np.zeros((batch_size, stim.shape[1],
                         stim.shape[2], stim_history))
  resp_batch = np.zeros((batch_size, resp.shape[1]))

  random_times = np.random.randint(stim_history, stim.shape[0]-1, batch_size)
  for isample in range(batch_size):
    itime = random_times[isample]
    stim_batch[isample, :, :, :] = np.transpose(stim[itime:
                                                     itime-stim_history:-1,
                                                     :, :],
                                                [1, 2, 0])
    resp_batch[isample, :] = resp[itime, :]


  return stim_batch, resp_batch



if __name__ == '__main__':
   app.run()
