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
r"""Learn LN model for multiple cells in a population."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import tensorflow as tf
import tensorflow.contrib.slim as slim
from absl import app
from absl import gfile
import numpy as np, h5py,numpy
import scipy.io as sio
from numpy.random import RandomState

# for plotting stuff
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import retina.response_model.python.metric_learning.end_to_end.utils as utils
import retina.response_model.python.metric_learning.end_to_end.data_util as data_util
import retina.response_model.python.metric_learning.end_to_end.config as config

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=()):

  # Load stimulus-response data
  datasets = gfile.ListDirectory(FLAGS.src_dir)
  responses = []
  print(datasets)
  for icnt, idataset in enumerate([datasets]): #TODO(bhaishahster): HACK.

    fullpath = os.path.join(FLAGS.src_dir, idataset)
    if gfile.IsDirectory(fullpath):
      key = 'stim_%d' % icnt
      op = data_util.get_stimulus_response(FLAGS.src_dir, idataset, key)
      stimulus, resp, dimx, dimy, num_cell_types = op

      responses += resp

    for idataset in range(len(responses)):
      k, b, ttf = fit_ln_population(responses[idataset]['responses'], stimulus)  # Use FLAGS.taskID
      save_dict = {'k': k, 'b': b, 'ttf': ttf}

      save_analysis_filename = os.path.join(FLAGS.save_folder,
                                            responses[idataset]['piece']
                                            + '_ln_model.pkl')
      pickle.dump(save_dict, gfile.Open(save_analysis_filename, 'w'))



def fit_ln_population(response, stimulus, reg=0.00001):

  leng = np.minimum(response.shape[0], stimulus.shape[0])
  stimulus = stimulus[:leng, :, :]
  response = response[:leng, :]
  n_cells = response.shape[1]

  stimx = stimulus.shape[1]
  stimy = stimulus.shape[2]
  stim_2d = np.reshape(stimulus, [-1, stimx*stimy])

  # sta = stim_2d[:-7].T.dot(response[7:, :])

  with tf.Graph().as_default():
    with tf.Session() as sess:


      stim_tf = tf.placeholder(tf.float32) # T x stimx x stimy
      resp_tf = tf.placeholder(tf.float32) # T x # nells
      k_tf = tf.Variable(np.zeros((stimx, stimy, n_cells)).astype(np.float32))
      b_tf = tf.Variable(np.float32(np.zeros(n_cells)))

      stim_tf_flat = tf.reshape(stim_tf, [-1, stimx * stimy])

      # convolve each pixel in time.
      ttf_tf = tf.Variable(0.01 * np.ones(30).astype(np.float32))

      tfd = tf.expand_dims
      ttf_4d = tfd(tfd(tfd(ttf_tf, 1), 2), 3)
      stim_pad = tf.pad(stim_tf_flat, np.array([[29, 0], [0, 0]]).astype(np.int))
      stim_4d = tfd(tfd(tf.transpose(stim_pad, [1, 0]), 2), 3)
      stim_smooth = tf.nn.conv2d(stim_4d, ttf_4d, strides=[1, 1, 1, 1], padding="VALID")

      stim_smooth_2d = tf.squeeze(tf.transpose(stim_smooth, [2, 1, 0, 3]))

      k_tf_flat = tf.reshape(k_tf, [stimx*stimy, n_cells])

      lam_raw = tf.matmul(stim_smooth_2d, k_tf_flat) + b_tf
      lam = tf.exp(lam_raw)
      loss = tf.reduce_mean(lam) - tf.reduce_mean(resp_tf * lam_raw)
      train_op_part = tf.train.AdamOptimizer(0.01).minimize(loss)

      # Locally reweighted L1
      neighbor_mat = utils.get_neighbormat(np.ones((stimx, stimy)))
      n_mat = tf.constant(neighbor_mat.astype(np.float32))
      eps_neigh = 0.001

      with tf.control_dependencies([train_op_part]):
        wts_tf = 1 / (tf.matmul(n_mat, tf.reshape(tf.abs(k_tf), [stimx * stimy, n_cells])) + eps_neigh)
        wts_tf_3d = tf.reshape(wts_tf, [stimx, stimy, n_cells])
        proj_k = tf.assign(k_tf, tf.nn.relu(k_tf - wts_tf_3d * reg) - tf.nn.relu(-k_tf - wts_tf_3d * reg))

      train_op = tf.group(train_op_part, proj_k)
      sess.run(tf.global_variables_initializer())


      # loss_np_prev = np.inf
      eps = 1e-4
      for iiter in range(10000):
        tms = np.random.randint(leng - 10000)
        _, loss_np = sess.run([train_op, loss],
                              feed_dict={stim_tf : stimulus[tms:tms+10000, :].astype(np.float32),
                                         resp_tf : response[tms:tms+10000, :].astype(np.float32)})
        print(loss_np)
        # if np.abs(loss_np - loss_np_prev) < eps:
        #   break
        #else:
        # loss_np_prev = loss_np
        '''
        plt.ion()
        plt.subplot(2, 1, 1)
        plt.cla()
        plt.plot(sess.run(ttf_tf))

        plt.subplot(2, 1, 2)
        plt.cla()
        k_np = sess.run(k_tf)
        plt.imshow(k_np[:, :, 23].T, interpolation='nearest', cmap='gray')
        plt.show()
        plt.draw()
        plt.pause(0.05)
        '''
      k = sess.run(k_tf)
      b = sess.run(b_tf)
      ttf = sess.run(ttf_tf)

      return k, b, ttf

def predict_responses_ln(stimulus, k, b, ttf, n_trials=1):

  with tf.Graph().as_default():
    with tf.Session() as sess:

      stim_tf = tf.placeholder(tf.float32) # T x stimx x stimy
      k_tf = tf.constant(k.astype(np.float32))
      b_tf = tf.constant(np.float32(b))

      stim_tf_flat = tf.reshape(stim_tf, [-1, stimx * stimy])

      # convolve each pixel in time.
      ttf_tf = tf.constant(ttf)

      tfd = tf.expand_dims
      ttf_4d = tfd(tfd(tfd(ttf_tf, 1), 2), 3)
      stim_pad = tf.pad(stim_tf_flat, np.array([[29, 0], [0, 0]]).astype(np.int))
      stim_4d = tfd(tfd(tf.transpose(stim_pad, [1, 0]), 2), 3)
      stim_smooth = tf.nn.conv2d(stim_4d, ttf_4d, strides=[1, 1, 1, 1], padding="VALID")

      stim_smooth_2d = tf.squeeze(tf.transpose(stim_smooth, [2, 1, 0, 3]))

      k_tf_flat = tf.reshape(k_tf, [stimx*stimy, n_cells])

      lam_raw = tf.matmul(stim_smooth_2d, k_tf_flat) + b_tf
      lam = tf.nn.softplus(lam_raw)

      sess.run(tf.global_variables_initializer())
      lam_np = sess.run(lam, feed_dict={stim_tf: stimulus.astype(np.float32)})

      # repeat lam_np for number of trials
      lam_np = np.repeat(np.expand_dims(lam_np, 0), n_trials, axis=0)
      spikes = np.random.poisson(lam_np)
      return spikes, lam_np

if __name__ == '__main__':
  app.run(main)
