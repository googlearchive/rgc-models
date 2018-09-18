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
"""Whole population fits"""
import sys
import tensorflow as tf
from absl import app
from absl import flags
from absl import gfile

import matplotlib
matplotlib.use('TkAgg')

import numpy as np, h5py
import scipy.io as sio
from scipy import ndimage
import random

FLAGS = flags.FLAGS
flags.DEFINE_float('lam_W', 0.0001, 'sparsitiy regularization of w')
flags.DEFINE_float('lam_a', 0.0001, 'sparsitiy regularization of a')
flags.DEFINE_integer('ratio_SU', 2, 'ratio of subunits/cells')
flags.DEFINE_float('su_grid_spacing', 5.7, 'grid spacing')
flags.DEFINE_integer('np_randseed', 23, 'numpy RNG seed')
flags.DEFINE_integer('randseed', 65, 'python RNG seed')
flags.DEFINE_float('bias_init_scale', -1, 'bias initialized at scale*std')
flags.DEFINE_integer('model_id', 0, 'which model to learn?');

flags.DEFINE_string('save_location',
                    '/home/bhaishahster/Downloads/',
                    'where to store logs and outputs?');

flags.DEFINE_string('data_location',
                    '/home/bhaishahster/Downloads/',
                    'where to take data from?')


def hex_grid(gridx, d, n):
  x_log = np.array([])
  y_log = np.array([])
  for i in range(n):
    x_log = (np.append(x_log, (((i*d)%gridx) +
                               (np.floor(i*d/gridx)%2)*d/2)) +
             np.random.randn(1)*0.01)
    y_log = np.append(y_log, np.floor((i*d/gridx))*d/2) + np.random.randn(1)*0.01

  return x_log, y_log


def gauss_su(x_log, y_log, gridx=80, gridy=40):
  ns = x_log.shape[0]
  wts = np.zeros((3200, ns))
  for isu in range(ns):
    xx = np.zeros((gridy, gridx))
    if((np.round(y_log[isu]) >= gridy) |
       (np.round(y_log[isu]) < 0) |
       (np.round(x_log[isu]) >= gridx) | (np.round(x_log[isu]) < 0)):
      continue

    xx[np.round(y_log[isu]), np.round(x_log[isu])] = 1
    blurred_xx = ndimage.gaussian_filter(xx, sigma=2)
    wts[:,isu] = np.ndarray.flatten(blurred_xx)
  return wts


def initialize_su(n_su=107*2, gridx=80, gridy=40, spacing=5.7):
  spacing = FLAGS.su_grid_spacing
  x_log, y_log = hex_grid(gridx, spacing, n_su)
  wts = gauss_su(x_log, y_log)
  return wts


def main(argv):

  print('\nCode started')

  np.random.seed(FLAGS.np_randseed)
  random.seed(FLAGS.randseed)

  ## Load data
  file = h5py.File(FLAGS.data_location + 'Off_parasol.mat', 'r')
  print('\ndataset loaded')

  # Load Masked movie
  data = file.get('maskedMovdd')
  stimulus = np.array(data)
  cells = file.get('cells')
  nCells = cells.shape[0]
  total_mask_log = file.get('totalMaskAccept_log');
  Nsub = FLAGS.ratio_SU*nCells
  stim_dim = stimulus.shape[1]

  # Load spike Response of cells
  data = file.get('Y')
  response = np.array(data, dtype='float32')
  tot_spks = np.squeeze(np.sum(response,axis=0))

  print(sys.getsizeof(file))
  print(sys.getsizeof(stimulus))
  print(sys.getsizeof(response))

  with tf.Session() as sess:
    stim = tf.placeholder(tf.float32, shape=[None, stim_dim], name='stim')
    resp = tf.placeholder(tf.float32, name='resp')
    data_len = tf.placeholder(tf.float32, name='data_len')

    if FLAGS.model_id == 0:
      # lam_c(X) = sum_s(a_cs relu(k_s.x)) , a_cs>0
      w_init = initialize_su(n_su=Nsub)
      a_init = np.random.rand(Nsub, nCells)
      w = tf.Variable(np.array(w_init, dtype='float32'))
      a = tf.Variable(np.array(a_init, dtype='float32'))
      lam = tf.matmul(tf.nn.relu(tf.matmul(stim, w)), tf.nn.relu(a)) + 0.0001
      loss = (tf.reduce_sum(lam/tot_spks)/120. - tf.reduce_sum(resp*tf.log(lam)/tot_spks)) / data_len
      + FLAGS.lam_W*tf.reduce_sum(tf.abs(w)) + FLAGS.lam_a*tf.reduce_sum(tf.abs(a))
      train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=[w, a])
      save_filename = (FLAGS.save_location + 'data_model=' + str(FLAGS.model_id) +
                    '_lam_w=' + str(FLAGS.lam_W) +
                    '_lam_a='+str(FLAGS.lam_a) + '_ratioSU=' + str(FLAGS.ratio_SU) +
                    '_grid_spacing=' + str(FLAGS.su_grid_spacing) + '_normalized')

    if FLAGS.model_id == 1:
      # lam_c(X) = sum_s( relu(k_s.x + a_cs)); with a_cs initialized to a bit high negative value
      scale_cells = np.squeeze(np.sum(response,0))
      scale_cells = scale_cells / np.sum(scale_cells)
      w_init = initialize_su(n_su=Nsub)
      kx = stimulus[1000:2000,:].dot(w_init)
      std = np.diag(np.sqrt(kx.T.dot(kx) / kx.shape[0]))
      a_init =  FLAGS.bias_init_scale*std*np.ones((nCells,1,Nsub))
      w = tf.Variable(np.array(w_init, dtype='float32'))
      a = tf.Variable(np.array(a_init, dtype='float32'))
      lam = tf.transpose(tf.reduce_sum(tf.nn.relu(tf.matmul(stim,w) + a), 2)) * scale_cells + 0.001
      loss = (tf.reduce_sum(lam/tot_spks)/120. - tf.reduce_sum(resp*tf.log(lam)/tot_spks)) / data_len
      train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=[w, a])
      save_filename = (FLAGS.save_location + 'data_model=' + str(FLAGS.model_id) +
                    '_scale_fr_bias_init=' + str(FLAGS.bias_init_scale) + '_ratioSU=' + str(FLAGS.ratio_SU) +
                    '_grid_spacing=' + str(FLAGS.su_grid_spacing) + '_normalized')

    if FLAGS.model_id == 2:
      # lam_c(X) = sum_s(exp(k_s.x + b_cs)) ; used in earlier models.
      w_init = initialize_su(n_su=Nsub)
      kx = stimulus[1000:2000,:].dot(w_init)
      std = np.diag(np.sqrt(kx.T.dot(kx) / kx.shape[0]))
      a_init =  FLAGS.bias_init_scale*std*np.ones((nCells,1,Nsub))
      w = tf.Variable(np.array(w_init, dtype='float32'))
      a = tf.Variable(np.array(a_init, dtype='float32'))
      lam = tf.transpose(tf.reduce_sum(tf.exp(tf.matmul(stim,w) + a), 2))
      loss = (tf.reduce_sum(lam/tot_spks)/120. - tf.reduce_sum(resp*tf.log(lam)/tot_spks)) / data_len
      train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=[w, a])
      save_filename = (FLAGS.save_location + 'data_model=' + str(FLAGS.model_id) +
                    '_bias_init=' + str(FLAGS.bias_init_scale) + '_ratioSU=' + str(FLAGS.ratio_SU) +
                    '_grid_spacing=' + str(FLAGS.su_grid_spacing) + '_normalized')


    if FLAGS.model_id == 3:
      # lam_c(X) = sum_s(relu(k_s.x + a_cs)^2); MEL approximation of log-likelihood
      sigma = np.diag(np.diag(stimulus[100:200,:].T.dot(stimulus[100:200,: ])))
      sig_tf = tf.Variable(sigma,dtype='float32')
      w_init = initialize_su(n_su=Nsub)
      a_init = np.random.rand(Nsub, nCells)

      w = tf.Variable(np.array(w_init, dtype='float32'))
      a = tf.Variable(np.array(a_init, dtype='float32'))
      a_rel = tf.nn.relu(a)

      lam = tf.matmul(tf.pow(tf.nn.relu(tf.matmul(stim, w)), 2), a_rel) + 0.0001
      loss_p1 = tf.reduce_sum(tf.matmul(tf.transpose(a_rel / tot_spks),tf.expand_dims(tf.diag_part(tf.matmul(tf.transpose(w),tf.matmul(sig_tf,w))) / 2,1)))
      loss = (loss_p1 / 120.) - (tf.reduce_sum(resp * tf.log(lam) / tot_spks)) / data_len
      + FLAGS.lam_W*tf.reduce_sum(tf.abs(w)) + FLAGS.lam_a*tf.reduce_sum(tf.abs(a))
      train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=[w, a])
      save_filename = (FLAGS.save_location + 'data_model=' + str(FLAGS.model_id) +
                    '_lam_w=' + str(FLAGS.lam_W) +
                    '_lam_a='+str(FLAGS.lam_a) + '_ratioSU=' + str(FLAGS.ratio_SU) +
                    '_grid_spacing=' + str(FLAGS.su_grid_spacing) + '_normalized')


    if FLAGS.model_id == 4:
      # f(X) = sum_s(a_cs relu(k_s.x)), acs - any sign, logistic loss.
      w_init = initialize_su(n_su=Nsub)*0.01
      a_init = np.random.randn(Nsub, nCells)*0.01
      w = tf.Variable(np.array(w_init, dtype='float32'))
      a = tf.Variable(np.array(a_init, dtype='float32'))

      b_init = np.random.randn(nCells)#np.log((np.sum(response,0))/(response.shape[0]-np.sum(response,0)))
      b = tf.Variable(b_init,dtype='float32')
      f = tf.matmul(tf.nn.relu(tf.matmul(stim, w)), a) + b
      loss = -tf.reduce_sum(tf.log(1+tf.exp(-resp*f))) / data_len
      + FLAGS.lam_W*tf.reduce_sum(tf.abs(w)) + FLAGS.lam_a*tf.reduce_sum(tf.abs(a))
      sigmoid_input = -resp*f
      train_step = tf.train.AdagradOptimizer(1e-3).minimize(loss, var_list=[w, a, b])
      save_filename = (FLAGS.save_location + 'data_model=' + str(FLAGS.model_id) +
                    '_lam_w=' + str(FLAGS.lam_W) +
                    '_lam_a='+str(FLAGS.lam_a) + '_ratioSU=' + str(FLAGS.ratio_SU) +
                    '_grid_spacing=' + str(FLAGS.su_grid_spacing) + '_normalized')
      response = 2 * (response - 0.5) # make responses +1/-1

    # initialize the model
    logfile = gfile.Open(save_filename + '.txt', "w")
    logfile.write('Starting new code\n')
    logfile.flush()
    sess.run(tf.initialize_all_variables())

    # Do the fitting
    batchsz = 100
    icnt = 0
    fd_test = {stim: stimulus.astype('float32')[216000-1000: 216000-1, :],
               resp: response.astype('float32')[216000-1000: 216000-1, :],
               data_len: 1000}
    ls_train_log = np.array([])
    ls_test_log = np.array([])
    tms = np.random.permutation(np.arange(216000-1000))
    for istep in range(100000):
      fd_train = {stim: stimulus.astype('float32')[tms[icnt: icnt+batchsz], :],
                  resp: response.astype('float32')[tms[icnt: icnt+batchsz], :],
                  data_len: batchsz}
      sess.run(train_step, feed_dict=fd_train)
      if istep%10 == 0:
        # compute training and testing losses
        ls_train = sess.run(loss, feed_dict=fd_train)
        ls_test = sess.run(loss, feed_dict=fd_test)
        ls_train_log = np.append(ls_train_log, ls_train)
        ls_test_log = np.append(ls_test_log, ls_test)
        # log results
        logfile.write('\nIterations: ' + str(istep) + ' Training error: '
                      + str(ls_train) + ' Testing error: ' + str(ls_test) +
                      '  w_l1_norm: ' + str(np.sum(np.abs(w.eval()))) +
                      ' a_l1_norm: ' + str(np.sum(np.abs(a.eval()))))
        logfile.flush()

        sigmoid_inp = sess.run(sigmoid_input, feed_dict=fd_test)
        print(np.percentile(sigmoid_inp,5),np.percentile(sigmoid_inp,50),np.percentile(sigmoid_inp,95))

        sio.savemat(save_filename + '.mat',
                    {'w': w.eval(), 'a': a.eval(), 'w_init': w_init,
                            'a_init': a_init, 'ls_train_log': ls_train_log,
                            'ls_test_log': ls_test_log})

      icnt += batchsz
      if icnt > 216000-1000:
        icnt = 0
        tms = np.random.permutation(np.arange(216000-1000))

  logfile.close()
if __name__ == '__main__':
  app.run()

