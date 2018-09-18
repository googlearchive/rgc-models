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
"""Whole population fits, exponentiated gradient.
"""
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
flags.DEFINE_float('lam_w', 0.0001, 'sparsitiy regularization of w')

flags.DEFINE_integer('ratio_SU', 2, 'ratio of subunits/cells')
flags.DEFINE_float('su_grid_spacing', 5.7, 'grid spacing')
flags.DEFINE_integer('np_randseed', 23, 'numpy RNG seed')
flags.DEFINE_integer('randseed', 65, 'python RNG seed')
flags.DEFINE_float('bias_init_scale', -1, 'bias initialized at scale*std')
flags.DEFINE_integer('model_id', 0, 'which model to learn?');
flags.DEFINE_float('eta_a', 100, 'learning rate for optimization of w')
flags.DEFINE_float('eta_w', 1, 'learning rate for optimization of a')
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
      # MODEL: lam_c(X) = sum_s(a_cs relu(k_s.x)) , a_cs>0

      w_init = initialize_su(n_su=Nsub)
      a_init = np.random.rand(Nsub, nCells)
      a_init = a_init / np.sum(a_init, axis=0)# normalize initial a
      w = tf.Variable(np.array(w_init, dtype='float32'))
      a = tf.Variable(np.array(a_init, dtype='float32'))

      lam = tf.matmul(tf.nn.relu(tf.matmul(stim, w)), a) + 0.0001
      loss = (tf.reduce_sum(lam/tot_spks)/120. - tf.reduce_sum(resp*tf.log(lam)/tot_spks)) / data_len
      loss_with_reg = loss + FLAGS.lam_w*tf.reduce_sum(tf.abs(w))
      # steps to update a
      # as 'a' is positive, this is op soft-thresholding for L1 and projecting to feasible set
      eta_a_tf = tf.constant(np.squeeze(FLAGS.eta_a),dtype='float32')
      grads_a = tf.gradients(loss, a)
      exp_grad_a = tf.squeeze(tf.mul(a,tf.exp(-eta_a_tf * grads_a)))
      a_update = tf.assign(a,exp_grad_a/tf.reduce_sum(exp_grad_a,0))

      # steps to update w
      # gradient update of 'w'..
      train_step_w = tf.train.AdagradOptimizer(FLAGS.eta_w).minimize(loss, var_list=[w])
      # do soft thresholding for 'w'
      soft_th_w = tf.assign(w, tf.nn.relu(w - FLAGS.eta_w * FLAGS.lam_w) - tf.nn.relu(- w - FLAGS.eta_w * FLAGS.lam_w))

      save_filename = (FLAGS.save_location + 'data_model=' + str(FLAGS.model_id) +
                    '_lam_w=' + str(FLAGS.lam_w) +
                    '_eta_w=' + str(FLAGS.eta_w) + '_eta_a=' + str(FLAGS.eta_a) + '_ratioSU=' + str(FLAGS.ratio_SU) +
                    '_grid_spacing=' + str(FLAGS.su_grid_spacing) + '_eg')

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
    ls_train_reg_log = np.array([])
    ls_test_log = np.array([])
    ls_test_reg_log = np.array([])

    tms = np.random.permutation(np.arange(216000-1000))
    for istep in range(100000):
      fd_train = {stim: stimulus.astype('float32')[tms[icnt: icnt+batchsz], :],
                  resp: response.astype('float32')[tms[icnt: icnt+batchsz], :],
                  data_len: batchsz}
      # update w
      # gradient step for 'a' and 'w'
      sess.run(train_step_w, feed_dict=fd_train)
      # soft thresholding for w
      sess.run(soft_th_w)

      # update a
      sess.run(a_update, feed_dict=fd_train)

      if istep%10 == 0:
        # compute training and testing losses
        ls_train = sess.run(loss, feed_dict=fd_train)
        ls_train_log = np.append(ls_train_log, ls_train)
        ls_train_reg = sess.run(loss_with_reg, feed_dict=fd_train)
        ls_train_reg_log = np.append(ls_train_reg_log, ls_train_reg)

        ls_test = sess.run(loss, feed_dict=fd_test)
        ls_test_log = np.append(ls_test_log, ls_test)
        ls_test_reg = sess.run(loss_with_reg, feed_dict=fd_test)
        ls_test_reg_log = np.append(ls_test_reg_log, ls_test_reg)

        # print exponentiated 'a'
        '''
        a_eval = a.eval()
        print(a_eval)
        g_a = sess.run(grads_a,feed_dict=fd_test)
        print(g_a)
        '''
        # log results
        logfile.write('\nIterations: ' + str(istep) + ' Training loss: ' + str(ls_train) + ' with reg: ' + str(ls_train_reg) + ' Testing loss: ' + str(ls_test) + ' with reg: ' + str(ls_test_reg) + '  w_l1_norm: ' + str(np.sum(np.abs(w.eval()))) +' a_l1_norm: ' + str(np.sum(np.abs(a.eval()))))
        logfile.flush()

        sio.savemat(save_filename + '.mat',
                    {'w': w.eval(), 'a': a.eval(), 'w_init': w_init,
                     'a_init': a_init, 'ls_train_log': ls_train_log,
                     'ls_train_reg_log': ls_train_reg_log,
                     'ls_test_log': ls_test_log, 'ls_test_reg_log': ls_test_reg_log})

      icnt += batchsz
      if icnt > 216000-1000:
        icnt = 0
        tms = np.random.permutation(np.arange(216000-1000))

  logfile.close()
if __name__ == '__main__':
  app.run()

