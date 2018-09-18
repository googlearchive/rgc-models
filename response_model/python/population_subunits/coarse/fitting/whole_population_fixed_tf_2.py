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
"""Whole population model"""

import sys
import os.path
import tensorflow as tf
from absl import app
from absl import flags
from absl import gfile
import cPickle as pickle
import matplotlib
matplotlib.use('TkAgg')

import numpy as np, h5py
import scipy.io as sio
from scipy import ndimage
import random

FLAGS = flags.FLAGS
flags.DEFINE_float('lam_w', 0.0001, 'sparsitiy regularization of w')
flags.DEFINE_float('lam_a', 0.0001, 'sparsitiy regularization of a')
flags.DEFINE_integer('ratio_SU', 7, 'ratio of subunits/cells')
flags.DEFINE_float('su_grid_spacing', 3, 'grid spacing')
flags.DEFINE_integer('np_randseed', 23, 'numpy RNG seed')
flags.DEFINE_integer('randseed', 65, 'python RNG seed')
flags.DEFINE_float('eta_w', 1e-3, 'learning rate for optimization functions')
flags.DEFINE_float('eta_a', 1e-2, 'learning rate for optimization functions')
flags.DEFINE_float('bias_init_scale', -1, 'bias initialized at scale*std')
flags.DEFINE_string('model_id', 'relu', 'which model to learn?');
flags.DEFINE_float('step_sz', 10, 'step size for learning algorithm')
flags.DEFINE_integer('window', 3, 'size of window for each subunit in relu_window model')
flags.DEFINE_integer('stride', 3, 'stride for relu_window')
flags.DEFINE_string('folder_name', 'experiment4', 'folder where to store all the data')

flags.DEFINE_string('save_location',
                    '/home/bhaishahster/',
                    'where to store logs and outputs?');

flags.DEFINE_string('data_location',
                    '/home/bhaishahster/data_breakdown/',
                    'where to take data from?')

flags.DEFINE_integer('batchsz', 1000, 'batch size for training')
flags.DEFINE_integer('n_chunks', 216, 'number of data chunks') # should be 216
flags.DEFINE_integer('n_b_in_c', 1, 'number of batches in one chunk of data')

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


def initialize_su(n_su=107*10, gridx=80, gridy=40, spacing=5.7):
  spacing = FLAGS.su_grid_spacing
  x_log, y_log = hex_grid(gridx, spacing, n_su)
  wts = gauss_su(x_log, y_log)
  return wts


def get_test_data():
  # stimulus.astype('float32')[216000-1000: 216000-1, :]
  # response.astype('float32')[216000-1000: 216000-1, :]
  # length
  test_data_chunks = [FLAGS.n_chunks];
  for ichunk in test_data_chunks:
    filename = FLAGS.data_location + 'Off_par_data_' + str(ichunk) + '.mat'
    file_r = gfile.Open(filename, 'r')
    data = sio.loadmat(file_r)
    stim_part = data['maskedMovdd_part'].T
    resp_part = data['Y_part'].T
    test_len = stim_part.shape[0]
  #logfile.write('\nReturning test data')
  return stim_part, resp_part, test_len


# global stimulus variables
stim_train_part = np.array([])
resp_train_part = np.array([])
chunk_order = np.array([])
cells_choose = np.array([])
chosen_mask = np.array([])

def get_next_training_batch(iteration):
  # stimulus.astype('float32')[tms[icnt: icnt+FLAGS.batchsz], :],
  # response.astype('float32')[tms[icnt: icnt+FLAGS.batchsz], :]
  # FLAGS.batchsz

  # we will use global stimulus and response variables
  global stim_train_part
  global resp_train_part
  global chunk_order

  togo = True
  while togo:
    if(iteration % FLAGS.n_b_in_c == 0):
    # load new chunk of data
      ichunk = (iteration / FLAGS.n_b_in_c) % (FLAGS.n_chunks - 1 ) # last one chunks used for testing
      if (ichunk == 0): # shuffle training chunks at start of training data
        chunk_order = np.random.permutation(np.arange(FLAGS.n_chunks-1)) # remove first chunk - weired?
      #  if logfile != None :
      #    logfile.write('\nTraining chunks shuffled')

      if chunk_order[ichunk] + 1 != 1:
        filename = FLAGS.data_location + 'Off_par_data_' + str(chunk_order[ichunk] + 1) + '.mat'
        file_r = gfile.Open(filename, 'r')
        data = sio.loadmat(file_r)
        stim_train_part = data['maskedMovdd_part']
        resp_train_part = data['Y_part']

        ichunk = chunk_order[ichunk] + 1
        while stim_train_part.shape[1] < FLAGS.batchsz:
          #print('Need to add extra chunk')
          if (ichunk> FLAGS.n_chunks):
            ichunk = 2
          filename = FLAGS.data_location + 'Off_par_data_' + str(ichunk) + '.mat'
          file_r = gfile.Open(filename, 'r')
          data = sio.loadmat(file_r)
          stim_train_part = np.append(stim_train_part, data['maskedMovdd_part'], axis=1)
          resp_train_part = np.append(resp_train_part, data['Y_part'], axis=1)
          #print(np.shape(stim_train_part), np.shape(resp_train_part))
          ichunk = ichunk + 1

      #  if logfile != None:
      #    logfile.write('\nNew training data chunk loaded at: '+ str(iteration) + ' chunk #: ' + str(chunk_order[ichunk]))


    ibatch = iteration % FLAGS.n_b_in_c
    try:
      stim_train = np.array(stim_train_part[:,ibatch: ibatch + FLAGS.batchsz], dtype='float32').T
      resp_train = np.array(resp_train_part[:,ibatch: ibatch + FLAGS.batchsz], dtype='float32').T
      togo=False
    except:
      iteration = np.random.randint(1,100000)
      print('Load exception iteration: ' + str(iteration) + 'chunk: ' + str(chunk_order[ichunk]) + 'batch: ' + str(ibatch) )
      togo=True

  return stim_train, resp_train, FLAGS.batchsz

def main(argv):

  print('\nCode started')
  print('Model is ' + FLAGS.model_id)
  np.random.seed(FLAGS.np_randseed)
  random.seed(FLAGS.randseed)
  global chunk_order
  chunk_order = np.random.permutation(np.arange(FLAGS.n_chunks-1))

  ## Load data summary

  filename = FLAGS.data_location + 'data_details.mat'
  summary_file = gfile.Open(filename, 'r')
  data_summary = sio.loadmat(summary_file)
  cells = np.squeeze(data_summary['cells'])
  nCells = cells.shape[0]
  stim_dim = np.squeeze(data_summary['stim_dim'])
  tot_spks = np.squeeze(data_summary['tot_spks'])
  total_mask = np.squeeze(data_summary['totalMaskAccept_log']).T
  print(np.shape(total_mask))
  print('\ndataset summary loaded')

  # decide the number of subunits to fit
  Nsub = FLAGS.ratio_SU*nCells
  with tf.Session() as sess:
    stim = tf.placeholder(tf.float32, shape=[None, stim_dim], name='stim')
    resp = tf.placeholder(tf.float32, name='resp')
    data_len = tf.placeholder(tf.float32, name='data_len')

    if FLAGS.model_id == 'relu':
      # lam_c(X) = sum_s(a_cs relu(k_s.x)) , a_cs>0
      short_filename = ('data_model=' + str(FLAGS.model_id) +
                    '_lam_w=' + str(FLAGS.lam_w) +
                    '_lam_a='+str(FLAGS.lam_a) + '_ratioSU=' + str(FLAGS.ratio_SU) +
                    '_grid_spacing=' + str(FLAGS.su_grid_spacing) + '_normalized_bg')

    if FLAGS.model_id == 'exp':
      short_filename = ('data_model=' + str(FLAGS.model_id) +
                    '_bias_init=' + str(FLAGS.bias_init_scale) + '_ratioSU=' + str(FLAGS.ratio_SU) +
                    '_grid_spacing=' + str(FLAGS.su_grid_spacing) + '_normalized_bg')

    if FLAGS.model_id == 'mel_re_pow2':
      short_filename = ('data_model=' + str(FLAGS.model_id) +
                    '_lam_w=' + str(FLAGS.lam_w) +
                    '_lam_a='+str(FLAGS.lam_a) + '_ratioSU=' + str(FLAGS.ratio_SU) +
                    '_grid_spacing=' + str(FLAGS.su_grid_spacing) + '_normalized_bg')

    if FLAGS.model_id == 'relu_logistic':
      short_filename = ('data_model=' + str(FLAGS.model_id) +
                    '_lam_w=' + str(FLAGS.lam_w) +
                    '_lam_a='+str(FLAGS.lam_a) + '_ratioSU=' + str(FLAGS.ratio_SU) +
                    '_grid_spacing=' + str(FLAGS.su_grid_spacing) + '_normalized_bg')

    if FLAGS.model_id == 'relu_proximal':
      short_filename = ('data_model=' + str(FLAGS.model_id) +
                    '_lam_w=' + str(FLAGS.lam_w) +
                    '_lam_a='+str(FLAGS.lam_a) + '_eta_w=' + str(FLAGS.eta_w) + '_eta_a=' + str(FLAGS.eta_a) + '_ratioSU=' + str(FLAGS.ratio_SU) +
                    '_grid_spacing=' + str(FLAGS.su_grid_spacing) + '_proximal_bg')

    if FLAGS.model_id == 'relu_eg':
      short_filename = ('data_model=' + str(FLAGS.model_id) +
                    '_lam_w=' + str(FLAGS.lam_w) +
                    '_eta_w=' + str(FLAGS.eta_w) + '_eta_a=' + str(FLAGS.eta_a) + '_ratioSU=' + str(FLAGS.ratio_SU) +
                    '_grid_spacing=' + str(FLAGS.su_grid_spacing) + '_eg_bg')

    if FLAGS.model_id == 'relu_window':
      short_filename = ('data_model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) + '_lam_w=' + str(FLAGS.lam_w) + '_bg')

    if FLAGS.model_id == 'relu_window_mother':
      short_filename = ('data_model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) + '_lam_w=' + str(FLAGS.lam_w) + '_bg')

    if FLAGS.model_id == 'relu_window_mother_sfm':
      short_filename = ('data_model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) + '_lam_w=' + str(FLAGS.lam_w) + '_bg')

    if FLAGS.model_id == 'relu_window_mother_sfm_exp':
      short_filename = ('data_model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) + '_lam_w=' + str(FLAGS.lam_w) + '_bg')

    if FLAGS.model_id == 'relu_window_exp':
      short_filename = ('data_model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) + '_lam_w=' + str(FLAGS.lam_w) + '_bg')

    if FLAGS.model_id == 'relu_window_mother_exp':
      short_filename = ('data_model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) + '_lam_w=' + str(FLAGS.lam_w) + '_bg')

    if FLAGS.model_id == 'relu_window_a_support':
      short_filename = ('data_model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) + '_lam_w=' + str(FLAGS.lam_w) + '_bg')

    if FLAGS.model_id == 'exp_window_a_support':
      short_filename = ('data_model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) + '_lam_w=' + str(FLAGS.lam_w) + '_bg')

    parent_folder = FLAGS.save_location + FLAGS.folder_name + '/'
    if not gfile.IsDirectory(parent_folder):
      gfile.MkDir(parent_folder)
    FLAGS.save_location = parent_folder +short_filename + '/'
    print(gfile.IsDirectory(FLAGS.save_location))
    if not gfile.IsDirectory(FLAGS.save_location):
      gfile.MkDir(FLAGS.save_location)
    print(FLAGS.save_location)

    save_filename = FLAGS.save_location + short_filename

    '''
    # load previous iteration data, if available

    try:
      saved_filename = save_filename + '.pkl'
      saved_file = gfile.Open(saved_filename,'r')
      saved_data = pickle.load(saved_file)
      w_load = saved_data['w']
      a_load = saved_data['a']
      w_init = saved_data['w_init']
      a_init = saved_data['a_init']
      ls_train_log = np.squeeze(saved_data['ls_train_log'])
      ls_test_log = np.squeeze(saved_data['ls_test_log'])
      start_iter = np.squeeze(saved_data['last_iter'])
      chunk_order = np.squeeze(saved_data['chunk_order'])
      print(np.shape(w_init),np.shape(a_init))
      load_prev = True
    except:
      # w and a initialized same for all models! (maybe should be different for exp NL?)
      w_init = initialize_su(n_su=Nsub) * 0.01
      if FLAGS.model_id != 'exp':
        a_init = np.random.rand(Nsub, nCells) * 0.01
      else:
        a_init =  np.random.rand(nCells,1,Nsub) * 0.01
      w_load = w_init
      a_load = a_init
      ls_train_log = np.array([])
      ls_test_log = np.array([])
      start_iter=0
      print(np.shape(w_init),np.shape(a_init))
      load_prev = False
    '''
    w_init = initialize_su(n_su=Nsub) * 0.01
    if FLAGS.model_id != 'exp':
      a_init = np.random.rand(Nsub, nCells) * 0.01
    else:
      a_init =  np.random.rand(nCells,1,Nsub) * 0.01
    w_load = w_init
    a_load = a_init
    ls_train_log = np.array([])
    ls_test_log = np.array([])
    print(np.shape(w_init),np.shape(a_init))
    load_prev = False
    
    if FLAGS.model_id == 'relu':
      # LNL model with RELU nl
      w = tf.Variable(np.array(w_load, dtype='float32'))
      a = tf.Variable(np.array(a_load, dtype='float32'))
      lam = tf.matmul(tf.nn.relu(tf.matmul(stim, w)), tf.nn.relu(a)) + 0.0001
      loss_inter = (tf.reduce_sum(lam)/120. - tf.reduce_sum(resp*tf.log(lam))) / data_len
      loss = loss_inter
      + FLAGS.lam_w*tf.reduce_sum(tf.abs(w)) + FLAGS.lam_a*tf.reduce_sum(tf.abs(a))
      train_step = tf.train.AdagradOptimizer(FLAGS.step_sz).minimize(loss, var_list=[w, a])
      a_pos = tf.assign(a, (a + tf.abs(a))/2)
      def training(inp_dict):
        sess.run(train_step, feed_dict=inp_dict)
        sess.run(a_pos)
      def get_loss(inp_dict):
        ls = sess.run(loss,feed_dict = inp_dict)
        return ls
      w_summary = tf.histogram_summary('w', w)
      a_summary = tf.histogram_summary('a', a)

    if FLAGS.model_id == 'exp':
      # lam_c(X) = sum_s(exp(k_s.x + b_cs)) ; used in earlier models.
      w = tf.Variable(np.array(w_load, dtype='float32'))
      a = tf.Variable(np.array(a_load, dtype='float32'))
      lam = tf.transpose(tf.reduce_sum(tf.exp(tf.matmul(stim,w) + a), 2))
      loss_inter = (tf.reduce_sum(lam/tot_spks)/120. - tf.reduce_sum(resp*tf.log(lam)/tot_spks)) / data_len
      loss = loss_inter
      train_step = tf.train.AdamOptimizer(FLAGS.step_sz).minimize(loss, var_list=[w, a])
      def training(inp_dict):
        sess.run(train_step, feed_dict=inp_dict)
      def get_loss(inp_dict):
        ls = sess.run(loss,feed_dict = inp_dict)
        return ls
      w_summary = tf.histogram_summary('w', w)
      a_summary = tf.histogram_summary('a', a)

    if FLAGS.model_id == 'mel_re_pow2':
      # lam_c(X) = sum_s(relu(k_s.x + a_cs)^2); MEL approximation of log-likelihood
      stimulus,_,_ = get_next_training_batch(10)
      sigma = np.diag(np.diag(stimulus[1000:2000,:].T.dot(stimulus[1000:2000,: ])))
      sig_tf = tf.Variable(sigma,dtype='float32')
      w = tf.Variable(np.array(w_load, dtype='float32'))
      a = tf.Variable(np.array(a_load, dtype='float32'))
      a_pos = tf.assign(a, (a + tf.abs(a))/2)
      lam = tf.matmul(tf.pow(tf.nn.relu(tf.matmul(stim, w)), 2), a) + 0.0001
      loss_p1 = tf.reduce_sum(tf.matmul(tf.transpose(a / tot_spks),tf.expand_dims(tf.diag_part(tf.matmul(tf.transpose(w),tf.matmul(sig_tf,w))) / 2,1)))
      loss_inter = (loss_p1 / 120.) - (tf.reduce_sum(resp * tf.log(lam) / tot_spks)) / data_len
      loss = loss_inter
      + FLAGS.lam_w*tf.reduce_sum(tf.abs(w)) + FLAGS.lam_a*tf.reduce_sum(tf.abs(a))
      train_step = tf.train.AdagradOptimizer(FLAGS.step_sz).minimize(loss, var_list=[w, a])
      def training(inp_dict):
        sess.run(train_step, feed_dict=inp_dict)
        sess.run(a_pos)
      def get_loss(inp_dict):
        ls = sess.run(loss,feed_dict = inp_dict)
        return ls
      w_summary = tf.histogram_summary('w', w)
      a_summary = tf.histogram_summary('a', a)
      

    if FLAGS.model_id == 'relu_logistic':
      # f(X) = sum_s(a_cs relu(k_s.x)), acs - any sign, logistic loss.
      w = tf.Variable(np.array(w_load, dtype='float32'))
      a = tf.Variable(np.array(a_load, dtype='float32'))
      b_init = np.random.randn(nCells)#np.log((np.sum(response,0))/(response.shape[0]-np.sum(response,0)))
      b = tf.Variable(b_init,dtype='float32')
      f = tf.matmul(tf.nn.relu(tf.matmul(stim, w)), a) + b
      loss_inter = tf.reduce_sum(tf.nn.softplus(-2 * (resp - 0.5)*f))/ data_len
      loss = loss_inter
      + FLAGS.lam_w*tf.reduce_sum(tf.abs(w)) + FLAGS.lam_a*tf.reduce_sum(tf.abs(a))
      sigmoid_input = -resp*f
      train_step = tf.train.AdagradOptimizer(FLAGS.step_sz).minimize(loss, var_list=[w, a, b])
      a_pos = tf.assign(a, (a + tf.abs(a))/2)

      def training(inp_dict):
        sess.run(train_step, feed_dict=inp_dict)
        sess.run(a_pos)
      def get_loss(inp_dict):
        ls = sess.run(loss,feed_dict = inp_dict)
        return ls
      w_summary = tf.histogram_summary('w', w)
      a_summary = tf.histogram_summary('a', a)
      b_summary = tf.histogram_summary('b', b)

    if FLAGS.model_id == 'relu_proximal':
      # lnl model with regularization, with proximal updates
      w = tf.Variable(np.array(w_load, dtype='float32'))
      a = tf.Variable(np.array(a_load, dtype='float32'))
      lam = tf.matmul(tf.nn.relu(tf.matmul(stim, w)), a) + 0.0001
      loss_inter = (tf.reduce_sum(lam/tot_spks)/120. - tf.reduce_sum(resp*tf.log(lam)/tot_spks)) / data_len
      loss = loss_inter + FLAGS.lam_w*tf.reduce_sum(tf.abs(w)) + FLAGS.lam_a*tf.reduce_sum(tf.abs(a))
      # training steps for a.
      train_step_a = tf.train.AdagradOptimizer(FLAGS.eta_a).minimize(loss_inter, var_list=[a])
      # as 'a' is positive, this is op soft-thresholding for L1 and projecting to feasible set
      soft_th_a = tf.assign(a, tf.nn.relu(a - FLAGS.eta_a * FLAGS.lam_a))

      # training steps for w
      train_step_w = tf.train.AdagradOptimizer(FLAGS.eta_w).minimize(loss_inter, var_list=[w])
      # do soft thresholding for 'w'
      soft_th_w = tf.assign(w, tf.nn.relu(w - FLAGS.eta_w * FLAGS.lam_w) - tf.nn.relu(- w - FLAGS.eta_w * FLAGS.lam_w))
      def training(inp_dict):
        # gradient step for 'w'
        sess.run(train_step_w, feed_dict=inp_dict)
        # soft thresholding for w
        sess.run(soft_th_w, feed_dict=inp_dict)
        # gradient step for 'a'
        sess.run(train_step_a, feed_dict=inp_dict)
        # soft thresholding for a, and project in constraint set
        sess.run(soft_th_a, feed_dict=inp_dict)

      def get_loss(inp_dict):
        ls = sess.run(loss,feed_dict = inp_dict)
        return ls

    if FLAGS.model_id == 'relu_eg':
      a_load = a_load / np.sum(a_load, axis=0)# normalize initial a
      w = tf.Variable(np.array(w_load, dtype='float32'))
      a = tf.Variable(np.array(a_load, dtype='float32'))
      lam = tf.matmul(tf.nn.relu(tf.matmul(stim, w)), a) + 0.0001
      loss_inter = (tf.reduce_sum(lam/tot_spks)/120. - tf.reduce_sum(resp*tf.log(lam)/tot_spks)) / data_len
      loss = loss_inter + FLAGS.lam_w*tf.reduce_sum(tf.abs(w))
      # steps to update a
      # as 'a' is positive, this is op soft-thresholding for L1 and projecting to feasible set
      eta_a_tf = tf.constant(np.squeeze(FLAGS.eta_a),dtype='float32')
      grads_a = tf.gradients(loss_inter, a)
      exp_grad_a = tf.squeeze(tf.mul(a,tf.exp(-eta_a_tf * grads_a)))
      a_update = tf.assign(a,exp_grad_a/tf.reduce_sum(exp_grad_a,0))
      # steps to update w
      # gradient update of 'w'..
      train_step_w = tf.train.AdagradOptimizer(FLAGS.eta_w).minimize(loss_inter, var_list=[w])
      # do soft thresholding for 'w'
      soft_th_w = tf.assign(w, tf.nn.relu(w - FLAGS.eta_w * FLAGS.lam_w) - tf.nn.relu(- w - FLAGS.eta_w * FLAGS.lam_w))
      def training(inp_dict):
        # gradient step for 'a' and 'w'
        sess.run(train_step_w, feed_dict=inp_dict)
        # soft thresholding for w
        sess.run(soft_th_w)
        # update a
        sess.run(a_update, feed_dict=inp_dict)
      print('eg training made')
      def get_loss(inp_dict):
        ls = sess.run(loss,feed_dict = inp_dict)
        return ls

    if FLAGS.model_id == 'relu_window':
      # convolution weights, each layer is delta(x,y) - basically take window of stimulus.
      window = FLAGS.window
      n_pix = (2* window + 1) ** 2
      w_mask = np.zeros((2 * window + 1, 2 * window + 1, 1, n_pix))
      icnt = 0
      for ix in range(2 * window + 1):
        for iy in range(2 * window + 1):
          w_mask[ix, iy, 0, icnt] =1
          icnt = icnt + 1
      mask_tf = tf.constant(np.array(w_mask, dtype='float32'))

      # set weight and other variables
      dimx = np.floor(1 + ((40 - (2 * window + 1))/FLAGS.stride)).astype('int')
      dimy = np.floor(1 + ((80 - (2 * window + 1))/FLAGS.stride)).astype('int')

      w = tf.Variable(np.array(0.1+ 0.05*np.random.rand(dimx, dimy, n_pix),dtype='float32')) # exp 5
      #w = tf.Variable(np.array(np.random.randn(dimx, dimy, n_pix),dtype='float32')) # exp 4
      a = tf.Variable(np.array(np.random.rand(dimx*dimy, nCells),dtype='float32'))
      a_pos = tf.assign(a, (a + tf.abs(a))/2)
      # get firing rate
      stim4D = tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
      stim_masked = tf.nn.conv2d(stim4D, mask_tf, strides=[1, FLAGS.stride, FLAGS.stride, 1], padding="VALID" )
      stim_wts = tf.nn.relu(tf.reduce_sum(tf.mul(stim_masked, w), 3))

      lam = tf.matmul(tf.reshape(stim_wts, [-1,dimx*dimy]),a) + 0.00001
      loss_inter = (tf.reduce_sum(lam)/120. - tf.reduce_sum(resp*tf.log(lam)))/data_len
      loss = loss_inter +  FLAGS.lam_w * tf.reduce_sum(tf.nn.l2_loss(w))
      train_step = tf.train.AdagradOptimizer(FLAGS.step_sz).minimize(loss,var_list=[w,a])
      def training(inp_dict):
        sess.run(train_step, feed_dict=inp_dict)
        sess.run(a_pos)
      def get_loss(inp_dict):
        ls = sess.run(loss, feed_dict=inp_dict)
        return ls
      w_summary = tf.histogram_summary('w', w)
      a_summary = tf.histogram_summary('a', a)

    if FLAGS.model_id == 'relu_window_mother':
      # convolution weights, each layer is delta(x,y) - basically take window of stimulus.
      window = FLAGS.window
      n_pix = (2* window + 1) ** 2
      w_mask = np.zeros((2 * window + 1, 2 * window + 1, 1, n_pix))
      icnt = 0
      for ix in range(2 * window + 1):
        for iy in range(2 * window + 1):
          w_mask[ix, iy, 0, icnt] =1
          icnt = icnt + 1
      mask_tf = tf.constant(np.array(w_mask, dtype='float32'))

      # set weight and other variables
      dimx = np.floor(1 + ((40 - (2 * window + 1))/FLAGS.stride)).astype('int')
      dimy = np.floor(1 + ((80 - (2 * window + 1))/FLAGS.stride)).astype('int')

      w_del = tf.Variable(np.array(0.1+ 0.05*np.random.rand(dimx, dimy, n_pix),dtype='float32'))
      w_mother = tf.Variable(np.array(np.ones((2 * window + 1, 2 * window + 1, 1, 1)),dtype='float32'))
      a = tf.Variable(np.array(np.random.rand(dimx*dimy, nCells),dtype='float32'))
      a_pos = tf.assign(a, (a + tf.abs(a))/2)
      # get firing rate
      stim4D = tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
      # mother weight convolution
      stim_convolved = tf.reduce_sum( tf.nn.conv2d(stim4D, w_mother, strides=[1, FLAGS.stride, FLAGS.stride, 1], padding="VALID"),3)
      #
      stim_masked = tf.nn.conv2d(stim4D, mask_tf, strides=[1, FLAGS.stride, FLAGS.stride, 1], padding="VALID" )
      stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)
      su_act = tf.nn.relu(stim_del + stim_convolved)

      lam = tf.matmul(tf.reshape(su_act, [-1, dimx*dimy]),a) + 0.00001
      loss_inter = (tf.reduce_sum(lam)/120. - tf.reduce_sum(resp*tf.log(lam)))/data_len
      loss = loss_inter +   FLAGS.lam_w * tf.reduce_sum(tf.nn.l2_loss(w_del))
      train_step = tf.train.AdagradOptimizer(FLAGS.step_sz).minimize(loss,var_list=[w_mother, w_del, a])
      def training(inp_dict):
        sess.run(train_step, feed_dict=inp_dict)
        sess.run(a_pos)
      def get_loss(inp_dict):
        ls = sess.run(loss, feed_dict=inp_dict)
        return ls
      w_del_summary = tf.histogram_summary('w_del', w_del)
      w_mother_summary = tf.histogram_summary('w_mother', w_mother)
      a_summary = tf.histogram_summary('a', a)

    if FLAGS.model_id == 'relu_window_mother_sfm':
      # softmax weights used!
      # convolution weights, each layer is delta(x,y) - basically take window of stimulus.
      window = FLAGS.window
      n_pix = (2* window + 1) ** 2
      w_mask = np.zeros((2 * window + 1, 2 * window + 1, 1, n_pix))
      icnt = 0
      for ix in range(2 * window + 1):
        for iy in range(2 * window + 1):
          w_mask[ix, iy, 0, icnt] =1
          icnt = icnt + 1
      mask_tf = tf.constant(np.array(w_mask, dtype='float32'))

      # set weight and other variables
      dimx = np.floor(1 + ((40 - (2 * window + 1))/FLAGS.stride)).astype('int')
      dimy = np.floor(1 + ((80 - (2 * window + 1))/FLAGS.stride)).astype('int')

      w_del = tf.Variable(np.array(0.1 + 0.05*np.random.randn(dimx, dimy, n_pix),dtype='float32'))
      w_mother = tf.Variable(np.array(np.ones((2 * window + 1, 2 * window + 1, 1, 1)),dtype='float32'))
      a = tf.Variable(np.array(np.random.randn(dimx*dimy, nCells),dtype='float32'))
      b = tf.transpose(tf.nn.softmax(tf.transpose(a)))

      # get firing rate
      stim4D = tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
      # mother weight convolution
      stim_convolved = tf.reduce_sum( tf.nn.conv2d(stim4D, w_mother, strides=[1, FLAGS.stride, FLAGS.stride, 1], padding="VALID"),3)
      #
      stim_masked = tf.nn.conv2d(stim4D, mask_tf, strides=[1, FLAGS.stride, FLAGS.stride, 1], padding="VALID" )
      stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)
      su_act = tf.nn.relu(stim_del + stim_convolved)

      lam = tf.matmul(tf.reshape(su_act, [-1, dimx*dimy]), b) + 0.00001
      loss_inter = (tf.reduce_sum(lam)/120. - tf.reduce_sum(resp*tf.log(lam)))/data_len
      loss = loss_inter +  FLAGS.lam_w * tf.reduce_sum(tf.nn.l2_loss(w_del))
      train_step = tf.train.AdagradOptimizer(FLAGS.step_sz).minimize(loss,var_list=[w_mother, w_del, a])
      def training(inp_dict):
        sess.run(train_step, feed_dict=inp_dict)
      def get_loss(inp_dict):
        ls = sess.run(loss, feed_dict=inp_dict)
        return ls
      w_del_summary = tf.histogram_summary('w_del', w_del)
      w_mother_summary = tf.histogram_summary('w_mother', w_mother)
      a_summary = tf.histogram_summary('a', a)


    if FLAGS.model_id == 'relu_window_mother_sfm_exp':
      # softmax weights used!
      # convolution weights, each layer is delta(x,y) - basically take window of stimulus.
      window = FLAGS.window
      n_pix = (2* window + 1) ** 2
      w_mask = np.zeros((2 * window + 1, 2 * window + 1, 1, n_pix))
      icnt = 0
      for ix in range(2 * window + 1):
        for iy in range(2 * window + 1):
          w_mask[ix, iy, 0, icnt] =1
          icnt = icnt + 1
      mask_tf = tf.constant(np.array(w_mask, dtype='float32'))

      # set weight and other variables
      dimx = np.floor(1 + ((40 - (2 * window + 1))/FLAGS.stride)).astype('int')
      dimy = np.floor(1 + ((80 - (2 * window + 1))/FLAGS.stride)).astype('int')

      w_del = tf.Variable(np.array( 0.05*np.random.randn(dimx, dimy, n_pix),dtype='float32'))
      w_mother = tf.Variable(np.array(np.ones((2 * window + 1, 2 * window + 1, 1, 1)),dtype='float32'))
      a = tf.Variable(np.array(np.random.randn(dimx*dimy, nCells),dtype='float32'))
      b = tf.transpose(tf.nn.softmax(tf.transpose(a)))
      # get firing rate
      stim4D = tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
      # mother weight convolution
      stim_convolved = tf.reduce_sum( tf.nn.conv2d(stim4D, w_mother, strides=[1, FLAGS.stride, FLAGS.stride, 1], padding="VALID"),3)
      #
      stim_masked = tf.nn.conv2d(stim4D, mask_tf, strides=[1, FLAGS.stride, FLAGS.stride, 1], padding="VALID" )
      stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)
      su_act = tf.nn.relu(stim_del + stim_convolved)

      lam = tf.exp(tf.matmul(tf.reshape(su_act, [-1, dimx*dimy]), b)) + 0.00001
      loss_inter = (tf.reduce_sum(lam)/120. - tf.reduce_sum(resp*tf.log(lam)))/data_len
      loss = loss_inter +  FLAGS.lam_w * tf.reduce_sum(tf.nn.l2_loss(w_del))

      # version 0
      train_step = tf.train.AdagradOptimizer(FLAGS.step_sz).minimize(loss,var_list=[w_mother, w_del, a])

      # version 1
      '''
      optimizer = tf.train.AdagradOptimizer(..)
      grad_and_vars = optimizer.compute_gradientS(...)
      manipulated_grads_and_vars = []
      clip = 1.0
      for g, v in grad_and_vars:
        if g is not None:
          tf.histogram_summary(g.name + "/histogram", g)
          with tf.get_default_graph().colocate_with(g):
            clipped_g, _ = tf.clip_by_global_norm([g], clip)
        else:
          clipped_g = g
        maniuplated_grads_and_vars.append([clipped_g, v])
      train_step = optimizer.apply_gradients(maniuplated_grads_and_vars, global_step)
      '''

      # optimizer = tf.train.AdagradOptimizer(..)
      # train_step = optimizer.mminimize(..)
      #  -- or --
      # def minimize(self, ..):
      #  grad_and_vars = self.compute_gradients(loss, variables)  # returning a list of tuple, not an Op
      #  train_step = self.apply_gradients(grads_and_vars, global_step)  # returns an Op
      #  return train_step
      #
      # grad_and_vars = optimizer.compute_gradientS(loss, variables)
      # grad_and_vars <-- list([gradient0, variable0], [gradient1, variable1], ....]
      # for g, v in grad_and_vars:
      #   # manipulate g
      # train_step = optimizer.apply_gradients(maniuplated_grads_and_vars, global_step)


      def training(inp_dict):
        sess.run(train_step, feed_dict=inp_dict)
      def get_loss(inp_dict):
        ls = sess.run(loss, feed_dict=inp_dict)
        return ls
      w_del_summary = tf.histogram_summary('w_del', w_del)
      w_mother_summary = tf.histogram_summary('w_mother', w_mother)
      a_summary = tf.histogram_summary('a', a)


    if FLAGS.model_id == 'relu_window_exp':
      # convolution weights, each layer is delta(x,y) - basically take window of stimulus.
      window = FLAGS.window
      n_pix = (2* window + 1) ** 2
      w_mask = np.zeros((2 * window + 1, 2 * window + 1, 1, n_pix))
      icnt = 0
      for ix in range(2 * window + 1):
        for iy in range(2 * window + 1):
          w_mask[ix, iy, 0, icnt] =1
          icnt = icnt + 1
      mask_tf = tf.constant(np.array(w_mask, dtype='float32'))

      # set weight and other variables
      dimx = np.floor(1 + ((40 - (2 * window + 1))/FLAGS.stride)).astype('int')
      dimy = np.floor(1 + ((80 - (2 * window + 1))/FLAGS.stride)).astype('int')

      w = tf.Variable(np.array(0.01+ 0.005*np.random.rand(dimx, dimy, n_pix),dtype='float32'))
      a = tf.Variable(np.array(0.02+np.random.rand(dimx*dimy, nCells),dtype='float32'))

      # get firing rate
      stim4D = tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
      stim_masked = tf.nn.conv2d(stim4D, mask_tf, strides=[1, FLAGS.stride, FLAGS.stride, 1], padding="VALID" )
      stim_wts = tf.nn.relu(tf.reduce_sum(tf.mul(stim_masked, w), 3))
      a_pos = tf.assign(a, (a + tf.abs(a))/2)

      lam = tf.exp(tf.matmul(tf.reshape(stim_wts, [-1,dimx*dimy]),a))
      loss_inter = (tf.reduce_sum(lam)/120. - tf.reduce_sum(resp*tf.log(lam)))/data_len
      loss = loss_inter +  FLAGS.lam_w * tf.reduce_sum(tf.nn.l2_loss(w))
      train_step = tf.train.AdagradOptimizer(FLAGS.step_sz).minimize(loss,var_list=[w,a])
      def training(inp_dict):
        sess.run(train_step, feed_dict=inp_dict)
        sess.run(a_pos)
      def get_loss(inp_dict):
        ls = sess.run(loss, feed_dict=inp_dict)
        return ls
      w_summary = tf.histogram_summary('w', w)
      a_summary = tf.histogram_summary('a', a)


    if FLAGS.model_id == 'relu_window_mother_exp':
      # convolution weights, each layer is delta(x,y) - basically take window of stimulus.
      window = FLAGS.window
      n_pix = (2* window + 1) ** 2
      w_mask = np.zeros((2 * window + 1, 2 * window + 1, 1, n_pix))
      icnt = 0
      for ix in range(2 * window + 1):
        for iy in range(2 * window + 1):
          w_mask[ix, iy, 0, icnt] =1
          icnt = icnt + 1
      mask_tf = tf.constant(np.array(w_mask, dtype='float32'))

      # set weight and other variables
      dimx = np.floor(1 + ((40 - (2 * window + 1))/FLAGS.stride)).astype('int')
      dimy = np.floor(1 + ((80 - (2 * window + 1))/FLAGS.stride)).astype('int')

      w_del = tf.Variable(np.array(0.005*np.random.randn(dimx, dimy, n_pix),dtype='float32'))
      w_mother = tf.Variable(np.array(0.01*np.ones((2 * window + 1, 2 * window + 1, 1, 1)),dtype='float32'))
      a = tf.Variable(0.02+np.array(np.random.rand(dimx*dimy, nCells),dtype='float32'))

      # get firing rate
      stim4D = tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
      # mother weight convolution
      stim_convolved = tf.reduce_sum( tf.nn.conv2d(stim4D, w_mother, strides=[1, FLAGS.stride, FLAGS.stride, 1], padding="VALID"),3)
      #
      a_pos = tf.assign(a, (a + tf.abs(a))/2)

      stim_masked = tf.nn.conv2d(stim4D, mask_tf, strides=[1, FLAGS.stride, FLAGS.stride, 1], padding="VALID" )
      stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)
      su_act = tf.nn.relu(stim_del + stim_convolved)

      lam = tf.exp(tf.matmul(tf.reshape(su_act, [-1, dimx*dimy]),a))
      loss_inter = (tf.reduce_sum(lam)/120. - tf.reduce_sum(resp*tf.log(lam)))/data_len
      loss = loss_inter +   FLAGS.lam_w * tf.reduce_sum(tf.nn.l2_loss(w_del))
      train_step = tf.train.AdagradOptimizer(FLAGS.step_sz).minimize(loss,var_list=[w_mother, w_del, a])
      def training(inp_dict):
        sess.run(train_step, feed_dict=inp_dict)
        sess.run(a_pos)
      def get_loss(inp_dict):
        ls = sess.run(loss, feed_dict=inp_dict)
        return ls
      w_del_summary = tf.histogram_summary('w_del', w_del)
      w_mother_summary = tf.histogram_summary('w_mother', w_mother)
      a_summary = tf.histogram_summary('a', a)


    if FLAGS.model_id == 'relu_window_a_support':
      # convolution weights, each layer is delta(x,y) - basically take window of stimulus.
      window = FLAGS.window
      n_pix = (2* window + 1) ** 2
      w_mask = np.zeros((2 * window + 1, 2 * window + 1, 1, n_pix))
      icnt = 0
      for ix in range(2 * window + 1):
        for iy in range(2 * window + 1):
          w_mask[ix, iy, 0, icnt] =1
          icnt = icnt + 1
      mask_tf = tf.constant(np.array(w_mask, dtype='float32'))

      # set weight and other variables
      dimx = np.floor(1 + ((40 - (2 * window + 1))/FLAGS.stride)).astype('int')
      dimy = np.floor(1 + ((80 - (2 * window + 1))/FLAGS.stride)).astype('int')

      w = tf.Variable(np.array(0.001+ 0.0005*np.random.rand(dimx, dimy, n_pix),dtype='float32'))
      a = tf.Variable(np.array(0.002*np.random.rand(dimx*dimy, nCells),dtype='float32'))
      stim4D = tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
      stim_masked = tf.nn.conv2d(stim4D, mask_tf, strides=[1, FLAGS.stride, FLAGS.stride, 1], padding="VALID" )
      stim_wts = tf.nn.relu(tf.reduce_sum(tf.mul(stim_masked, w), 3))
      lam = tf.matmul(tf.reshape(stim_wts, [-1,dimx*dimy]),a)+0.0001
      loss_inter = (tf.reduce_sum(lam)/120. - tf.reduce_sum(resp*tf.log(lam)))/data_len
      loss = loss_inter +  FLAGS.lam_w * tf.reduce_sum(tf.nn.l2_loss(w))
      a_pos = tf.assign(a, (a + tf.abs(a))/2)

      # mask a to only relevant pixels
      w_mother = tf.Variable(np.array(0.01*np.ones((2 * window + 1, 2 * window + 1, 1, 1)),dtype='float32'))
      # mother weight convolution
      stim_convolved = tf.reduce_sum( tf.nn.conv2d(stim4D, w_mother, strides=[1, FLAGS.stride, FLAGS.stride, 1], padding="VALID"),3)
      sess.run(tf.initialize_all_variables())

      mask_conv = sess.run(stim_convolved,feed_dict = {stim: total_mask})
      mask_a_flat = np.array(np.reshape(mask_conv, [-1,dimx * dimy]).T >0, dtype='float32')
      a_proj = tf.assign(a, a * mask_a_flat)
 
      train_step = tf.train.AdagradOptimizer(FLAGS.step_sz).minimize(loss,var_list=[w,a])
      def training(inp_dict):
        sess.run([train_step, a_proj], feed_dict=inp_dict)
        sess.run(a_pos)
      def get_loss(inp_dict):
        ls = sess.run(loss, feed_dict=inp_dict)
        return ls
      w_summary = tf.histogram_summary('w', w)
      a_summary = tf.histogram_summary('a', a)


    if FLAGS.model_id == 'exp_window_a_support':
      # convolution weights, each layer is delta(x,y) - basically take window of stimulus.
      window = FLAGS.window
      n_pix = (2* window + 1) ** 2
      w_mask = np.zeros((2 * window + 1, 2 * window + 1, 1, n_pix))
      icnt = 0
      for ix in range(2 * window + 1):
        for iy in range(2 * window + 1):
          w_mask[ix, iy, 0, icnt] =1
          icnt = icnt + 1
      mask_tf = tf.constant(np.array(w_mask, dtype='float32'))

      # set weight and other variables
      dimx = np.floor(1 + ((40 - (2 * window + 1))/FLAGS.stride)).astype('int')
      dimy = np.floor(1 + ((80 - (2 * window + 1))/FLAGS.stride)).astype('int')

      w = tf.Variable(np.array(0.001+ 0.0005*np.random.rand(dimx, dimy, n_pix),dtype='float32'))
      a = tf.Variable(np.array(0.002*np.random.rand(dimx*dimy, nCells),dtype='float32'))
      stim4D = tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
      stim_masked = tf.nn.conv2d(stim4D, mask_tf, strides=[1, FLAGS.stride, FLAGS.stride, 1], padding="VALID" )
      stim_wts = tf.exp(tf.reduce_sum(tf.mul(stim_masked, w), 3))
      lam = tf.matmul(tf.reshape(stim_wts, [-1,dimx*dimy]),a)
      loss_inter = (tf.reduce_sum(lam)/120. - tf.reduce_sum(resp*tf.log(lam)))/data_len
      loss = loss_inter +  FLAGS.lam_w * tf.reduce_sum(tf.nn.l2_loss(w))
      a_pos = tf.assign(a, (a + tf.abs(a))/2)

      # mask a to only relevant pixels
      w_mother = tf.Variable(np.array(0.01*np.ones((2 * window + 1, 2 * window + 1, 1, 1)),dtype='float32'))
      # mother weight convolution
      stim_convolved = tf.reduce_sum( tf.nn.conv2d(stim4D, w_mother, strides=[1, FLAGS.stride, FLAGS.stride, 1], padding="VALID"),3)
      sess.run(tf.initialize_all_variables())

      mask_conv = sess.run(stim_convolved,feed_dict = {stim: total_mask})
      mask_a_flat = np.array(np.reshape(mask_conv, [-1,dimx * dimy]).T >0, dtype='float32')
      a_proj = tf.assign(a, a * mask_a_flat)

      train_step = tf.train.AdagradOptimizer(FLAGS.step_sz).minimize(loss,var_list=[w,a])
      def training(inp_dict):
        sess.run([train_step, a_proj], feed_dict=inp_dict)
        sess.run(a_pos)
      def get_loss(inp_dict):
        ls = sess.run(loss, feed_dict=inp_dict)
        return ls
      w_summary = tf.histogram_summary('w', w)
      a_summary = tf.histogram_summary('a', a)

    # initialize the model
    # make summary writers

    #logfile = gfile.Open(save_filename + '.txt', "a")

    # make summary writers

    l_summary = tf.scalar_summary('loss',loss)
    l_inter_summary = tf.scalar_summary('loss_inter',loss_inter)
    #tf.image_summary('a_image',tf.expand_dims(tf.expand_dims(tf.squeeze(tf.nn.relu(a)),0),-1))
    #tf.image_summary('w_image',tf.expand_dims(tf.transpose(tf.reshape(w, [40, 80, Nsub]), [2, 0, 1]), -1), max_images=50)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.save_location + 'train',
                                      sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.save_location + 'test')

    print('\nStarting new code')
    print('\nModel:' + FLAGS.model_id)

    sess.run(tf.initialize_all_variables())
    saver_var = tf.train.Saver(tf.all_variables(), keep_checkpoint_every_n_hours=0.05)
    load_prev = False
    start_iter=0
    try:
      latest_filename = short_filename + '_latest_fn'
      restore_file = tf.train.latest_checkpoint(FLAGS.save_location, latest_filename)
      start_iter = int(restore_file.split('/')[-1].split('-')[-1])
      saver_var.restore(sess, restore_file)
      load_prev = True
    except:
      print('No previous dataset')

    if load_prev:
      #logfile.write('\nPrevious results loaded')
      print('\nPrevious results loaded')
    else:
      #logfile.write('\nVariables initialized')
      print('\nVariables initialized')
    #logfile.flush()
    # Do the fitting

    icnt = 0
    stim_test,resp_test,test_length = get_test_data()
    fd_test = {stim: stim_test,
               resp: resp_test,
               data_len: test_length}

    #logfile.close()

    for istep in np.arange(start_iter,400000):


      # get training data
      stim_train, resp_train, train_len = get_next_training_batch(istep)
      fd_train = {stim: stim_train,
                  resp: resp_train,
                  data_len: train_len}
      # take training step
      training(fd_train)


      if istep%10 == 0:
        # compute training and testing losses
        ls_train = get_loss(fd_train)
        ls_test = get_loss(fd_test)
        ls_train_log = np.append(ls_train_log, ls_train)
        ls_test_log = np.append(ls_test_log, ls_test)
        latest_filename = short_filename + '_latest_fn'
        saver_var.save(sess, save_filename, global_step=istep, latest_filename = latest_filename)

        # add training summary
        summary = sess.run(merged, feed_dict=fd_train)
        train_writer.add_summary(summary,istep)
        # add testing summary
        summary = sess.run(merged, feed_dict=fd_test)
        test_writer.add_summary(summary,istep)
        

        # log results
        #logfile = gfile.Open(save_filename + '.txt', "a")
        #logfile.write('\nIterations: ' + str(istep) + ' Training error: '
        #              + str(ls_train) + ' Testing error: ' + str(ls_test) +
        #              '  w_l1_norm: ' + str(np.sum(np.abs(w.eval()))) +
        #              ' a_l1_norm: ' + str(np.sum(np.abs(a.eval()))))
        #logfile.close()
        #logfile.flush()



      icnt += FLAGS.batchsz
      if icnt > 216000-1000:
        icnt = 0
        tms = np.random.permutation(np.arange(216000-1000))

 #   write_filename = save_filename + '.pkl'
 #   write_file = gfile.Open(write_filename, 'wb')
 #   save_data = {'w': w.eval(), 'a': a.eval(), 'w_init': w_init,
 #                         'a_init': a_init, 'w_load': w_load, 'a_load': a_load,
 #                         'ls_train_log': ls_train_log,
 #                         'ls_test_log': ls_test_log, 'last_iter': istep, 'chunk_order': chunk_order}


 #   pickle.dump(save_data,write_file)
 #   write_file.close()

  #logfile.close()
if __name__ == '__main__':
  app.run()

