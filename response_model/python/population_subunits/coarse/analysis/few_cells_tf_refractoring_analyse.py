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
""" Analyse fitted subunits for multiple cells simultaneously.
This script has the extensions of single cell models from earlier
as well as new population subunit models -
most notably the almost convolutional model - where each subunit is
summation of mother subunit and subunit specific modification.

"""

import sys
import os.path
import collections
import tensorflow as tf
from absl import app
from absl import flags
from absl import gfile
import cPickle as pickle
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import ndimage
import random


FLAGS = flags.FLAGS
# flags for data locations
flags.DEFINE_string('folder_name', 'experiment4', 'folder where to store all the data')
flags.DEFINE_string('save_location',
                    '/home/bhaishahster/',
                    'where to store logs and outputs?');
flags.DEFINE_string('data_location',
                    '/home/bhaishahster/data_breakdown/',
                    'where to take data from?')

# flags for stochastic learning
flags.DEFINE_integer('batchsz', 1000, 'batch size for training')
flags.DEFINE_integer('n_chunks', 216, 'number of data chunks') # should be 216
flags.DEFINE_integer('n_b_in_c', 1, 'number of batches in one chunk of data')
flags.DEFINE_integer('train_len', 216 - 21, 'how much training length to use?')
flags.DEFINE_float('step_sz', 10, 'step size for learning algorithm')

# random number generators initialized
# removes unneccessary data variabilities while comparing algorithms
flags.DEFINE_integer('np_randseed', 23, 'numpy RNG seed')
flags.DEFINE_integer('randseed', 65, 'python RNG seed')


# flags for model/loss specification
flags.DEFINE_string('model_id', 'relu', 'which model to fit')
flags.DEFINE_string('loss', 'poisson', 'which loss to use?')
flags.DEFINE_string('masked_stimulus', 'False',
                    'use all pixels or only those inside RF of selected cells?')
flags.DEFINE_string('all_cells', 'True',
                    'learn model for all cells or a few chosen ones?')


# model specific terms
# subunit grid spacing
flags.DEFINE_float('su_grid_spacing', 3, 'grid spacing')
# useful for models which take a specific number of subunits as input
flags.DEFINE_integer('ratio_SU', 2, 'ratio of subunits/cells')
# useful for convolution-like models
flags.DEFINE_integer('window', 3,
                     'size of window for each subunit in relu_window model')
flags.DEFINE_integer('stride', 3, 'stride for relu_window')
# some models need regularization of parameters
flags.DEFINE_float('lam_w', 0.0001, 'sparsitiy regularization of w')
flags.DEFINE_float('lam_a', 0.0001, 'sparsitiy regularization of a')
FLAGS = flags.FLAGS

model = collections.namedtuple("model",
                                     ["stim", "su_act","lam"])

model_lnp = collections.namedtuple("model_lnp",
                                     ["w","b", "mask", "loss_log_train"])

# global stimulus variables
stim_train_part = np.array([])
resp_train_part = np.array([])
chunk_order = np.array([])
cells_choose = np.array([])
chosen_mask = np.array([])



def get_test_data():
  # get test data from the last chunk of the data
  test_data_chunks = [FLAGS.n_chunks]
  for ichunk in test_data_chunks:
    filename = FLAGS.data_location + 'Off_par_data_' + str(ichunk) + '.mat'
    file_r = gfile.Open(filename, 'r')
    data = sio.loadmat(file_r)
    stim_part = data['maskedMovdd_part'].T
    resp_part = data['Y_part'].T
    test_len = stim_part.shape[0]
  stim_part = stim_part[:, chosen_mask]
  resp_part = resp_part[:, cells_choose]
  return stim_part, resp_part, test_len



def get_next_training_batch(iteration):
  # Returns a new batch of training data : stimulus and response arrays
  # we will use global stimulus and response variables to permute training data
  # chunks and store where we are in list of training data

  # each chunk might have multiple training batches.
  # So go through all batches in a 'chunk' before moving on to the next chunk
  global stim_train_part
  global resp_train_part
  global chunk_order

  togo = True
  while togo:
    if(iteration % FLAGS.n_b_in_c == 0):
    # iteration is multiple of number of batches in a chunk means
    # finished going through a chunk, load new chunk of data
      ichunk = (iteration / FLAGS.n_b_in_c) % (FLAGS.train_len-1 ) # -1 as last one chunk used for testing
      if (ichunk == 0):
        # if starting over the chunks again, shuffle the chunks
        chunk_order = np.random.permutation(np.arange(FLAGS.train_len)) # remove first chunk - weired?
      if chunk_order[ichunk] + 1 != 1: # 1st chunk was weired for the dataset used
        filename = FLAGS.data_location + 'Off_par_data_' + str(chunk_order[ichunk] + 1) + '.mat'
        file_r = gfile.Open(filename, 'r')
        data = sio.loadmat(file_r)
        stim_train_part = data['maskedMovdd_part'] # stimulus
        resp_train_part = data['Y_part'] # response

        ichunk = chunk_order[ichunk] + 1
        while stim_train_part.shape[1] < FLAGS.batchsz:
          # if the current loaded data is smaller than batch size, load more chunks
          if (ichunk > FLAGS.n_chunks):
            ichunk = 2
          filename = FLAGS.data_location + 'Off_par_data_' + str(ichunk) + '.mat'
          file_r = gfile.Open(filename, 'r')
          data = sio.loadmat(file_r)
          stim_train_part = np.append(stim_train_part, data['maskedMovdd_part'],
                                      axis=1)
          resp_train_part = np.append(resp_train_part, data['Y_part'], axis=1)
          ichunk = ichunk + 1

    ibatch = iteration % FLAGS.n_b_in_c # which section of current chunk to use
    try:
      stim_train = np.array(stim_train_part[:,ibatch: ibatch + FLAGS.batchsz],
                            dtype='float32').T
      resp_train = np.array(resp_train_part[:,ibatch: ibatch + FLAGS.batchsz],
                            dtype='float32').T
      togo=False
    except:
      iteration = np.random.randint(1,100000)
      print('Load exception iteration: ' + str(iteration) +
            'chunk: ' + str(chunk_order[ichunk]) + 'batch: ' + str(ibatch) )
      togo=True

  stim_train = stim_train[:, chosen_mask]
  resp_train = resp_train[:, cells_choose]
  return stim_train, resp_train, FLAGS.batchsz



def get_windows():
  # use FLAGS to get convolutional 'windows' for convolutional models.
  window = FLAGS.window
  n_pix = (2* window + 1) ** 2 # number of pixels in the window
  w_mask = np.zeros((2 * window + 1, 2 * window + 1, 1, n_pix))
  icnt = 0

  # make mask_tf: weight (dimx X dimy X npix) for convolutional layer,
  # where each layer is 1 for a particular pixel in window and 0 for others.
  # this is used for flattening the pixels in a window,
  # so that different weights could be applied to each window
  for ix in range(2 * window + 1):
    for iy in range(2 * window + 1):
      w_mask[ix, iy, 0, icnt] =1
      icnt = icnt + 1
  mask_tf = tf.constant(np.array(w_mask, dtype='float32'))

  # number of windows in x and y dimensions
  dimx = np.floor(1 + ((40 - (2 * window + 1))/FLAGS.stride)).astype('int')
  dimy = np.floor(1 + ((80 - (2 * window + 1))/FLAGS.stride)).astype('int')
  return mask_tf, dimx, dimy, n_pix



def main(argv):

  # global variables will be used for getting training data
  global cells_choose
  global chosen_mask
  global chunk_order

  # set random seeds
  np.random.seed(FLAGS.np_randseed)
  random.seed(FLAGS.randseed)
  chunk_order = np.random.permutation(np.arange(FLAGS.n_chunks-1))

  # Load data summary
  data_filename = FLAGS.data_location + 'data_details.mat'
  summary_file = gfile.Open(data_filename, 'r')
  data_summary = sio.loadmat(summary_file)
  cells = np.squeeze(data_summary['cells'])

  # which cells to train subunits for
  if FLAGS.all_cells == 'True':
    cells_choose = np.array(np.ones(np.shape(cells)), dtype='bool')
  else:
    cells_choose = (cells ==3287) | (cells ==3318 ) | (cells ==3155) | (cells ==3066)
  n_cells = np.sum(cells_choose) # number of cells

  # load spikes and relevant stimulus pixels for chosen cells
  tot_spks = np.squeeze(data_summary['tot_spks'])
  tot_spks_chosen_cells = np.array(tot_spks[cells_choose] ,dtype='float32')
  total_mask = np.squeeze(data_summary['totalMaskAccept_log']).T
  # chosen_mask = which pixels to learn subunits over
  if FLAGS.masked_stimulus == 'True':
    chosen_mask = np.array(np.sum(total_mask[cells_choose,:],0)>0, dtype='bool')
  else:
    chosen_mask = np.array(np.ones(3200).astype('bool'))
  stim_dim = np.sum(chosen_mask) # stimulus dimensions
  stas = np.array(data_summary['stas'])
  print('\ndataset summary loaded')

  print('Save folder name: ' + str(FLAGS.folder_name) +
        '\nmodel:' + str(FLAGS.model_id) +
        '\nLoss:' + str(FLAGS.loss) +
        '\nmasked stimulus:' + str(FLAGS.masked_stimulus) +
        '\nall_cells?' + str(FLAGS.all_cells) +
        '\nbatch size' + str(FLAGS.batchsz) +
        '\nstep size' + str(FLAGS.step_sz) +
        '\ntraining length: ' + str(FLAGS.train_len) +
        '\nn_cells: '+str(n_cells))


  # decide the number of subunits to fit
  n_su = FLAGS.ratio_SU*n_cells

  # saving details
  short_filename = ('_masked_stim=' + str(FLAGS.masked_stimulus) + '_all_cells='+
                    str(FLAGS.all_cells) + '_loss='+
                    str(FLAGS.loss) + '_batch_sz='+ str(FLAGS.batchsz) +
                    '_step_sz'+ str(FLAGS.step_sz) +
                    '_tlen=' + str(FLAGS.train_len) + '_bg')


  with tf.Session() as sess:
    # set up stimulus and response palceholders
    stim = tf.placeholder(tf.float32, shape=[None, stim_dim], name='stim')
    resp = tf.placeholder(tf.float32, name='resp')
    data_len = tf.placeholder(tf.float32, name='data_len')

    if FLAGS.loss == 'poisson':
      b_init = np.array(0.000001*np.ones(n_cells))
    else:
      b_init =  np.log((tot_spks_chosen_cells)/(216000. - tot_spks_chosen_cells))

    # different firing rate models
    if FLAGS.model_id == 'exp_additive':
      # This model was implemented for earlier work.
      # firing rate for cell c: lam_c = sum_s exp(w_s.x + a_sc)

      # filename
      short_filename = ('model=' + str(FLAGS.model_id) + short_filename)
      # variables
      w = tf.Variable(np.array(0.01 * np.random.randn(stim_dim, n_su),
                               dtype='float32'), name='w')
      a = tf.Variable(np.array(0.01 * np.random.rand(n_cells, 1, n_su),
                               dtype='float32'), name='a')
      # firing rate model
      lam = tf.transpose(tf.reduce_sum(tf.exp(tf.matmul(stim, w) + a), 2))
      regularization = 0
      vars_fit = [w, a]
      def proj():
        pass

    if FLAGS.model_id == 'relu':
      # firing rate for cell c: lam_c = a_c'.relu(w.x) + b
      # we know a>0 and for poisson loss, b>0
      # for poisson loss: small b added to prevent lam_c going to 0

      # filename
      short_filename = ('model=' + str(FLAGS.model_id) +
                        '_lam_w=' + str(FLAGS.lam_w) + '_lam_a=' +
                        str(FLAGS.lam_a) + '_nsu=' + str(n_su) + short_filename)
      # variables
      w = tf.Variable(np.array(0.01 * np.random.randn(stim_dim, n_su),
                               dtype='float32'), name='w')
      a = tf.Variable(np.array(0.01 * np.random.rand(n_su, n_cells),
                               dtype='float32'), name='a')
      b = tf.Variable(np.array(b_init,dtype='float32'), name='b')
      # firing rate model
      lam = tf.matmul(tf.nn.relu(tf.matmul(stim, w)), a) + b
      vars_fit = [w, a]
      if not FLAGS.loss == 'poisson':
        vars_fit = vars_fit + [b]


      # regularization of parameters
      regularization = (FLAGS.lam_w * tf.reduce_sum(tf.abs(w)) +
                        FLAGS.lam_a * tf.reduce_sum(tf.abs(a)))
      # projection to satisfy constraints
      a_pos = tf.assign(a, (a + tf.abs(a))/2)
      b_pos = tf.assign(b, (b + tf.abs(b))/2)
      def proj():
        sess.run(a_pos)
        if FLAGS.loss == 'poisson':
          sess.run(b_pos)


    if FLAGS.model_id == 'relu_window':
      # firing rate for cell c: lam_c = a_c'.relu(w.x) + b,
      # where w_i are over a small window which are convolutionally related with each other.
      # we know a>0 and for poisson loss, b>0
      # for poisson loss: small b added to prevent lam_c going to 0

      # filename
      short_filename = ('model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) +
                        '_lam_w=' + str(FLAGS.lam_w) + short_filename )
      mask_tf, dimx, dimy, n_pix = get_windows()

      # variables
      w = tf.Variable(np.array(0.1 + 0.05*np.random.rand(dimx, dimy, n_pix),
                               dtype='float32'), name='w')
      a = tf.Variable(np.array(np.random.rand(dimx*dimy, n_cells),
                               dtype='float32'), name='a')
      b = tf.Variable(np.array(b_init,dtype='float32'), name='b')
      vars_fit = [w, a]
      if not FLAGS.loss == 'poisson':
        vars_fit = vars_fit + [b]


      # get firing rate
      stim4D = tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
      stim_masked = tf.nn.conv2d(stim4D, mask_tf,
                                 strides=[1, FLAGS.stride, FLAGS.stride, 1],
                                 padding="VALID" )
      stim_wts = tf.nn.relu(tf.reduce_sum(tf.mul(stim_masked, w), 3))
      lam = tf.matmul(tf.reshape(stim_wts, [-1,dimx*dimy]),a) + b

      # regularization
      regularization = FLAGS.lam_w * tf.reduce_sum(tf.nn.l2_loss(w))

      # projection to satisfy hard variable constraints
      a_pos = tf.assign(a, (a + tf.abs(a))/2)
      b_pos = tf.assign(b, (b + tf.abs(b))/2)
      def proj():
        sess.run(a_pos)
        if FLAGS.loss == 'poisson':
          sess.run(b_pos)


    if FLAGS.model_id == 'relu_window_mother':
      # firing rate for cell c: lam_c = a_c'.relu(w.x) + b,
      # where w_i are over a small window which are convolutionally related with each other.
      # w_i = w_mother + w_del_i,
      # where w_mother is common accross all 'windows' and w_del is different for different windows.

      # filename
      short_filename = ('model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) +
                        '_lam_w=' + str(FLAGS.lam_w) + short_filename )
      mask_tf, dimx, dimy, n_pix = get_windows()

      # variables
      w_del = tf.Variable(np.array(0.1+ 0.05*np.random.rand(dimx, dimy, n_pix),
                                   dtype='float32'), name='w_del')
      w_mother = tf.Variable(np.array(np.ones((2 * FLAGS.window + 1,
                                               2 * FLAGS.window + 1, 1, 1)),
                                      dtype='float32'), name='w_mother')
      a = tf.Variable(np.array(np.random.rand(dimx*dimy, n_cells),
                               dtype='float32'), name='a')
      b = tf.Variable(np.array(b_init,dtype='float32'), name='b')
      vars_fit = [w_mother, w_del, a]
      if not FLAGS.loss == 'poisson':
        vars_fit = vars_fit + [b]


      # get firing rate
      stim4D = tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
      stim_convolved = tf.reduce_sum( tf.nn.conv2d(stim4D, w_mother,
                                                   strides=[1, FLAGS.stride,
                                                            FLAGS.stride, 1],
                                                   padding="VALID"),3)
      stim_masked = tf.nn.conv2d(stim4D, mask_tf,
                                 strides=[1, FLAGS.stride, FLAGS.stride, 1],
                                 padding="VALID" )
      stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)
      su_act = tf.nn.relu(stim_del + stim_convolved)
      lam = tf.matmul(tf.reshape(su_act, [-1, dimx*dimy]),a) + b

      # regularization
      regularization =  FLAGS.lam_w * tf.reduce_sum(tf.nn.l2_loss(w_del))

      # projection to satisfy hard variable constraints
      a_pos = tf.assign(a, (a + tf.abs(a))/2)
      b_pos = tf.assign(b, (b + tf.abs(b))/2)
      def proj():
        sess.run(a_pos)
        if FLAGS.loss == 'poisson':
          sess.run(b_pos)


    if FLAGS.model_id == 'relu_window_mother_sfm':
      # firing rate for cell c: lam_c = a_sfm_c'.relu(w.x) + b,
      # a_sfm_c = softmax(a) : so a cell cannot be connected to all subunits equally well.

      # where w_i are over a small window which are convolutionally related with each other.
      # w_i = w_mother + w_del_i,
      # where w_mother is common accross all 'windows' and w_del is different for different windows.

      # we know a>0 and for poisson loss, b>0
      # for poisson loss: small b added to prevent lam_c going to 0

      # filename
      short_filename = ('model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) +
                        '_lam_w=' + str(FLAGS.lam_w) + short_filename)
      mask_tf, dimx, dimy, n_pix = get_windows()

      # variables
      w_del = tf.Variable(np.array( 0.05*np.random.randn(dimx, dimy, n_pix),
                                   dtype='float32'), name='w_del')
      w_mother = tf.Variable(np.array(np.ones((2 * FLAGS.window + 1,
                                               2 * FLAGS.window + 1, 1, 1)),
                                      dtype='float32'), name='w_mother')
      a = tf.Variable(np.array(np.random.randn(dimx*dimy, n_cells),
                               dtype='float32'), name='a')
      a_sfm = tf.transpose(tf.nn.softmax(tf.transpose(a)))
      b = tf.Variable(np.array(b_init,dtype='float32'), name='b')
      vars_fit = [w_mother, w_del, a]
      if not FLAGS.loss == 'poisson':
        vars_fit = vars_fit + [b]

      # get firing rate
      stim4D = tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
      stim_convolved = tf.reduce_sum(tf.nn.conv2d(stim4D, w_mother,
                                                  strides=[1, FLAGS.stride,
                                                           FLAGS.stride, 1],
                                                  padding="VALID"),3)
      stim_masked = tf.nn.conv2d(stim4D, mask_tf, strides=[1, FLAGS.stride,
                                                           FLAGS.stride, 1],
                                 padding="VALID" )
      stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)
      su_act = tf.nn.relu(stim_del + stim_convolved)
      lam = tf.matmul(tf.reshape(su_act, [-1, dimx*dimy]), a_sfm) + b

      # regularization
      regularization = FLAGS.lam_w * tf.reduce_sum(tf.nn.l2_loss(w_del))

      # projection to satisfy hard variable constraints
      b_pos = tf.assign(b, (b + tf.abs(b))/2)
      def proj():
        if FLAGS.loss == 'poisson':
          sess.run(b_pos)



    if FLAGS.model_id == 'relu_window_mother_sfm_exp':
      # firing rate for cell c: lam_c = exp(a_sfm_c'.relu(w.x)) + b,
      # a_sfm_c = softmax(a) : so a cell cannot be connected to all subunits equally well.
      # exponential output NL would cancel the log() in poisson and might get better estimation properties.

      # where w_i are over a small window which are convolutionally related with each other.
      # w_i = w_mother + w_del_i,
      # where w_mother is common accross all 'windows' and w_del is different for different windows.

      # we know a>0 and for poisson loss, b>0
      # for poisson loss: small b added to prevent lam_c going to 0

      # filename
      short_filename = ('model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) +
                        '_lam_w=' + str(FLAGS.lam_w) + short_filename)
      mask_tf, dimx, dimy, n_pix = get_windows()

      # model
      w_del = tf.Variable(np.array(0.1 + 0.05*np.random.randn(dimx, dimy, n_pix),
                                   dtype='float32'), name='w_del')
      w_mother = tf.Variable(np.array(0.05 * np.ones((2 * FLAGS.window + 1,
                                                      2 * FLAGS.window + 1, 1, 1)),
                                      dtype='float32'), name='w_mother')
      a = tf.Variable(np.array(np.random.randn(dimx*dimy, n_cells),
                               dtype='float32'), name='a')
      a_sfm = tf.transpose(tf.nn.softmax(tf.transpose(a)))
      b = tf.Variable(np.array(b_init,dtype='float32'), name='b')
      vars_fit = [w_mother, w_del, a]
      if not FLAGS.loss == 'poisson':
        vars_fit = vars_fit + [b]

      # get firing rate
      stim4D = tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
      stim_convolved = tf.reduce_sum( tf.nn.conv2d(stim4D, w_mother,
                                                   strides=[1, FLAGS.stride,
                                                            FLAGS.stride, 1],
                                                   padding="VALID"),3)
      stim_masked = tf.nn.conv2d(stim4D, mask_tf, strides=[1, FLAGS.stride,
                                                           FLAGS.stride, 1],
                                 padding="VALID" )
      stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)
      su_act = tf.nn.relu(stim_del + stim_convolved)
      lam = tf.exp(tf.matmul(tf.reshape(su_act, [-1, dimx*dimy]), a_sfm)) + b

      # regularization
      regularization = FLAGS.lam_w * tf.reduce_sum(tf.nn.l2_loss(w_del))

      # projection to satisfy hard variable constraints
      b_pos = tf.assign(b, (b + tf.abs(b))/2)
      def proj():
        if FLAGS.loss == 'poisson':
          sess.run(b_pos)



    # different loss functions
    if FLAGS.loss == 'poisson':
      loss_inter = (tf.reduce_sum(lam)/120. -
                    tf.reduce_sum(resp*tf.log(lam))) / data_len

    if FLAGS.loss == 'logistic':
      loss_inter = tf.reduce_sum(tf.nn.softplus(-2 * (resp - 0.5)*lam)) / data_len

    if FLAGS.loss == 'hinge':
      loss_inter = tf.reduce_sum(tf.nn.relu(1 -2 * (resp - 0.5)*lam))  / data_len
 
    loss = loss_inter + regularization
    train_step = tf.train.AdagradOptimizer(FLAGS.step_sz).minimize(loss,
                                                                   var_list=
                                                                   vars_fit)
    def training(inp_dict):
      sess.run(train_step, feed_dict=inp_dict)
      proj() # model specific projection operations

    def get_loss(inp_dict):
      ls = sess.run(loss,feed_dict = inp_dict)
      return ls

    model_obj = model(stim, su_act, lam)

    # saving details - to load checkpoints from
    parent_folder = FLAGS.save_location + FLAGS.folder_name + '/'
    # make folder if does not exist
    save_location = parent_folder + short_filename + '/'
    save_filename = save_location + short_filename


    # create summary writers
    for ivar in vars_fit:
      tf.histogram_summary(ivar.name, ivar)
    l_summary = tf.scalar_summary('loss',loss)
    l_inter_summary = tf.scalar_summary('loss_inter',loss_inter)
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(save_location + 'train',
                                      sess.graph)
    test_writer = tf.train.SummaryWriter(save_location + 'test')


    ## load fits
    print('Load fits')
    sess.run(tf.initialize_all_variables())
    saver_var = tf.train.Saver(tf.all_variables(),
                               keep_checkpoint_every_n_hours=0.05)
    load_prev = False
    start_iter=0
    try:
      latest_filename = short_filename + '_latest_fn'
      restore_file = tf.train.latest_checkpoint(save_location, latest_filename)
      start_iter = int(restore_file.split('/')[-1].split('-')[-1])
      saver_var.restore(sess, restore_file)
      load_prev = True
    except:
      print('No previous dataset')

    if load_prev:
      print('\nPrevious results loaded')
    else:
      print('\nVariables initialized')

    print('Iteration: ' + str(start_iter))


    ## Load testing data
    icnt = 0
    stim_test,resp_test,test_length = get_test_data()
    fd_test = {stim: stim_test,
               resp: resp_test,
               data_len: test_length}


    ## Perform analysis
    # plot a - the connection between cells and subunits
    if (FLAGS.model_id == 'relu_window_mother_sfm' or
        FLAGS.model_id == 'relu_window_mother_sfm_exp'):
      # for some models, plot soft-maxed weights
      a_sfm_eval = sess.run(a_sfm)
      plt.imshow(a_sfm_eval, interpolation='nearest', cmap='gray')
    else:
      plt.imshow(sess.run(a), interpolation='nearest', cmap='gray')
    plt.title('subunit to cell connection matrix')
    plt.show()
    plt.draw()

    # Plot subunit filters
    # for models where weights were learnt over full space
    if FLAGS.model_id == 'exp_additive' or FLAGS.model_id == 'relu':
      wts = w.eval()
      plot_wts_full(wts)


    # for models where weights are learnt only over a small window
    if FLAGS.model_id == 'relu_window':
      wts = w.eval()
      #plot_wts_windows(wts, dimx, dimy)


    # plot mother subunit for 'almost convolutional - model + delta models'
    if (FLAGS.model_id == 'relu_window_mother' or
        FLAGS.model_id == 'relu_window_mother_sfm' or
        FLAGS.model_id == 'relu_window_mother_sfm_exp'):
      # plot mother subunit
      w_mot = np.squeeze(w_mother.eval())
      plt.imshow(w_mot, interpolation='nearest', cmap='gray')
      plt.title('Mother subunit')
      plt.show()
      plt.draw()

      # plot delta subunit for 'almost convolutional - model + delta models'
      w_del_e = np.squeeze(w_del.eval())
      wts = np.array(0 * np.random.randn(dimx, dimy, (2*FLAGS.window +1)**2))
      for idimx in np.arange(dimx):
        print(idimx)
        for idimy in np.arange(dimy):
          wts[idimx, idimy, :] = (np.ndarray.flatten(w_mot) +
                                  w_del_e[idimx, idimy, :])
      plt.plot(np.ndarray.flatten(w_del_e))
      plt.draw()
      plt.show()
      #plot_wts_windows(wts, dimx, dimy)


    # plot subunits inferred for a few selected cells
    if (FLAGS.model_id == 'relu_window_mother_sfm' or
        FLAGS.model_id == 'relu_window_mother_sfm_exp'):
      a_e = sess.run(a_sfm)
    else:
      a_e = sess.run(a)


    from IPython.terminal.embed import InteractiveShellEmbed
    ipshell = InteractiveShellEmbed()
    ipshell()

    #plot_su_for_cells(a_e, wts, np.array([0,1,2,48,95,23]),
    #                  total_mask, dimx, dimy, stas)


    ## TODO(bhaishahster): plot subunits in convolution mesh for particular models

    ## TODO(bhaishahster): degradation due to zeroing out weak subunits

    ## how many subunits are shared ?
    #print('shared subunits')
    #analyse_shared_su(a_e, wts, dimx, dimy, stas)


    ## plot 2D contour of all subunits which are strongly connected to atleast one cell
    #plot_su_contour(a_e, wts,dimx, dimy)

    ## compare with LNP model
    #make_LNP_model(a_e, wts, dimx, dimy, stas, model_obj, sess, total_mask)

    # what is the amount of correlation induced by shared subunits?
    #correlation_su_strength_analysis(a_e, wts, dimx, dimy, stas, model_obj, sess)

    ## plot 2D gaussian fit contour of cells
    # plot_su_gaussian(a_e, wts, dimx, dimy, stas)
    plot_su_gaussian_spokes(a_e, wts, dimx, dimy, stas, w_mot, w_del_e)

    ## test repeats response prediction analysis
    print('multiple trial repeats')
    stim_test, resp_test = load_repeats_data(a_e, wts)
    lam_test = sess.run(lam, feed_dict = {stim:2*stim_test})
    resp_model_test = np.zeros(resp_test.shape)
    for itrial in np.arange(resp_test.shape[2]):
      resp_model_test[:,:,itrial] = np.random.poisson(lam_test/120.)

    cell_idx_permute = np.random.permutation(np.arange(n_cells))
    ncells = 10
    # single cell, many trials
    plt.figure()
    for icell in np.arange(ncells):
      plt.subplot(ncells, 1, icell+1)
      plot_raster(np.squeeze(resp_model_test[:,cell_idx_permute[icell],:]),
                  col='r')
      plot_raster(np.squeeze(resp_test[:,cell_idx_permute[icell],:]),
                  col='k',shift=30)
      plt.title('Cell ID: ' + str(cells[cell_idx_permute[icell]]))
    plt.show()
    plt.draw()


    # multiple cells, single trials
    plt.figure()
    n_trials = 10
    for itrial in np.arange(n_trials):
      plt.subplot(n_trials, 1, itrial+1)
      plot_raster(np.squeeze(resp_model_test[:,:,itrial]), col='r')
      plot_raster(np.squeeze(resp_test[:,:,itrial]),col='k',
                  shift=np.squeeze(cells).shape[0])
      plt.title('Trial: ' + str(itrial))
    plt.show()
    plt.draw()



    ## correlation plot between nearby cells
    # get cell responses
    print('Correlation curves')
    # get pair of nearby cells
    stim_test,resp_test,test_length = get_test_data()
    fd_test = {stim: stim_test,
              resp: resp_test,
              data_len: test_length}
    lam_test = sess.run(lam, feed_dict = {stim:stim_test})
    resp_model_test = np.random.poisson(lam_test/120.)


    # get nearby cells
    cell_origin, cell_dest, _ = get_nearby_cells(stas)
    ipair = 1
    cell_a = cell_origin[ipair].astype('int')
    cell_b = cell_dest[ipair].astype('int')
    print(np.correlate(np.squeeze(resp_test[:,cell_a]),
                       np.squeeze(resp_test[:,cell_b])))
    plt.figure()
    plt.plot(np.correlate(np.squeeze(resp_test[:,cell_a]),
                          np.squeeze(resp_test[:,cell_b])))
    plt.hold(True)
    plt.plot(np.correlate(np.squeeze(resp_model_test[:,cell_a]),
                          np.squeeze(resp_model_test[:,cell_b])))
    plt.title('Response correlation between nearby cells')
    plt.legend(['Recorded data', 'Model response'])
    plt.show()
    plt.draw()




def get_nearby_cells(stas):
  # ger nearby cells
  nCells = stas.shape[1]
  centersx=np.array([])
  centersy=np.array([])
  for icell in range(nCells):
    xx=stas[:,icell]
    xx = np.reshape(xx,[40,80])
    out =np.where(xx==np.amin(xx[:]))
    centersx = np.append(centersx, np.squeeze(np.array(out[0])))
    centersy = np.append(centersy, np.squeeze(np.array(out[1])))


  dist_mat = np.zeros((nCells,nCells))
  for icell in range(nCells):
    for jcell in range(nCells):
      dist_mat[icell,jcell] = np.sqrt((centersx[icell]-centersx[jcell])**2 +
                                      (centersy[icell] - centersy[jcell])**2)

  threshold = 6

  cell_origin = np.array([])
  cell_dest = np.array([])

  plt.figure()
  plt.hold(True)
  xx =np.arange(nCells)
  for icell in range(nCells):
    for jcell in xx[dist_mat[icell,:]<threshold]:
      if jcell>icell:
        plt.plot([centersy[icell],centersy[jcell]],[centersx[icell],
                                                    centersx[jcell]],'r')
        cell_origin = np.append(cell_origin,icell)
        cell_dest = np.append(cell_dest,jcell)
            
  cell_pairs = np.array([cell_origin,cell_dest],dtype='int').T
  plt.plot(centersy,centersx,'.')
  plt.axis('image')
  plt.xlim([0,80])
  plt.ylim([0,40])
  plt.title('Cell centers and interaction terms')
  plt.draw()
  plt.show()

  return cell_origin, cell_dest, dist_mat



def plot_raster(bin_matrix, col='r', shift=0):
  # bin_matrix is time x rows
  tms = np.arange(bin_matrix.shape[0])
  for itrial in np.arange(bin_matrix.shape[1]):
    itrial
    tms_spks = tms[bin_matrix[:, itrial]>0]
    plt.vlines(tms_spks,shift+itrial+0.5, shift+itrial+1.5, color=col)


def load_repeats_data(a_e, wts):
  # load white noise repeats - stimulus and response
  data_filename = FLAGS.data_location + 'OFF_parasol_trial_resp_data2.mat'
  summary_file = gfile.Open(data_filename, 'r')
  data_summary = sio.loadmat(summary_file)

  cids_test = data_summary['cids']
  condMov_mat_test = data_summary['condMov_mat']
  dataRuns = data_summary['dataRuns']
  movies = data_summary['movies']
  resp_log_mat = data_summary['resp_log_mat']

  use_condition = 2
  print('Using condition: ' + str(use_condition))
  print(condMov_mat_test.shape)
  print(resp_log_mat.shape)
  stimulus = np.transpose(np.squeeze(condMov_mat_test[use_condition, :, :, :]),
                          (2, 0, 1))
  stimulus = np.reshape(stimulus, (-1, 80*40))
  response = np.transpose(np.squeeze(resp_log_mat[:, use_condition, :, :]),
                          (2,0,1))
  return stimulus, response


# Analysis functions
def analyse_shared_su(b_eval, wts, dimx, dimy, stas):
  # plot shared subunits with corresponding cells to which they connect
  window = FLAGS.window
  b_thr = 0.02 # threshold for saying if a subunit is connected to a cell
  isharedSU = 1
  icnt = -1
  print(stas.shape)
  plt.figure()
  n_su_max = 20
  n_cells_max = 5;
  cell_list = np.arange(stas.shape[1])
  for idimx in np.arange(dimx):
    for idimy in np.arange(dimy):
      if(isharedSU > n_su_max):
        break
      icnt = icnt + 1
      connected_cells = np.abs(b_eval[icnt, :]) > b_thr
      if(np.sum(connected_cells)>=2): # found a shared SU
        print('plotting:', idimx, idimy)
        ww = np.zeros((40,80))
        ww[idimx*FLAGS.stride: idimx*FLAGS.stride + (2*window+1),
           idimy*FLAGS.stride: idimy*FLAGS.stride + (2*window+1)] = (
            np.reshape(wts[idimx, idimy, :], (2*window+1,2*window+1)))
        cells_connected = cell_list[connected_cells]

        # plot the shared subunit
        ax1 = plt.subplot(n_su_max, n_cells_max, n_cells_max*(isharedSU-1)+1)
        fig = plt.imshow(ww, interpolation='nearest', cmap='gray')
        plt.hold(True)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        ax1.set_ylim([idimx*FLAGS.stride-3,
                      idimx*FLAGS.stride + (2*window+1)+3])
        ax1.set_xlim([idimy*FLAGS.stride-3,
                      idimy*FLAGS.stride + (2*window+1)+3])

        for icntcell, ipltcell in enumerate(cells_connected):
          if icntcell>=n_cells_max:
            break
          ax1 = plt.subplot(n_su_max, n_cells_max,
                            n_cells_max*(isharedSU-1)+1+icntcell+1)
          fig = plt.imshow(np.reshape(stas[:, ipltcell], [40, 80]),
                           interpolation='nearest', cmap='gray')
          plt.hold(True)
          fig.axes.get_xaxis().set_visible(False)
          fig.axes.get_yaxis().set_visible(False)
          ax1.set_ylim([idimx*FLAGS.stride-3,
                        idimx*FLAGS.stride + (2*window+1)+3])
          ax1.set_xlim([idimy*FLAGS.stride-3,
                        idimy*FLAGS.stride + (2*window+1)+3])

          plt.title(str(b_eval[icnt, ipltcell]))

        isharedSU = isharedSU + 1
  plt.show()
  plt.draw()


def plot_wts_full(wts):
  # plot weights which over the full stimulus space
  for isu in range(100):
    fig = plt.subplot(10, 10, isu+1)
    plt.imshow(np.reshape(wts[:, isu],[40, 80]),
               interpolation='nearest', cmap='gray')
  plt.title('Iteration: ' + str(int(iter_plot)))
  fig.axes.get_xaxis().set_visible(False)
  fig.axes.get_yaxis().set_visible(False)


def plot_wts_windows(wts, dimx, dimy):
  # plot weights which are over small windows
  window = FLAGS.window
  print('wts shape:', np.shape(wts))
  icnt=1
  for idimx in np.arange(dimx):
    for idimy in np.arange(dimy):
      fig = plt.subplot(dimx, dimy, icnt)
      plt.imshow(np.reshape(np.squeeze(wts[idimx, idimy, :]),
                            (2*window+1,2*window+1)),
                 interpolation='nearest', cmap='gray')
      icnt = icnt+1
      fig.axes.get_xaxis().set_visible(False)
      fig.axes.get_yaxis().set_visible(False)
  plt.show()
  plt.draw()


def plot_su_contour(b_eval, wts, dimx, dimy):
# plot contour over fitted subunits
    window = FLAGS.window
    ww_sum = np.zeros((40, 80))
    ifig = 0
    plt.figure()
    for icell_cnt, icell in enumerate(np.arange(1)):
      icnt = -1
      a_thr = 0.1#np.percentile(np.abs(b_eval[:, icell]), 99.5)
      for idimx in np.arange(dimx):
        for idimy in np.arange(dimy):
          icnt = icnt + 1
          if(np.abs(b_eval[icnt,icell]) > a_thr):
            print('plotting:', icell, idimx, idimy)
            ww = np.zeros((40,80))
            ww[idimx*FLAGS.stride: idimx*FLAGS.stride + (2*window+1),
               idimy*FLAGS.stride:
               idimy*FLAGS.stride + (2*window+1)] =  b_eval[icnt, icell] * (
                np.reshape(wts[idimx, idimy, :], (2*window+1,2*window+1)))
            #plt.imshow(ww, interpolation='nearest', cmap='gray')
            plt.hold(True)
            r = lambda: random.randint(0,255)
            rgbl = [255, 128, 0]
            random.shuffle(rgbl)
            print(rgbl)
            plt.contour(ww < 0.8*np.max(np.abs(np.ndarray.flatten(ww))), 1,
                        colors='#%02x%02x%02x' % tuple(rgbl))
            ww_sum = ww_sum + (ww)**2
    plt.show()
    plt.draw()

    plt.figure()
    plt.imshow(ww_sum, interpolation='nearest', cmap='gray')
    plt.show()
    plt.draw()    


def plot_su_for_cells(b_eval, wts, cells, total_mask, dimx, dimy, stas):
    # plot strong subunits, true STA and STA from fitted subunits for each cell.
    n_cells = cells.shape[0]
    window = FLAGS.window
    plt.hist(np.ndarray.flatten(b_eval))
    plt.show()
    plt.draw()

    from IPython.terminal.embed import InteractiveShellEmbed
    ipshell = InteractiveShellEmbed()
    ipshell()


    plt.ion()
    for icell_cnt, icell in enumerate(cells):
      plt.figure()
      mask2D = np.reshape(total_mask[icell,: ], [40, 80])
      nz_idx = np.nonzero(mask2D)
      np.shape(nz_idx)
      print(nz_idx)
      ylim = np.array([np.min(nz_idx[0])-1, np.max(nz_idx[0])+1])
      xlim = np.array([np.min(nz_idx[1])-1, np.max(nz_idx[1])+1])

      icnt = -1
      xx = np.sort(np.ndarray.flatten(b_eval[:,icell]));
      idx = np.min(np.where(np.diff(xx)>0.01))
      a_thr = (xx[idx] + xx[idx+1])/2

      n_plots = np.sum(np.abs(b_eval[:, icell]) > a_thr)
      nx = np.ceil(np.sqrt(n_plots)).astype('int')
      ny = np.ceil(np.sqrt(n_plots)).astype('int')
      ifig=0
      ww_sum = np.zeros((40,80))

      for idimx in np.arange(dimx):
        for idimy in np.arange(dimy):
          icnt = icnt + 1
          n_plots_max = np.max(np.sum(b_eval[:,icell] > a_thr,0))+2
          if (np.abs(b_eval[icnt,icell]) > a_thr): # strongly connected subunit
            ifig = ifig + 1
            fig = plt.subplot(1, n_plots_max, ifig + 2)
            print(n_cells, n_plots_max, icell_cnt*n_plots_max + ifig + 2)

            ww = np.zeros((40,80))
            ww[idimx*FLAGS.stride: idimx*FLAGS.stride + (2*window+1),
               idimy*FLAGS.stride:
               idimy*FLAGS.stride + (2*window+1)] = (b_eval[icnt, icell] * (
                np.reshape(wts[idimx, idimy, :], (2*window+1,2*window+1))))
            plt.imshow(ww, interpolation='nearest', cmap='gray')
            plt.ylim(ylim)
            plt.xlim(xlim)
            plt.title(str(b_eval[icnt,icell]), fontsize=10)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            ww_sum = ww_sum + ww

      # plot STA from the fitted subunits of the model by
      # just adding them together
      fig = plt.subplot(1, n_plots_max, 2)
      plt.imshow(ww_sum, interpolation='nearest', cmap='gray')
      plt.ylim(ylim)
      plt.xlim(xlim)
      fig.axes.get_xaxis().set_visible(False)
      fig.axes.get_yaxis().set_visible(False)
      plt.title('STA from model',fontsize=10)

      # plot true STA from WN
      fig = plt.subplot(1, n_plots_max, 1)
      plt.imshow(np.reshape(stas[:, icell], [40, 80]), interpolation='nearest', cmap='gray')
      plt.ylim(ylim)
      plt.xlim(xlim)
      fig.axes.get_xaxis().set_visible(False)
      fig.axes.get_yaxis().set_visible(False)
      plt.title('True STA'+ ' thr: ' + str(a_thr),fontsize=10)

    plt.ioff()

def make_LNP_model(b_eval, wts, dimx, dimy, stas, model, sess, total_mask):


  from IPython.terminal.embed import InteractiveShellEmbed
  ipshell = InteractiveShellEmbed()
  ipshell()

  # get stimulus and responses
  n_cells = stas.shape[1]
  responses = np.zeros((0, n_cells))
  stimulus = np.zeros((216000, 3200))
  lam_model = np.zeros((216000, n_cells))
  su_act_model = np.zeros((216000, dimx, dimy))
  for ichunk in np.arange(FLAGS.n_chunks):
    print('Loading %d' %ichunk)
    X, Y,_ = get_next_training_batch(ichunk)
    responses = np.append(responses, Y,0)
    stimulus[ichunk*1000: (ichunk+1)*1000, :] = X
    lam_model[ichunk*1000: (ichunk+1)*1000, :] = sess.run(model.lam, feed_dict={model.stim:X})
    su_act_model[ichunk*1000: (ichunk+1)*1000, :] = sess.run(model.su_act, feed_dict={model.stim:X})


  # fit LNP models
  model_lnp_log = [[]]*n_cells
  cell_permuted = np.random.permutation(np.arange(n_cells))
  for icell in cell_permuted:
    gra = tf.Graph()
    with gra.as_default():
      with tf.Session() as sess2:
        # masks?
        mask = np.array(total_mask[icell, :]>0)

        # prepare stimulus and response
        stim_cell = stimulus[:,mask]
        resp_cell = np.array(np.expand_dims(responses[:, icell],1), dtype='float32')

        # prepare model
        stim_lnp = tf.placeholder(tf.float32, shape=(None, np.sum(total_mask[icell, :]>0)))
        resp_lnp = tf.placeholder(tf.float32)
        data_len  = tf.placeholder(tf.float32)

        w_lnp = tf.Variable(0.1*np.array(np.random.randn(np.sum(mask),1),dtype='float32'))
        bias_lnp = tf.Variable(np.array(0, dtype='float32')) #tf.Variable(np.array(0, dtype='float32'))
        lam_lnp = tf.nn.relu(tf.matmul(stim_lnp, w_lnp) + bias_lnp)+ 0.0000001
        loss_lnp =  (tf.reduce_sum(lam_lnp)/120. - tf.matmul(tf.transpose(resp_lnp),tf.log(lam_lnp))) / data_len
        train_step = tf.train.AdagradOptimizer(20).minimize(loss_lnp)

        # fit results
        sess2.run(tf.initialize_all_variables())
        loss_lnp_np = np.inf
        loss_lnp_np_log = np.array([])
        for iiter in  range(50000):
          loss_lnp_np_prev = loss_lnp_np
          _, loss_lnp_np = sess2.run([train_step,loss_lnp],
                                     feed_dict = {stim_lnp: stim_cell,
                                                  resp_lnp: resp_cell,
                                                  data_len: resp_cell.shape[0]})

          if np.abs(loss_lnp_np-loss_lnp_np_prev)<1e-7:
            break

          print(icell, iiter, loss_lnp_np)
          loss_lnp_np_log = np.append(loss_lnp_np_log, loss_lnp_np)
        w_lnp_np  =sess2.run(w_lnp)
        bias_lnp_np = sess2.run(bias_lnp)
    model_lnp_log[icell] = [model_lnp(w_lnp_np, bias_lnp_np, mask, loss_lnp_np_log)]

  print('multiple trial repeats')
  stim_test, resp_test = load_repeats_data(b_eval, wts)

# population model LNP
  lam_population = sess.run(model.lam, feed_dict={model.stim: stim_test})
  LL_pop =  (np.sum(np.sum(np.repeat(np.expand_dims(lam_population,2),30, axis=2)/120 - resp_test*np.log(np.expand_dims(lam_population, axis=2)),axis=2), axis=0))/(1200*30)

  # LNP population cel LL
  LL_lnp = np.zeros(n_cells)
  use_cells = []
  n_cells_analysed = 0
  for icell in np.arange(n_cells):
    if not model_lnp_log[icell] == []:
      use_cells +=[icell]
      n_cells_analysed +=1
      print(n_cells_analysed)

      mask = model_lnp_log[icell][0].mask
      stim_masked = stim_test[:, mask]
      wts = model_lnp_log[icell][0].w
      bias = model_lnp_log[icell][0].b
      lam_eval_lnp = np.dot(stim_masked, wts) + bias
      lam_eval_lnp *= lam_eval_lnp>0
      lam_eval_lnp = np.squeeze(lam_eval_lnp) +0.0000001
      ll = np.sum(np.repeat(np.expand_dims(lam_eval_lnp,1),30,1)/120 - resp_test[:, icell,:]*np.log(np.expand_dims(lam_eval_lnp, axis=1)))/(1200*30)
      LL_lnp[icell] = ll

  from IPython.terminal.embed import InteractiveShellEmbed
  ipshell = InteractiveShellEmbed()
  ipshell()

  plt.ion()
  plt.plot(LL_lnp[use_cells], LL_pop[use_cells],'.',markersize=15)
  plt.xlabel('LNP')
  plt.ylabel('population model')



def correlation_su_strength_analysis(b_eval, wts, dimx, dimy, stas, model, sess):


  from IPython.terminal.embed import InteractiveShellEmbed
  ipshell = InteractiveShellEmbed()
  ipshell()

  # get stimulus and responses
  n_cells = stas.shape[1]
  responses = np.zeros((0, n_cells))
  stimulus = np.zeros((216000, 3200))
  lam_model = np.zeros((216000, n_cells))
  su_act_model = np.zeros((216000, dimx, dimy))
  for ichunk in np.arange(FLAGS.n_chunks):
    print('Loading %d' %ichunk)
    X, Y,_ = get_next_training_batch(ichunk)
    responses = np.append(responses, Y,0)
    stimulus[ichunk*1000: (ichunk+1)*1000, :] = X
    lam_model[ichunk*1000: (ichunk+1)*1000, :] = sess.run(model.lam, feed_dict={model.stim:X})
    su_act_model[ichunk*1000: (ichunk+1)*1000, :] = sess.run(model.su_act, feed_dict={model.stim:X})

  # find shared subunits
  su_list_cells =  [[[] for i in range(dimy)] for j in range(dimx)]
  su_weight_cells =  [[[] for i in range(dimy)] for j in range(dimx)]
  su_num_cells = np.zeros((dimx, dimy))
  for icell in np.arange(n_cells):
    # remove bad cells
    if icell==30 or icell==56: # has low number of spikes = (5000 or 8000)
      continue;

    if icell==43: # not well spike sorted - has bad STA in vision
      continue;

    if icell==105: # high # spikes, STA ok, but connects to a far away su - remove it!
      continue;

    # select thereshold based on breakpoint in values
    xx = np.sort(np.ndarray.flatten(b_eval[:,icell]));
    idx = np.min(np.where(np.diff(xx)>0.01))
    a_thr = (xx[idx] + xx[idx+1])/2
    icnt=-1;
    for idimx in np.arange(dimx):
      for idimy in np.arange(dimy):
        icnt+=1;
        if b_eval[icnt, icell]>a_thr :
          su_num_cells[idimx, idimy] +=1
          su_list_cells[idimx][idimy] += [icell]
          su_weight_cells[idimx][idimy] += [np.mean(su_act_model[:,idimx, idimy])*b_eval[icnt, icell]]


  # response correlation v/s activation of shared subunit curves.
  x,y = np.where(su_num_cells>1)
  xcor = []
  mean_wts = []
  for isu_shared in np.arange(len(x)):
    ix = x[isu_shared]
    iy = y[isu_shared]
    cells = su_list_cells[ix][iy]
    cc = np.corrcoef(responses[:, cells[0]], responses[:, cells[1]])
    xcor += [cc[0,1]]
    mean_wts += [np.mean([su_weight_cells[ix][iy][0], su_weight_cells[ix][iy][1]])]
    #plt.text(mean_wts[-1], xcor[-1], str([cells[0],cells[1]]))
    print(isu_shared, mean_wts[-1], xcor[-1])

  plt.ion()
  cc = np.corrcoef(mean_wts, xcor)
  plt.plot(mean_wts, xcor,'.', markersize=20, color='k');
  plt.hold(True);
  plt.plot([0,1.6],[0,np.mean(xcor)*1.6/np.mean(mean_wts)],'k',linewidth=2, alpha=0.2);
  plt.title('slope %.3f, corr_coef %.3f' % (np.mean(mean_wts)/np.mean(xcor) , cc[0,1]) )
  plt.xlim([0,1.6])
  plt.ylim([0,0.12])
  plt.xlabel('Mean strength of shared SU')
  plt.ylabel('Correlation between cell responses')
  plt.savefig('Correlation.pdf')
  plt.show()


  # count how many subunits shared between cells
  cell_common_su = np.zeros((n_cells, n_cells))
  x,y = np.where(su_num_cells>1)
  for isu_shared in np.arange(len(x)):
    ix = x[isu_shared]
    iy = y[isu_shared]
    cells = su_list_cells[ix][iy]
    for icell in cells:
      for jcell in cells:
        if(icell>=jcell): # avoid repeat and self counting
          continue
        cell_common_su[icell, jcell] +=1
        cell_common_su[jcell, icell] +=1

  # conditional dependence analysis
  pts = np.arange(5,95,1)
  npts = pts.shape[0]
  pop_mean_su_act = np.zeros(npts)
  pop_cella1 = np.zeros(npts);
  pop_cella1_cellb1 = np.zeros(npts);
  pop_cella1_cellb0 = np.zeros(npts);
  icnt=0
  for isu_shared in range(len(x)):#[2]:#[2, 42]:
    print(isu_shared)
    ix = x[isu_shared]
    iy = y[isu_shared]
    cells = su_list_cells[ix][iy]
    if(cell_common_su[cells[0], cells[1]] >1):
      print('skipping a pair with more than one subunit common')
      continue
    resp0 = responses[:, cells[0]]
    resp1 = responses[:, cells[1]]
    su_act_analysis = su_act_model[:, ix, iy]

    mean_fr = [[[] for i in range(2)] for j in range(2)]
    mean_su_act = np.array([])
    for ithr in pts:
      lb = np.percentile(su_act_analysis, max(ithr-5,0))
      ub = np.percentile(su_act_analysis, min(ithr+5,100))
      tms = np.logical_and(su_act_analysis>=lb, su_act_analysis<=ub+0.0000001)
      #mean_su_act = np.append(mean_su_act, np.mean(su_act_analysis[tms]))
      mean_su_act = np.append(mean_su_act, ithr)
      for conda in [0,1]:
        for condb in [0,1]:
          mean_val = np.mean(np.double(np.logical_and(resp0[tms]==conda, resp1[tms] ==condb)))
          mean_fr[conda][condb] = np.append(mean_fr[conda][condb], mean_val)

    pop_mean_su_act +=np.array(mean_su_act)

    # cell 1
    icnt+=1
    plt.hold(True)
    cella1 = mean_fr[1][1]+mean_fr[1][0]
    plt.plot(mean_su_act, cella1,'b', alpha=0.2);
    pop_cella1 +=cella1

    plt.hold(True);
    cella1_b1 = (mean_fr[1][1])/(mean_fr[0][1]+mean_fr[1][1])
    plt.plot(mean_su_act,cella1_b1,'g', alpha=0.2);
    pop_cella1_cellb1+=cella1_b1

    plt.hold(True)
    cella1_b0 = (mean_fr[1][1])/(mean_fr[0][0]+mean_fr[1][0])
    plt.plot(mean_su_act,cella1_b0 ,'r', alpha=0.2);
    pop_cella1_cellb0 += cella1_b0

    # cell 2
    icnt+=1
    plt.hold(True)
    cella1 = mean_fr[1][1]+mean_fr[0][1]
    plt.plot(mean_su_act, cella1,'b', alpha=0.2);
    pop_cella1 += cella1

    plt.hold(True);
    cella1_b1 = (mean_fr[1][1])/(mean_fr[1][0]+mean_fr[1][1])
    plt.plot(mean_su_act,cella1_b1,'g', alpha=0.2);
    pop_cella1_cellb1+=cella1_b1

    plt.hold(True)
    cella1_b0 = (mean_fr[1][1])/(mean_fr[0][0]+mean_fr[0][1])
    plt.plot(mean_su_act, cella1_b0, 'r', alpha=0.2);
    pop_cella1_cellb0 += cella1_b0

  pop_mean_su_act = 2*pop_mean_su_act/icnt
  pop_cella1 = pop_cella1/icnt
  pop_cella1_cellb1 = pop_cella1_cellb1/icnt
  pop_cella1_cellb0 = pop_cella1_cellb0/icnt

  plt.hold(True)
  plt.plot(pop_mean_su_act, pop_cella1, 'b', linewidth=2)
  plt.hold(True)
  plt.plot(pop_mean_su_act, pop_cella1_cellb1, 'g', linewidth=2)
  plt.hold(True)
  plt.plot(pop_mean_su_act, pop_cella1_cellb0, 'r', linewidth=2)
  
  plt.xlabel('activation of shared SU');
  plt.ylabel('response probability of cell 1')
  plt.savefig('conditional_resp_curves.pdf')
  plt.show()


  # cross correlation function analysis
  for isu_shared in [42]:#[2, 42]:
    ix = x[isu_shared]
    iy = y[isu_shared]
    cells = su_list_cells[ix][iy]
    recorded_xcorr = np.correlate(responses[:, cells[0]], responses[:, cells[1]], 'same')
    model_xcorr = np.correlate(lam_model[:, cells[0]]/120, lam_model[:, cells[1]]/120, 'same')
    length = responses.shape[0]
    plt.plot(np.double(np.arange(-30,30,1))/120,recorded_xcorr[length/2-30: length/2+30],color='k', linewidth=2.0)
    plt.hold(True)
    plt.plot(np.double(np.arange(-30,30,1))/120,model_xcorr[length/2-30: length/2+30],color='r', linewidth=2.0)
    plt.xlabel('time (s)');
    plt.legend(['recorded','model'])
    plt.savefig('correlation plots su:%d, cells: %d, %d.pdf' % (isu_shared, cells[0], cells[1] ))
    plt.show()


def plot_su_gaussian_spokes(b_eval, wts, dimx, dimy, stas, w_mot, w_del_e):

  # plot contour over fitted subunits
  window = FLAGS.window
  ww_sum = np.zeros((40, 80))
  ifig = 0

  from IPython.terminal.embed import InteractiveShellEmbed
  ipshell = InteractiveShellEmbed()
  ipshell()

  # get cell centers
  ncells = b_eval.shape[1]
  cell_centers = [[]]*ncells

  for icell in np.arange(ncells):
    try:
      cell_centers[icell],_ = plot_wts_gaussian_fits(np.reshape(stas[:, icell], (40,80)), colors='r',
                               levels = 0.8 , alpha_fill=0.3, fill_color='k', toplot=False)
      plt.hold(True)
      print('cell %d done' % icell)
    except:
      print('Failed fitting of STA')
    plt.hold(True)

  # choose colors for cells by approximately solving 3-graph coloring problem.
  cell_cell_distance = np.zeros((ncells, ncells))
  for icell in np.arange(ncells):
    print(icell)
    for jcell in np.arange(ncells):
      cell_cell_distance[icell, jcell] = np.linalg.norm(np.array(cell_centers[icell])-np.array(cell_centers[jcell]))
  A = np.exp(-cell_cell_distance/150)
  w,v = np.linalg.eig(A)
  col_idx = np.argmax(v[:,-5:], axis=1)

  # plot subunits for all the cells, and make su-cell spokes
  su_cell_cnt = np.zeros((dimx, dimy));
  su_centersx = np.zeros((dimx, dimy));
  su_centersy = np.zeros((dimx, dimy));
  cell_num_su = np.array([])
  plt.ion()
  fig = plt.figure()
  ax = plt.subplot(111)
  ax.set_axis_bgcolor((0.9, 0.9, 0.9))

  dist_log=np.array([])
  r = lambda: np.double(random.randint(0,255))/255
  cols = np.array([[0,0,0.8],[0.8,0,0],[0,0.8,0], [0.8,0,0.8], [0.4, 0, 0.8]])
  for icell_cnt, icell in enumerate(np.arange(b_eval.shape[1])):
    if icell==30 or icell==56: # has low number of spikes = (5000 or 8000)
      continue;

    if icell==43: # not well spike sorted - has bad STA in vision
      continue;
    if icell==105: # high # spikes, STA ok, but connects to a far away su - remove it!
      continue;

    icnt = -1
    new_col = cols[col_idx[icell]] # np.array([r(), r(), r()])
    # select thereshold based on breakpoint in values
    xx = np.sort(np.ndarray.flatten(b_eval[:,icell]));
    idx = np.min(np.where(np.diff(xx)>0.01))
    a_thr = (xx[idx] + xx[idx+1])/2

    #print('cell: %d, threshold %.3f' % (icell, a_thr))
    #a_thr = 0.1#np.percentile(np.abs(b_eval[:, icell]), 99.5)
    isu_cnt = 0
    for idimx in np.arange(dimx):
      for idimy in np.arange(dimy):
        icnt = icnt + 1

        if(np.abs(b_eval[icnt,icell]) > a_thr):
          isu_cnt += 1
          print('plotting cell: %d, su (%d, %d), weight: %.3f, su_#: %d'% (icell, idimx, idimy, b_eval[icnt,icell], isu_cnt))

          ww = np.zeros((40,80))
          ww[idimx*FLAGS.stride: idimx*FLAGS.stride + (2*window+1),
             idimy*FLAGS.stride:
             idimy*FLAGS.stride + (2*window+1)] =  b_eval[icnt, icell] * (
             np.reshape(wts[idimx, idimy, :], (2*window+1,2*window+1)))
          wts_sq = np.reshape(wts[idimx, idimy, :], (2*window+1,2*window+1))
          try:
            center,_ = plot_wts_gaussian_fits(wts_sq, shiftx=idimx*FLAGS.stride,
                                              shifty=idimy*FLAGS.stride,
                                              colors='k',
                                              levels = 0.5,
                                              toplot=True,
                                              alpha_fill=0,
                                              fill_color='k')

            dx = np.array([cell_centers[icell][0], center[0]])
            dy = np.array([cell_centers[icell][1], center[1]])
            dist_log = np.append(dist_log, np.sqrt(np.sum(dx**2) + np.sum(dy**2)))

            plt.plot(dx,dy, linewidth=2.5, color=new_col)
            #plt.plot(cell_centers[icell][0],cell_centers[icell][1], markersize=10, color=new_col)
            su_cell_cnt[idimx, idimy] +=1
            su_centersx[idimx, idimy] = center[0]
            su_centersy[idimx, idimy] = center[1]
            plt.hold(True)
            #plt.text(cell_centers[icell][0], cell_centers[icell][1], str(icell), fontsize=8, color='r')
            #plt.text(center[0], center[1], str([idimx,idimy,icnt]), fontsize=8, color='k')
            plt.hold(True)
          except:
            print('Failed fitting of subunit')
          #plt.title('cell %d, %d, wt: %0.3f' %(idimx, idimy, b_eval[icnt, icell]))
          plt.hold(True)

    cell_num_su = np.append(cell_num_su, isu_cnt) # count the number of subunits for each cell.

  # plot dots at centers of connected subunits
  toplot = True
  for idimx in np.arange(dimx):
    for idimy in np.arange(dimy):
      wts_sq = np.reshape(wts[idimx, idimy, :], (2*window+1,2*window+1))
      if su_cell_cnt[idimx, idimy]>0:
        col='k'
        msz = 6
        fill_col='k'
        if su_cell_cnt[idimx, idimy]>1:
          col = 'k'
          msz=6
          fill_col='b'
        center,_ = plot_wts_gaussian_fits(wts_sq, shiftx=idimx*FLAGS.stride,
                                              shifty=idimy*FLAGS.stride,
                                              colors='k',
                                              levels = 0.5,
                                              toplot=toplot,
                                              alpha_fill=0.4,
                                              fill_color=fill_col)
        plt.plot(su_centersx[idimx,idimy], su_centersy[idimx,idimy], '.', markersize=msz, color=col)

  plt.axis('off')
  ax.set_yticks([])
  ax.set_xticks([])
  plt.axis('Image')

  plt.savefig('coarse_subunits.pdf', facecolor=fig.get_facecolor(), transparent=True)
  plt.show()
  plt.draw()

  # plot the histogram for number of subunits across cells.
  plt.ion()
  plt.hist(cell_num_su)
  plt.title('# subunits accross cells')
  plt.show()
  plt.draw()

'''
  ## study convolutional properties of subunits
  # convolved mother cell
  centers_convolved_mother = np.zeros((0,2))
  plt.figure()
  icnt=-1
    for idimx in np.arange(dimx):
      print(idimx)
      for idimy in np.arange(dimy):
        icnt = icnt + 1
        wts_sq = np.reshape(w_mot, (2*window+1,2*window+1))
        try:
          center, sigma = plot_wts_gaussian_fits(wts_sq, shiftx=idimx*FLAGS.stride,
                                              shifty=idimy*FLAGS.stride,
                                              colors='k',
                                              levels = 0.5,
                                              toplot=True,
                                              alpha_fill=0.1,
                                              fill_color='k')
          centers_convolved_mother = np.append(centers_convolved_mother,
                                     np.expand_dims(np.array(center),0), 0)
          sigma_mother = np.sqrt(np.abs(sigma[0]*sigma[1]))
        except:s
          print('Failed to draw a subunit @ (%d, %d)' % (idimx, idimy))
        plt.axis('Image')


  # Plot all subunits (w_delta + w_mother)
  plt.ion()
  centers_all_su = np.zeros((0,2))
  sigma_all_su = np.array([])
  # plot all subunits and compute their center positions
  plt.figure()
  icnt=-1
  for idimx in np.arange(dimx):
    print(idimx)
    for idimy in np.arange(dimy):
      icnt = icnt + 1
      wts_sq = np.reshape(wts[idimx, idimy, :], (2*window+1,2*window+1))
      try:
        center, sigma = plot_wts_gaussian_fits(wts_sq, shiftx=idimx*FLAGS.stride,
                                              shifty=idimy*FLAGS.stride,
                                              colors='k',
                                              levels = 0.5,
                                              toplot=True,
                                              alpha_fill=0.1,
                                              fill_color='k')
        centers_all_su = np.append(centers_all_su,
                                     np.expand_dims(np.array(center),0), 0)
        sigma_all_su = np.append(sigma_all_su, np.sqrt(np.abs(sigma[0]*sigma[1])))
      except:
        print('Failed to draw a subunit @ (%d, %d)' % (idimx, idimy))
      plt.axis('Image')


  # show strongly connected subunits
  plt.figure()
  isu_cnt=0
  dist_log=np.array([])
  su_select = np.zeros(b_eval.shape[0])
  for icell_cnt, icell in enumerate(np.arange(b_eval.shape[1])):
    if icell==30 or icell==56: # has low number of spikes = (5000 or 8000)
      continue;

    if icell==43: # not well spike sorted - has bad STA in vision
      continue;
    if icell==105: # high # spikes, STA ok, but connects to a far away su - remove it!
      continue;

    icnt = -1

    # select thereshold based on breakpoint in values
    xx = np.sort(np.ndarray.flatten(b_eval[:,icell]));
    idx = np.min(np.where(np.diff(xx)>0.01))
    a_thr = (xx[idx] + xx[idx+1])/2

    #print('cell: %d, threshold %.3f' % (icell, a_thr))
    #a_thr = 0.1#np.percentile(np.abs(b_eval[:, icell]), 99.5)
    for idimx in np.arange(dimx):
      for idimy in np.arange(dimy):
        icnt = icnt + 1
        if(np.abs(b_eval[icnt,icell]) > a_thr):
          su_select[icnt]+=1
          isu_cnt += 1
          print('plotting cell: %d, su (%d, %d), weight: %.3f, su_#: %d'% (icell, idimx, idimy, b_eval[icnt,icell], isu_cnt))

          ww = np.zeros((40,80))
          ww[idimx*FLAGS.stride: idimx*FLAGS.stride + (2*window+1),
             idimy*FLAGS.stride:
             idimy*FLAGS.stride + (2*window+1)] =  b_eval[icnt, icell] * (
             np.reshape(wts[idimx, idimy, :], (2*window+1,2*window+1)))
          wts_sq = np.reshape(wts[idimx, idimy, :], (2*window+1,2*window+1))
          try:
            center = plot_wts_gaussian_fits(wts_sq, shiftx=idimx*FLAGS.stride,
                                            shifty=idimy*FLAGS.stride,
                                            colors='k',
                                            levels = 0.5,
                                            toplot=True,
                                            alpha_fill=0.2,
                                            fill_color='k')
          except:
            print('Failed subunit plotting')
        plt.show()


  # plot nearest neighbor distribution of convolved mother subunit
  # data in centers_convolved_mother
  n_su = centers_convolved_mother.shape[0]
  nn_dist_mother = np.zeros(n_su)
  for isu in range(n_su):
    print(isu)
    idx = np.array(np.ones(n_su), dtype='bool')
    idx[isu]=False
    distances = centers_convolved_mother[idx ,:] - centers_convolved_mother[isu, :]
    nn_dist_mother[isu] = np.linalg.norm(distances, axis=1).min()/sigma_mother
  nn_dist_all_mother = nn_dist_mother


  # plot nearest neighbour distribution of all subunits
  # data in centers_all_su
  n_su = centers_all_su.shape[0]
  nn_dist_su = np.zeros(n_su)
  for isu in range(n_su):
    print(isu)
    idx = np.array(np.ones(n_su), dtype='bool')
    idx[isu]=False
    distances = np.linalg.norm(centers_all_su[idx ,:] - centers_all_su[isu, :], axis=1)/np.sqrt(sigma_all_su[idx]*sigma_all_su[isu])
    nn_dist_su[isu] = distances.min()
  nn_dist_all_su = nn_dist_su


  # selects strongly connected subunits
  # su_select
  n_su = centers_all_su.shape[0]
  nn_dist_su = np.zeros(n_su)
  for isu in range(n_su):
    print(isu)
    idx = np.array(np.ones(n_su), dtype='bool')
    idx[isu]=False
    idx = np.logical_and(idx, su_select>0)
    distances = np.linalg.norm(centers_all_su[idx ,:] - centers_all_su[isu, :], axis=1)/np.sqrt(sigma_all_su[idx]*sigma_all_su[isu])
    nn_dist_su[isu] = distances.min()
    if su_select[isu]==0:
      nn_dist_su[isu] = np.nan
  nn_dist_strong_su = nn_dist_su

  nbins=20
  n, bins, patches = plt.hist(nn_dist_all_mother[np.logical_not(np.isnan(nn_dist_all_mother))],nbins, alpha=0.5)
  plt.hold(True)
  #for item in patches:
  #  item.set_height(item.get_height()/sum(n))
  n, bins, patches = plt.hist(nn_dist_all_su[np.logical_not(np.isnan(nn_dist_su))], nbins,  alpha=0.5)
  plt.hold(True)
  #for item in patches:
  #  item.set_height(item.get_height()/sum(n))
  n, bins, patches = plt.hist(nn_dist_strong_su[np.logical_not(np.isnan(nn_dist_su))], nbins, alpha=0.5)
  #for item in patches:
  #  item.set_height(item.get_height()/sum(n))
  plt.legend(['mother convolved', 'All subunits', 'Strong subunits'])
  #plt.legend(['All subunits', 'Strong subunits'])
  plt.title('Nearest neighbor distances (in pixels)')
  plt.xlabel('Pixels')
  plt.ylabel('% of subunits')
  plt.show()

def plot_su_gaussian(b_eval, wts, dimx, dimy, stas):
# plot contour over fitted subunits
    window = FLAGS.window
    ww_sum = np.zeros((40, 80))
    ifig = 0
    plt.figure()
    for icell_cnt, icell in enumerate(np.arange(b_eval.shape[1])):
      icnt = -1

      # select thereshold based on breakpoint in values
      xx = np.sort(np.ndarray.flatten(b_eval[:,icell]));
      idx = np.min(np.where(np.diff(xx)>0.01))
      a_thr = (xx[idx] + xx[idx+1])/2
      print('cell: %d, threshold %.3f' % (icell, a_thr))
      #a_thr = 0.1#np.percentile(np.abs(b_eval[:, icell]), 99.5)
      for idimx in np.arange(dimx):
        for idimy in np.arange(dimy):
          icnt = icnt + 1
          if(np.abs(b_eval[icnt,icell]) > a_thr):
            print('plotting:', icell, idimx, idimy)
            ww = np.zeros((40,80))
            ww[idimx*FLAGS.stride: idimx*FLAGS.stride + (2*window+1),
               idimy*FLAGS.stride:
               idimy*FLAGS.stride + (2*window+1)] =  b_eval[icnt, icell] * (
                np.reshape(wts[idimx, idimy, :], (2*window+1,2*window+1)))
            wts_sq = np.reshape(wts[idimx, idimy, :], (2*window+1,2*window+1))
            try:
              plot_wts_gaussian_fits(wts_sq, shiftx=idimx*FLAGS.stride,
                                     shifty=idimy*FLAGS.stride, colors='k',
                                     levels = 0.5, alpha_fill=0.2,
                                     fill_color='k')
            except:
              print('Failed fitting of subunit')
            #plt.title('cell %d, %d, wt: %0.3f' %(idimx, idimy, b_eval[icnt, icell]))
            plt.hold(True)


    for icell_cnt, icell in enumerate(np.arange(b_eval.shape[1])):
      try:
        plot_wts_gaussian_fits(np.reshape(stas[:, icell], (40,80)), colors='r',
                               levels = 0.8 , alpha_fill=0.2, fill_color='r')
        print('cell %d done' % icell)
      except:
        print('Failed fitting of STA')
      plt.hold(True)
    plt.axis('Image')
    plt.show()
    plt.draw()

    from IPython.terminal.embed import InteractiveShellEmbed
    ipshell = InteractiveShellEmbed()
    ipshell()

            #plt.imshow(ww, interpolation='nearest', cmap='gray')
            #plt.hold(True)
            #r = lambda: random.randint(0,255)
            #rgbl = [255, 128, 0]
            #random.shuffle(rgbl)
            #print(rgbl)


    plt.show()
    plt.draw()

    plt.figure()
    plt.imshow(ww_sum, interpolation='nearest', cmap='gray')
    plt.show()
    plt.draw()    
'''


def plot_wts_gaussian_fits(zobs, shiftx=0, shifty=0, colors='r', levels=1, fill_color='w', alpha_fill=0.0, toplot=True):

  #define model function and pass independant variables x and y as a list
  def gauss2d((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

  # upsample the subunit image - gives better fits that way
  scale=10
  zobs = np.repeat(np.repeat(zobs,scale,1),scale,0)
  shiftx *= scale
  shifty *= scale

  dimx, dimy= zobs.shape
  zobs = np.ndarray.flatten(zobs)
  x = np.repeat(np.expand_dims(np.arange(dimx), 1),dimy, 1);
  y = np.repeat(np.expand_dims(np.arange(dimy), 0),dimx, 0);
  x = np.ndarray.flatten(x)
  y = np.ndarray.flatten(y)
  xy = [np.ndarray.flatten(x),np.ndarray.flatten(y)]
  import scipy.optimize as opt
  i = zobs.argmax()
  guess = [1, x[i], y[i], 1, 1, 1]
  pred_params, uncert_cov = opt.curve_fit(gauss2d, xy, zobs, p0=guess)

  x = np.repeat(np.expand_dims(np.arange(0,dimx,1), 1),dimy, 1);
  y =np.repeat(np.expand_dims(np.arange(0, dimy,1), 0),dimx, 0);
  xy = [np.ndarray.flatten(x),np.ndarray.flatten(y)]
  zpred = gauss2d(xy, *pred_params)
  max_val = gauss2d([pred_params[1], pred_params[2]], *pred_params)

  zpred_bool = np.double(zpred < max_val*np.exp(-(levels**2)/2)**2)

  if toplot:
    #plt.contour(np.reshape(zpred ,(50,50)),1)
    plt.contour(y+shifty, x+shiftx, np.reshape(zpred_bool,(dimx, dimy)) ,1,linewidth=3,alpha=0.6, colors=colors);
    plt.hold(True)
    zpred_bool[zpred_bool==1] = np.nan
    plt.contourf(y+shifty,x+shiftx, np.reshape(zpred_bool,(dimx, dimy)) ,1, linewidth=0,
                 colors=(fill_color,'w'), alpha=alpha_fill)
    plt.hold(True)

  center = [pred_params[2]+shifty, pred_params[1]+shiftx]
  sigmas = [pred_params[4], pred_params[3]]
  return center, sigmas

if __name__ == '__main__':
  app.run()

