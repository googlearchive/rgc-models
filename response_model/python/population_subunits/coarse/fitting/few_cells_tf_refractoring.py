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
""" Fit subunits for multiple cells simultaneously.
This script has the extensions of single cell models from earlier
as well as new population subunit models -
most notably the almost convolutional model - where each subunit is
summation of mother subunit and subunit specific modification.
"""


import sys
import os.path
import tensorflow as tf
from absl import app
from absl import flags
from absl import gfile

import cPickle as pickle
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import scipy.io as sio
from scipy import ndimage
import random

FLAGS = flags.FLAGS
# flags for data location
flags.DEFINE_string('folder_name', 'experiment4',
                    'folder where to store all the data')
flags.DEFINE_string('save_location',
                    '/home/bhaishahster/',
                    'where to store logs and outputs?');
flags.DEFINE_string('data_location',
                    '/home/bhaishahster/data_breakdown/',
                    'where to take data from?')
# flags for stochastic learning and loading data
# data is split and stored as small .mat files
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

# global stimulus variables
stim_train_part = np.array([])
resp_train_part = np.array([])
chunk_order = np.array([])
cells_choose = np.array([])
chosen_mask = np.array([])


def get_test_data():
  # the last chunk of data is test data
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
  window = FLAGS.window # 2*window +1 is the width and height of windows
  n_pix = (2* window + 1) ** 2 # number of pixels in the window
  w_mask = np.zeros((2 * window + 1, 2 * window + 1, 1, n_pix))
  icnt = 0

  # make mask_tf: weight (dimx X dimy X npix) for convolutional layer,
  # where each layer is 1 for a particular pixel in window and 0 for others.
  # This is used for flattening the pixels in a window,
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

  # set random seeds: when same algorithm run with different FLAGS,
  # the sequence of random data is same.
  np.random.seed(FLAGS.np_randseed)
  random.seed(FLAGS.randseed)
  # initial chunk order (will be re-shuffled everytime we go over a chunk)
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
  print('\ndataset summary loaded')

  # print parameters
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

  # filename for saving file
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
      b_init = np.array(0.000001*np.ones(n_cells)) # a very small positive bias needed to avoid log(0) in poisson loss
    else:
      b_init =  np.log((tot_spks_chosen_cells)/(216000. - tot_spks_chosen_cells)) # log-odds, a good initialization for some losses (like logistic)

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
      def proj(): # called after every training step - to project to parameter constraints
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
      vars_fit = [w, a] # which variables are learnt
      if not FLAGS.loss == 'poisson': # don't learn b for poisson loss
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
      mask_tf, dimx, dimy, n_pix = get_windows() # get convolutional windows

      # variables
      w = tf.Variable(np.array(0.1+ 0.05*np.random.rand(dimx, dimy, n_pix),dtype='float32'), name='w')
      a = tf.Variable(np.array(np.random.rand(dimx*dimy, n_cells),dtype='float32'), name='a')
      b = tf.Variable(np.array(b_init,dtype='float32'), name='b')
      vars_fit = [w, a] # which variables are learnt
      if not FLAGS.loss == 'poisson':  # don't learn b for poisson loss
        vars_fit = vars_fit + [b]

      # stimulus filtered with convolutional windows
      stim4D = tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
      stim_masked = tf.nn.conv2d(stim4D, mask_tf,
                                 strides=[1, FLAGS.stride, FLAGS.stride, 1],
                                 padding="VALID" )
      stim_wts = tf.nn.relu(tf.reduce_sum(tf.mul(stim_masked, w), 3))
      # get firing rate
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

      # we know a>0 and for poisson loss, b>0
      # for poisson loss: small b added to prevent lam_c going to 0

      # filename
      short_filename = ('model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) +
                        '_lam_w=' + str(FLAGS.lam_w) + short_filename )
      mask_tf, dimx, dimy, n_pix = get_windows()

      # variables
      w_del = tf.Variable(np.array(  0.05*np.random.randn(dimx, dimy, n_pix),
                                   dtype='float32'), name='w_del')
      w_mother = tf.Variable(np.array( np.ones((2 * FLAGS.window + 1,
                                                2 * FLAGS.window + 1, 1, 1)),
                                      dtype='float32'), name='w_mother')
      a = tf.Variable(np.array(np.random.rand(dimx*dimy, n_cells),
                               dtype='float32'), name='a')
      b = tf.Variable(np.array(b_init,dtype='float32'), name='b')
      vars_fit = [w_mother, w_del, a] # which variables to learn
      if not FLAGS.loss == 'poisson':
        vars_fit = vars_fit + [b]

      #  stimulus filtered with convolutional windows
      stim4D = tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
      stim_convolved = tf.reduce_sum( tf.nn.conv2d(stim4D,
                                                   w_mother,
                                                   strides=[1, FLAGS.stride,
                                                            FLAGS.stride, 1],
                                                   padding="VALID"),3)
      stim_masked = tf.nn.conv2d(stim4D,
                                 mask_tf,
                                 strides=[1, FLAGS.stride, FLAGS.stride, 1],
                                 padding="VALID" )
      stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)

      # activation of differnet subunits
      su_act = tf.nn.relu(stim_del + stim_convolved)

      # get firing rate
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
      w_mother = tf.Variable(np.array( np.ones((2 * FLAGS.window + 1,
                                                2 * FLAGS.window + 1, 1, 1)),
                                      dtype='float32'), name='w_mother')
      a = tf.Variable(np.array(np.random.randn(dimx*dimy, n_cells),
                               dtype='float32'), name='a')
      a_sfm = tf.transpose(tf.nn.softmax(tf.transpose(a)))
      b = tf.Variable(np.array(b_init,dtype='float32'), name='b')
      vars_fit = [w_mother, w_del, a] # which variables to fit
      if not FLAGS.loss == 'poisson':
        vars_fit = vars_fit + [b]

      # stimulus filtered with convolutional windows
      stim4D = tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
      stim_convolved = tf.reduce_sum(tf.nn.conv2d(stim4D,
                                                  w_mother,
                                                  strides=[1, FLAGS.stride,
                                                           FLAGS.stride, 1],
                                                  padding="VALID"),3)
      stim_masked = tf.nn.conv2d(stim4D, mask_tf,
                                 strides=[1, FLAGS.stride, FLAGS.stride, 1],
                                 padding="VALID" )
      stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)

      # activation of differnet subunits
      su_act = tf.nn.relu(stim_del + stim_convolved)

      # get firing rate
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
      # get windows
      mask_tf, dimx, dimy, n_pix = get_windows()

      # declare variables
      w_del = tf.Variable(np.array(  0.05*np.random.randn(dimx, dimy, n_pix),
                                   dtype='float32'), name='w_del')
      w_mother = tf.Variable(np.array( np.ones((2 * FLAGS.window + 1,
                                                2 * FLAGS.window + 1, 1, 1)),
                                      dtype='float32'), name='w_mother')
      a = tf.Variable(np.array(np.random.randn(dimx*dimy, n_cells),
                               dtype='float32'), name='a')
      a_sfm = tf.transpose(tf.nn.softmax(tf.transpose(a)))
      b = tf.Variable(np.array(b_init,dtype='float32'), name='b')
      vars_fit = [w_mother, w_del, a]
      if not FLAGS.loss == 'poisson':
        vars_fit = vars_fit + [b]

      # filter stimulus
      stim4D = tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
      stim_convolved = tf.reduce_sum( tf.nn.conv2d(stim4D,
                                                   w_mother,
                                                   strides=[1, FLAGS.stride,
                                                            FLAGS.stride, 1],
                                                   padding="VALID"),3)
      stim_masked = tf.nn.conv2d(stim4D, mask_tf,
                                 strides=[1, FLAGS.stride, FLAGS.stride, 1],
                                 padding="VALID" )
      stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)

      # get subunit activation
      su_act = tf.nn.relu(stim_del + stim_convolved)

      # get cell firing rates
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
      loss_inter = tf.reduce_sum(tf.nn.relu(1 -2 * (resp - 0.5)*lam)) / data_len

    loss = loss_inter + regularization # add regularization to get final loss function

    # training consists of calling training()
    # which performs a train step and
    # project parameters to model specific constraints using proj()
    train_step = tf.train.AdagradOptimizer(FLAGS.step_sz).minimize(loss,
                                                                   var_list=
                                                                   vars_fit)
    def training(inp_dict):
      sess.run(train_step, feed_dict=inp_dict) # one step of gradient descent
      proj() # model specific projection operations

    # evaluate loss on given data.
    def get_loss(inp_dict):
      ls = sess.run(loss,feed_dict = inp_dict)
      return ls


    # saving details
    # make a folder with name derived from parameters of the algorithm
    # - it saves checkpoint files and summaries used in tensorboard
    parent_folder = FLAGS.save_location + FLAGS.folder_name + '/'
    # make folder if it does not exist
    if not gfile.IsDirectory(parent_folder):
      gfile.MkDir(parent_folder)
    FLAGS.save_location = parent_folder + short_filename + '/'
    if not gfile.IsDirectory(FLAGS.save_location):
      gfile.MkDir(FLAGS.save_location)
    save_filename = FLAGS.save_location + short_filename



    # create summary writers
    # create histogram summary for all parameters which are learnt
    for ivar in vars_fit:
      tf.histogram_summary(ivar.name, ivar)
    # loss summary
    l_summary = tf.scalar_summary('loss',loss)
    # loss without regularization summary
    l_inter_summary = tf.scalar_summary('loss_inter',loss_inter)
    # Merge all the summary writer ops into one op (this way,
    # calling one op stores all summaries)
    merged = tf.merge_all_summaries()
    # training and testing has separate summary writers
    train_writer = tf.train.SummaryWriter(FLAGS.save_location + 'train',
                                      sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.save_location + 'test')



    ## Fitting procedure
    print('Start fitting')
    sess.run(tf.initialize_all_variables())
    saver_var = tf.train.Saver(tf.all_variables(),
                               keep_checkpoint_every_n_hours=0.05)
    load_prev = False
    start_iter=0
    try:
      # restore previous fits if they are available
      # - useful when programs are preempted frequently on .
      latest_filename = short_filename + '_latest_fn'
      restore_file = tf.train.latest_checkpoint(FLAGS.save_location,
                                                latest_filename)
      # restore previous iteration count and start from there.
      start_iter = int(restore_file.split('/')[-1].split('-')[-1])
      saver_var.restore(sess, restore_file) # restore variables
      load_prev = True
    except:
      print('No previous dataset')

    if load_prev:
      print('Previous results loaded')
    else:
      print('Variables initialized')


    # Finally, do fitting
    icnt = 0
    # get test data and make test dictionary
    stim_test,resp_test,test_length = get_test_data()
    fd_test = {stim: stim_test,
               resp: resp_test,
               data_len: test_length}

    for istep in np.arange(start_iter,400000):
      print(istep)
      # get training data and make test dictionary
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
        latest_filename = short_filename + '_latest_fn'
        saver_var.save(sess, save_filename, global_step=istep,
                       latest_filename = latest_filename)

        # add training summary
        summary = sess.run(merged, feed_dict=fd_train)
        train_writer.add_summary(summary,istep)

        # add testing summary
        summary = sess.run(merged, feed_dict=fd_test)
        test_writer.add_summary(summary,istep)
        print(istep, ls_train, ls_test)

      icnt += FLAGS.batchsz
      if icnt > 216000-1000:
        icnt = 0
        tms = np.random.permutation(np.arange(216000-1000))


if __name__ == '__main__':
  app.run()

