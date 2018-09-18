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
r"""Learn models for predicting population responses.

The WN stimulus is prefiltered in time using STA time course.
So, only spatial structure of the subunit is learnt

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import random

import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab
import matplotlib.pyplot as plt

from absl import app
from absl import flags
from absl import gfile

from retina.response_model.python.population_subunits.coarse.fitting import data_utils


# Flags for data locations.
flags.DEFINE_string('folder_name', 'experiment_tfrec',
                    'folder where to store all the data')

flags.DEFINE_string('save_location',
                    '/home/bhaishahster/',
                    'where to store logs and outputs?')

flags.DEFINE_string('data_location',
                    '/home/bhaishahster/data_breakdown/',
                    'where to take data from?')

flags.DEFINE_string('tmp_dir',
                    '/home/bhaishahster/data_breakdown/',
                    'temporary folder on machine for better I/O')

flags.DEFINE_integer('taskid', 0, 'Task ID')


# Flags for stochastic learning.
flags.DEFINE_integer('batchsz', 10000, 'batch size for training')
flags.DEFINE_integer('num_epochs', 40000, 'maximum number of iterations')
flags.DEFINE_float('step_sz', 10, 'step size for learning algorithm')
flags.DEFINE_float('learn', True,
                   'whether to learn a model, or analyse a fitted one')

# Random number generators initialized.
# Removes unneccessary data variabilities while comparing algorithms.
flags.DEFINE_integer('np_randseed', 23, 'numpy RNG seed')
flags.DEFINE_integer('randseed', 65, 'python RNG seed')

# Flags for model/loss specification.
flags.DEFINE_string('model_id', 'almost_convolutional',
                    'model for firing rate: almost_convolutional')
flags.DEFINE_string('loss_string', 'poisson',
                    'which loss to use: poisson or conditional_poisson')
flags.DEFINE_string('masked_stimulus', False,
                    'use all pixels or only those inside RF of selected cells?')
flags.DEFINE_string('chosen_cells', None,
                    'learn model for which cells? if None, learn of all cells')
flags.DEFINE_integer('n_cells', 107, 'number of cells in the dataset')

# Model specific terms.
# Useful for convolution-like models.
flags.DEFINE_integer('window', 2,
                     'size of window for each subunit in relu_window model')
flags.DEFINE_integer('stride', 1, 'stride for relu_window')
# Some models need regularization of parameters.
flags.DEFINE_float('lam_w', 0.0, 'sparsitiy regularization of w')


FLAGS = flags.FLAGS


def get_filename():
  """"Generate partial filename using FLAGS for running the code."""

  if FLAGS.chosen_cells is None:
    all_cells = True
  else:
    all_cells = False

  tf.logging.info('Save folder name: ' + str(FLAGS.folder_name) +
                  '\nmodel: ' + str(FLAGS.model_id) +
                  '\nmasked stimulus: ' + str(FLAGS.masked_stimulus) +
                  '\nall_cells? ' + str(all_cells) +
                  '\nbatch size ' + str(FLAGS.batchsz) +
                  '\nstep size ' + str(FLAGS.step_sz) +
                  '\ntaskid:' + str(FLAGS.taskid))

  # saving details
  short_filename = ('_masked_stim=' + str(FLAGS.masked_stimulus) +
                    '_all_cells='+ str(all_cells) +
                    '_loss='+ str(FLAGS.loss_string) +
                    '_batch_sz='+ str(FLAGS.batchsz) +
                    '_step_sz='+ str(FLAGS.step_sz) +
                    '_taskid=' + str(FLAGS.taskid))

  return short_filename


def main(unused_argv):

  # Set random seeds.
  np.random.seed(FLAGS.np_randseed)
  random.seed(FLAGS.randseed)

  # Set up dataset as a local tfrecords file.

  data = data_utils.CoarseDataUtils(FLAGS.data_location,
                                    np.int(FLAGS.batchsz),
                                    masked_stimulus=FLAGS.masked_stimulus,
                                    chosen_cells=FLAGS.chosen_cells,
                                    test_length=500)
  n_cells = data.response.shape[1]

  with tf.Session() as sess:

    # windowed convolutional model
    # get windows
    mask_tf, dimx, dimy, n_pix = get_windows(FLAGS.window, FLAGS.stride)


    # initialize variables
    w = tf.Variable(np.array(0.5 + 0.2*np.random.randn(dimx, dimy, n_pix), dtype='float32'), name='w')

    # bias_cell_su
    # initialize shape
    bias_cell_su = tf.Variable(np.array(0.0*np.random.randn(1, dimx,dimy, n_cells), dtype='float32'), name='bias_cell_su')
    # TODO(bhaishahster) initialize based on STA.
    
    # get stimulus and response
    stim = tf.placeholder(tf.float32) # time x stimulus_dimensions
    resp = tf.placeholder(tf.float32) # time x n_cells

    # get sigma
    sig_scalar = np.var(data.stimulus)
    sigma = tf.constant((sig_scalar*np.eye(n_pix)).astype(np.float32))

    # build activations
    stim4D = tf.expand_dims(tf.reshape(stim, (-1, 40, 80)), 3)
    stim_masked = tf.nn.conv2d(stim4D, mask_tf, strides=[1, FLAGS.stride, FLAGS.stride, 1], padding="VALID") # time x dimx x dimy x n_pix
    stim_del = tf.reduce_sum(tf.mul(stim_masked, w), 3)

    # input from convolutional SU and delta SU
    su_act = tf.expand_dims(stim_del, 3) + bias_cell_su  # time x dimx x dimy x n_cells

    # softmax for each cell over time and subunits
    su_act_t_s_c = tf.reshape(su_act, [-1, dimx*dimy, n_cells])
    su_act_tsc_sfm = tf.nn.softmax(su_act_t_s_c, dim=1) # time x subunits x n_cells
    su_act_tsc_sfm_reshape = tf.reshape(su_act_tsc_sfm , [-1, dimx, dimy, n_cells]) # time x dimx x dimy x n_cells

    response_5d = tf.expand_dims(tf.expand_dims(tf.expand_dims(resp, 1), 2), 3)     # time x 1    x 1    x 1     x n_cells
    su_act_tsc_sfm_5d = tf.expand_dims(su_act_tsc_sfm_reshape, axis=3)              # time x dimx x dimy x 1     x n_cells
    stim_masked_5d = tf.expand_dims(stim_masked, axis=4)                            # time x dimx x dimy x n_pix x 1

    su_weighted_resp = tf.mul(su_act_tsc_sfm_5d, response_5d) # time x dimx x dimy x 1 x n_cells
    numerator = tf.reduce_sum(tf.reduce_sum(tf.mul(su_weighted_resp, stim_masked_5d), 4), 0) # dimx x dimy x n_pix
    denominator = tf.reduce_sum(tf.reduce_sum(su_weighted_resp, 4), 0) # dimx x dimy x 1

    num_accumulate = tf.Variable(np.zeros((dimx, dimy, n_pix)).astype(np.float32), name='num_accumulate')
    den_accumulate = tf.Variable(np.zeros((dimx, dimy, 1)).astype(np.float32), name='den_accumulate')

    num_acc_reset = tf.assign(num_accumulate, 0*num_accumulate)
    den_acc_reset = tf.assign(den_accumulate, 0*den_accumulate)

    num_acc_update = tf.assign_add(num_accumulate, numerator)
    den_acc_update = tf.assign_add(den_accumulate, denominator)

    wts_new = num_accumulate/ den_accumulate
    update_wts = tf.assign(w, wts_new)

    # bias should be 1, dimx,dimy, n_cells
    bias_new_part = tf.expand_dims(tf.reduce_sum(tf.reduce_sum(su_weighted_resp, 3), 0), 0) # shape= 1, dimx, dimy, n_cells

    bias_term_accumulate = tf.Variable(np.zeros((1, dimx, dimy, n_cells)). astype(np.float32), name='bias_term_accumulate')
    num_sums = tf.Variable(np.float32(0), name='num_sums')

    num_sums_reset = tf.assign(num_sums, 0)
    bias_term_acc_reset = tf.assign(bias_term_accumulate, 0*bias_term_accumulate)

    bias_term_acc_update = tf.assign_add(bias_term_accumulate, bias_new_part)
    num_sums_update = tf.assign_add(num_sums, 1)

    w_sq = tf.reshape(w, [-1, n_pix])
    bias_new_term2 = tf.expand_dims(tf.reshape(tf.diag_part(0.5*tf.matmul(w_sq, tf.matmul(sigma, w_sq, transpose_b=True))), [dimx, dimy]), 2) # shape = dimx, dimx x 1

    bias_new = tf.log(bias_term_accumulate/num_sums) - bias_new_term2

    # 1 x dimx x dim y x 1
    update_bias = tf.assign(bias_cell_su, bias_new)



    # intialize
    sess.run(tf.global_variables_initializer())

    try :
      for iiter in range(1000):
        # reset sums to 0

        _, _, _, _ = sess.run([num_acc_reset, den_acc_reset, bias_term_acc_reset, num_sums_reset])
        print('accumulators reset')
        batchsz=100
        for itrain_chunk in range(int(216000/batchsz)): # batchsz = 1000
          # get next batch of data and store in feed_dict
          tidx = np.arange(itrain_chunk*batchsz, (itrain_chunk+1)*batchsz)
          feed_dict={stim:data.stimulus[tidx, :], resp: data.response[tidx, :]}
          # update accumulators
          print('accumulating chunk % iiter, %d' % (iiter, itrain_chunk))
          _, _, _, _ = sess.run([num_acc_update, den_acc_update, bias_term_acc_update, num_sums_update], feed_dict = feed_dict)

        # update variables
        _, _ = sess.run([update_wts, update_bias])
        print('parameters updated')
    except:
      from IPython.terminal.embed import InteractiveShellEmbed
      ipshell = InteractiveShellEmbed()
      ipshell()

    # train the model

  #

def get_windows(window, stride):
  """Get locations and arrangement of the convolutional windows.

  Args:
    window : (2*window+1) is the symmetrical convolutional window size
    stride : the stride between nearby convolutional windows

  Returns:
    mask_tf : Mask to identify each window.
    dimx : number of windows in x dimension
    dimy : number of windows in y dimension
    n_pix : number of pixels in each window
  """

  n_pix = (2* window + 1) ** 2  # number of pixels in the window
  w_mask = np.zeros((2 * window + 1, 2 * window + 1, 1, n_pix))
  icnt = 0

  # Make mask_tf: weight (dimx X dimy X npix) for convolutional layer,
  # where each layer is 1 for a particular pixel in window and 0 for others.
  # this is used for flattening the pixels in a window,
  # so that different weights could be applied to each window.
  for ix in range(2 * window + 1):
    for iy in range(2 * window + 1):
      w_mask[ix, iy, 0, icnt] = 1
      icnt += 1
  mask_tf = tf.constant(w_mask.astype(np.float32))

  # Number of windows in x and y dimensions.
  dimx = np.floor(1 + ((40 - (2 * window + 1))/stride)).astype('int')
  dimy = np.floor(1 + ((80 - (2 * window + 1))/stride)).astype('int')
  return mask_tf, dimx, dimy, n_pix


if __name__ == '__main__':
  app.run()
