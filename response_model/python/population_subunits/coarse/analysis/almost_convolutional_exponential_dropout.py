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
"""Create models for computing firing rates of neuronal populations for a
given stimulus with almost convolutional structure, exponential NL and
optimize for few cells at a time.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import collections
import tensorflow as tf
from absl import app
from absl import flags
from absl import gfile
import cPickle as pickle
import numpy as np
import random

from retina.response_model.python.population_subunits.coarse.analysis import almost_convolutional

class AlmostConvolutionalExponentialDropout(almost_convolutional.AlmostConvolutionalModel):
  # Model firing rate for a cell population by almost convolutional subunits.

  def __init__(self, loss_string, stim, resp, short_filename, window=2,
               stride=1, lam_w=0, step_sz=1, n_cells=107, task_id=0):
    '''
    firing rate for cell c: lam_c = a_sfm_c'.relu(w.x + bias_su) + bias_cell,
    x: stimulus, lam_c: firing rate of cell
    bias_c and bias_su : cell and subunit bias
    a_sfm_c = softmax(a) : so a cell cannot be connected to
    all subunits equally well.

    where w_i are over a small window which are
    convolutionally related with each other.
    w_i = w_mother + w_del_i,
    where w_mother is common accross all 'windows' and
    w_del is different for different windows.

    stim, resp: the stimulus and response data tensors
    short_filename: filename to store results
    window: (2*window +1) is the convolutional window size
    stride: stride for convolutions
    lam_w : regularizing modification weights
    step_sz : step size for SGD
    n_cells : total number of cells in response tensor.
    '''
    # add model specific names to filename
    short_filename = ('model=' + 'almost_convolutional_exponential_dropout' + '_window=' +
                     str(window) + '_stride=' + str(stride) +
                    '_lam_w=' + str(lam_w) + short_filename)

    # convolution_parameters
    model_params = collections.namedtuple("model_params",
                                          ["mask_tf", "dimx", "dimy",
                                           "n_pix", "window", "stride",
                                           "n_cells"])
    mask_tf, dimx, dimy, n_pix = almost_convolutional.get_windows(window, stride)
    model_pars = model_params(mask_tf, dimx, dimy, n_pix,
                              window, stride, n_cells)

    # variables
    model_vars = self.build_variables(model_pars)

    # get firing rate
    lam_normalized, su_act_softmax, su_act, lam = self.build_firing_rate(model_vars,
                                                            model_pars,
                                                            stim)

    # get loss according to specification
    #if not loss_string == 'conditional_poisson':
    #  print(loss_string)
    #  raise ValueError('Inconsistent loss type and model')
    if loss_string == 'poisson':
      fraction=0.2  # select only 20% of the cells
      select_cells = tf.random_uniform([model_pars.n_cells]) > (1-fraction)
      print(select_cells)
      lam_select_cells = tf.transpose(tf.boolean_mask(tf.transpose(lam), select_cells))
      resp_select_cells = tf.transpose(tf.boolean_mask(tf.transpose(resp), select_cells))
      loss_unregularized = tf.reduce_mean(lam_select_cells/120. -
                                        resp_select_cells*tf.log(lam_select_cells))  # poisson loss

    if loss_string == 'conditional_poisson' :
      loss_unregularized =  -tf.reduce_mean(resp*tf.log(lam_normalized))

    # regularization  keep 'delta' weights small
    regularization = lam_w * tf.reduce_sum(tf.nn.l2_loss(model_vars.w_del))
    loss = loss_unregularized + regularization  # add regularization
    gradient_update = tf.train.AdagradOptimizer(step_sz).minimize(loss)

    #with tf.control_dependencies([gradient_update]):
    #  scale_biases = tf.reduce_sum(resp,0)/tf.reduce_sum(lam,0)
    #  bias_cell_su_scale = tf.assign(model_vars.bias_cell_su,
    #                                 model_vars.bias_cell_su -
    #                                 tf.log(scale_biases))

    # make a combined model update op
    model_update = gradient_update #tf.group(gradient_update, bias_cell_su_scale)

    # make model probes
    model_probes = collections.namedtuple("model_probes",
                                          ["su_act_softmax", "su_act", "lam_normalized",
                                           "loss",
                                           "loss_unregularized",
                                           "stim",
                                           "resp",
                                           "lam",
                                           "select_cells"])

    model_prb = model_probes(su_act_softmax, su_act, lam_normalized, loss,
                             loss_unregularized, stim, resp, lam, select_cells)

    self.stim = stim
    self.resp = resp
    self.params = model_pars
    self.update = model_update
    self.probes = model_prb
    self.variables = model_vars
    self.short_filename = short_filename
    self.build_summaries()

  def build_variables(self, model_pars):

    # get convolutional windows
    dimx = model_pars.dimx
    dimy = model_pars.dimy
    n_pix = model_pars.n_pix
    window = model_pars.window
    n_cells  =model_pars.n_cells

    # build model variables
    w_mother = tf.Variable(np.array(1 + 0 * np.random.rand(2 * window + 1,
                                             2 * window + 1, 1, 1),
                                    dtype='float32'), name='w_mother')
    w_del = tf.Variable(np.array(0.5*np.random.randn(dimx, dimy, n_pix),
                                 dtype='float32'), name='w_del')

    # initialize bias_cell_su to 0. use initialize_b to
    bias_cell_su = tf.Variable(np.array(0.0*np.random.randn(1, dimx,
                                                                 dimy, n_cells),
                                        dtype='float32'), name='bias_cell')

    # collect model parameters
    model_variables = collections.namedtuple("model_variables",
                                             ["w_mother", "w_del",
                                              "bias_cell_su"])
    model_vars = model_variables(w_mother, w_del, bias_cell_su)

    return model_vars

  
  def build_firing_rate(self, model_vars, model_pars, stim):
    # now compute the firing rate and subunit activations
    # given stimulus-response and model parameters

    # get model parameters
    mask_tf = model_pars.mask_tf
    dimx = model_pars.dimx
    dimy = model_pars.dimy
    stride = model_pars.stride
    n_cells = model_pars.n_cells

    # get model variables
    w_mother = model_vars.w_mother
    w_del = model_vars.w_del
    bias_cell_su = model_vars.bias_cell_su

    stim4D = tf.expand_dims(tf.reshape(stim, (-1, 40, 80)), 3)
    stim_convolved = tf.reduce_sum(tf.nn.conv2d(stim4D, w_mother,
                                                strides=[1, stride, stride, 1],
                                                padding="VALID"), 3)
    stim_masked = tf.nn.conv2d(stim4D, mask_tf, strides=[1, stride, stride, 1],
                               padding="VALID")
    stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)
    # input from convolutional SU and delta SU
    su_act_raw = tf.expand_dims(stim_del + stim_convolved, 3)  # time x dimx x dimy x 1
    su_act = su_act_raw + bias_cell_su  # time x dimx x dimy x n_cells

    # softmax for each cell over time and subunits
    su_act_softmax = tf.reshape(tf.nn.softmax(tf.reshape(su_act,
                                                      [-1, n_cells]), dim=0),
                                [-1, dimx, dimy, n_cells])
    lam_normalized = tf.reduce_sum(tf.reduce_sum(su_act_softmax, 2), 1)

    # calculate actual firing rate
    lam = tf.reduce_sum(tf.reduce_sum(tf.exp(su_act), 2), 1)

    return lam_normalized, su_act_softmax, su_act, lam




  def initialize_variables(self, sess):
    ## Initialize variables or restore from previous fits
    sess.run(tf.group(tf.initialize_all_variables(),
                      tf.initialize_local_variables()))
    saver_var = tf.train.Saver(tf.all_variables(),
                               keep_checkpoint_every_n_hours=4)
    load_prev = False
    start_iter = 0
    try:
      # restore previous fits if they are available
      # - useful when programs are preempted frequently on .
      latest_filename = self.short_filename + '_latest_fn'
      restore_file = tf.train.latest_checkpoint(self.save_location,
                                                latest_filename)
      # restore previous iteration count and start from there.
      start_iter = int(restore_file.split('/')[-1].split('-')[-1])
      saver_var.restore(sess, restore_file) # restore variables
      load_prev = True
    except:
      tf.logging.info('No previous dataset')

    if load_prev:
      tf.logging.info('Previous results loaded from: ' + restore_file)
    else:
      self.initialize_b(sess)
      tf.logging.info('Variables initialized')

    writer = tf.summary.FileWriter(self.save_location + 'train', sess.graph)

    tf.logging.info('Loaded iteration: %d' % start_iter)

    self.saver_var = saver_var
    self.iter = start_iter
    self.writer = writer


  def initialize_b(self,sess, n_batches_init=20):
    # initialize b based on <yexp(kx)>


    tf.logging.info('initializing b_cell_su')
    # setup data threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tf.logging.info('threads started')

    resp_expanded = tf.expand_dims(tf.expand_dims(self.probes.resp, 1), 2)
    b_avg = tf.expand_dims(tf.reduce_mean(tf.mul(self.probes.su_act,
                                                        resp_expanded), 0), 0)

    b_initialize = np.zeros((1, self.params.dimx, self.params.dimy, self.params.n_cells))
    for ibatch in range(n_batches_init):
      print('init b: %d' % ibatch)
      b_initialize += sess.run(b_avg)
    b_initialize /= 1000*n_batches_init

    #b_max = np.max(np.reshape(b_initialize, [-1, self.params.n_cells]), axis=0)
    #mask = b_initialize > b_max*0.6
    #b_initial_masked =  - 40*(1-mask)
    #b_initial_masked_reshape = np.reshape(b_initial_masked, [1, self.params.dimx,
    #                                                       self.params.dimy,
    #                                                       self.params.n_cells])

    # from IPython.terminal.embed import InteractiveShellEmbed
    # ipshell = InteractiveShellEmbed()
    # ipshell()

    b_init_square = np.zeros((1, self.params.dimx, self.params.dimy, self.params.n_cells))
    for icell in np.arange(self.params.n_cells):
      ix, iy = np.where(b_initialize[0, :, :, icell] == np.max(np.ndarray.flatten(b_initialize[0, :, :, icell])))
      ix = int(ix)
      iy = int(iy)
      xx = -40*np.ones((self.params.dimx, self.params.dimy))
      xx[ix-5:ix+5, iy-5:iy+5] = 0
      b_init_square[0, :, :, icell] = xx

    b_init_tf = tf.assign(self.variables.bias_cell_su,
                          b_init_square.astype(np.float32))
    sess.run(b_init_tf)

    #coord.request_stop()
    #coord.join(threads)

    tf.logging.info('b_cell_su initialzed based on average activity')


