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

class AlmostConvolutionalMELDropoutOnlyWdelta(almost_convolutional.AlmostConvolutionalModel):
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
    short_filename = ('model=' + 'almost_convolutional_mel_dropout_only_wdelta' + '_window=' +
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
    # get stimulus points where response is non-zero

    model_update, select_cells, update_bias_only = self.update_model(model_vars,
                                                   model_pars,
                                                   stim, resp, step_sz)

    '''
    # get loss according to specification
    #if not loss_string == 'conditional_poisson':
    #  print(loss_string)
    #  raise ValueError('Inconsistent loss type and model')
    if loss_string == 'poisson':

      # resp_select_cells = tf.transpose(tf.boolean_mask(tf.transpose(resp), select_cells))
      resp_select_cells = tf.transpose(tf.gather(tf.transpose(resp), select_cells))
      loss_unregularized = tf.reduce_mean(lam_select_cells/120. -
                                        resp_select_cells*tf.log(lam_select_cells))  # poisson loss

    if loss_string == 'conditional_poisson' :
      raise NotImplementedError()

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

    '''
    # get loss
    loss, lam = self.get_loss(model_vars, model_pars, stim, resp)

    # make model probes
    model_probes = collections.namedtuple("model_probes",
                                          ["loss",
                                           "loss_unregularized",
                                           "stim",
                                           "resp",
                                           "select_cells", "lam", "update_bias_only"])

    model_prb = model_probes( loss, loss ,stim, resp, select_cells, lam, update_bias_only)

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

    # stimulus stimulus_var
    stim_var = tf.Variable(0.18**2, name ='stimulus_var')

    # count cells
    cell_cnt = tf.Variable(np.zeros(n_cells).astype(np.float32), name='cell_count')

    # add dummy w_mother
    w_mother = tf.constant(np.array(0 + 0 * np.random.rand(2 * window + 1,
                                             2 * window + 1, 1, 1)).astype(np.float32), name='w_mother')
    # build model variables
    w_del = tf.Variable(np.array(0.5+0.25*np.random.randn(dimx, dimy, n_pix),
                                 dtype='float32'), name='w_del')

    # initialize bias_cell_su to 0. use initialize_b to
    #bias_cell_su = tf.Variable(np.array(0.0*np.random.randn(1, dimx, dimy,
    #                                                        n_cells),
    #                                    dtype='float32'), name='bias_cell')
    bias_cell_su = tf.Variable(np.array(0.0*np.random.randn(n_cells, dimx, dimy),
                                        dtype='float32'), name='bias_cell')
    # collect model parameters
    model_variables = collections.namedtuple("model_variables",
                                             ["w_del","w_mother",
                                              "bias_cell_su", "stim_var", "cell_cnt"])
    model_vars = model_variables(w_del, w_mother, bias_cell_su, stim_var, cell_cnt)

    return model_vars

  def get_loss(self, model_vars, model_pars, stim, resp):
    # now compute the firing rate and subunit activations
    # given stimulus-response and model parameters

    # get model parameters
    mask_tf = model_pars.mask_tf
    dimx = model_pars.dimx
    dimy = model_pars.dimy
    stride = model_pars.stride
    n_cells = model_pars.n_cells

    # get model variables
    w_del = model_vars.w_del
    bias_cell_su = model_vars.bias_cell_su
    stim_var = model_vars.stim_var
    bias_cell_su_reshape = tf.transpose(bias_cell_su, [1, 2, 0])

    stim4D = tf.expand_dims(tf.reshape(stim, (-1, 40, 80)), 3)
    stim_masked = tf.nn.conv2d(stim4D, mask_tf, strides=[1, stride, stride, 1],
                               padding="VALID")
    stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)
    # input from convolutional SU and delta SU
    su_act_raw = tf.expand_dims(stim_del , 3)  # time x dimx x dimy x 1
    su_act = su_act_raw + bias_cell_su_reshape  # time x dimx x dimy x n_cells


    # calculate actual firing rate
    lam = tf.reduce_sum(tf.reduce_sum(tf.exp(su_act), 2), 1)
    loss_unregularized = tf.reduce_mean(lam - resp*tf.log(lam))  # poisson loss

    '''
    # select cells
    cells_tf = tf.constant(np.arange(model_pars.n_cells).astype(np.int))
    cells_shuffled = tf.random_shuffle(cells_tf)
    n_cells_iter = np.floor(1).astype(np.int)
    print('Number of cells in each iteration is %d ' % n_cells_iter)
    select_cells =  tf.range(n_cells) # [43] #cells_shuffled[0: n_cells_iter] # TODO(bhaishahster): select one cell smartly
    resp_cell = tf.transpose(tf.gather(tf.transpose(resp), select_cells))

    bias_cell_su_selected = tf.gather(bias_cell_su, select_cells)
    print(select_cells)

    stim4D = tf.expand_dims(tf.reshape(stim, (-1, 40, 80)), 3)
    stim_masked = tf.nn.conv2d(stim4D, mask_tf, strides=[1, stride, stride, 1],
                               padding="VALID")
    print(stim_masked)

    stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)
    # input from convolutional SU and delta SU
    su_act_raw = stim_del  # time x dimx x dimy
    su_act_select_cells = su_act_raw + bias_cell_su_selected  # time x dimx x dimy

    # calculate actual firing rate
    lam_select_cells = tf.reduce_sum(tf.reduce_sum(tf.exp(su_act_select_cells), 2), 1)

    loss_unregularized = tf.reduce_mean(lam_select_cells - resp_cell*tf.log(lam_select_cells))  # poisson loss
    '''

    return loss_unregularized, lam

  def update_model(self, model_vars, model_pars, stim, resp, step_sz):
    # now compute the firing rate and subunit activations
    # given stimulus-response and model parameters

    # get model parameters
    mask_tf = model_pars.mask_tf
    dimx = model_pars.dimx
    dimy = model_pars.dimy
    stride = model_pars.stride
    n_cells = model_pars.n_cells

    # get model variables
    w_del = model_vars.w_del
    bias_cell_su = model_vars.bias_cell_su
    stim_var = model_vars.stim_var
    cell_cnt = model_vars.cell_cnt


    # select cells
    '''
    # randomly select a cell
    cells_tf = tf.constant(np.arange(model_pars.n_cells).astype(np.int))
    cells_shuffled = tf.random_shuffle(cells_tf)
    n_cells_iter = np.floor(1).astype(np.int)
    print('Number of cells in each iteration is %d ' % n_cells_iter)
    select_cells = cells_shuffled[0: n_cells_iter] # tf.constant(np.array([23]).astype(np.int32)) # cells_shuffled[0: n_cells_iter]  # TODO(bhaishahster): select one cell smartly
    '''

    # circulate through cells
    cell_counter = tf.Variable([0], name='cell_counter')
    select_cells  = tf.mod(cell_counter, n_cells)


    resp_cell = tf.transpose(tf.gather(tf.transpose(resp), select_cells))
    T_full_resp = tf.reduce_sum(resp_cell*0 + 1)
    resp_pos_loc = tf.where(resp_cell>0)[:,0]
    resp_pos_loc_shuff = tf.random_shuffle(resp_pos_loc)
    n_resp = tf.minimum(tf.constant(np.array(10000).astype(np.int32)), tf.to_int32(tf.reduce_sum(0*resp_pos_loc_shuff+1)))
    T = T_full_resp * tf.to_float(n_resp) / tf.to_float(tf.reduce_sum(0*resp_pos_loc_shuff+1))

    tf.logging.info(n_resp)
    tf.logging.info(resp_pos_loc_shuff)
    resp_pos_loc_shuff_selec = tf.gather(resp_pos_loc_shuff, tf.range(0, n_resp-1))
    tf.logging.info(resp_pos_loc_shuff_selec)
    resp_pos = tf.gather(resp_cell, resp_pos_loc_shuff_selec)
    stim_pos = tf.gather(stim, resp_pos_loc_shuff_selec)

    #from IPython.terminal.embed import InteractiveShellEmbed
    #ipshell = InteractiveShellEmbed()
    #ipshell()

    bias_cell_su_selected = tf.gather(bias_cell_su, select_cells)
    #bias_cell_su_selected = tf.transpose(tf.expand_dims(tf.gather(tf.transpose(bias_cell_su, [3, 0, 1, 2]), select_cells),0), [1, 2, 3, 0]) # when 1 cell selected
    #select_cells = tf.random_uniform([model_pars.n_cells]) > (1-model_pars.fraction_cells)
    #bias_cell_su_selected = tf.transpose(tf.boolean_mask(tf.transpose(bias_cell_su, [3, 0, 1, 2]), select_cells), [1, 2, 3, 0])
    print(select_cells)

    stim4D = tf.expand_dims(tf.reshape(stim_pos, (-1, 40, 80)), 3)
    stim_masked = tf.nn.conv2d(stim4D, mask_tf, strides=[1, stride, stride, 1],
                               padding="VALID")
    print(stim_masked)

    stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)
    # input from convolutional SU and delta SU
    su_act_raw = stim_del  # time x dimx x dimy
    su_act_select_cells = su_act_raw + bias_cell_su_selected  # time x dimx x dimy
    # su_act = su_act_raw + bias_cell_su # time x dimx x dimy x n_cells

    # calculate actual firing rate
    # lam = tf.reduce_sum(tf.reduce_sum(tf.exp(su_act_select_cells), 2), 1)

    # softmax
    su_act_select_cells_softmax = tf.reshape(tf.nn.softmax(tf.reshape(su_act_select_cells, [-1, dimx*dimy])), [-1, dimx, dimy])

    su_update_nr = tf.reduce_sum(tf.expand_dims(tf.expand_dims(resp_pos,2),3) * tf.expand_dims(su_act_select_cells_softmax,3) * stim_masked, 0)/T
    su_update_dr = tf.reduce_sum(tf.expand_dims(tf.expand_dims(resp_pos,2),3) * tf.expand_dims(su_act_select_cells_softmax,3), 0)/T

    new_wdel = (w_del + step_sz*su_update_nr)/ (1 + step_sz*stim_var*su_update_dr)

    update_wdel = tf.assign(w_del, new_wdel)
    with tf.control_dependencies([update_wdel]):
      new_bias_cell = tf.squeeze(tf.log(su_update_dr) - 0.5*tf.expand_dims(tf.reduce_sum(tf.mul(w_del, w_del),2)*stim_var,2))
      update_bias = tf.scatter_update(bias_cell_su, select_cells, tf.expand_dims(new_bias_cell,0))

      update_cell_cnt = tf.scatter_add(cell_cnt, select_cells, tf.constant(np.array([1]).astype(np.float32)))
      with tf.control_dependencies([update_bias, update_wdel]):
        cell_counter_increment = tf.assign(cell_counter, cell_counter+1)
    model_update = tf.group(update_wdel, update_bias, update_cell_cnt, cell_counter_increment)

    # to update biases only. run for all cells without changing the weights.
    new_bias_cell_only = tf.squeeze(tf.log(su_update_dr) - 0.5*tf.expand_dims(tf.reduce_sum(tf.mul(w_del, w_del),2)*stim_var,2))
    update_bias2 = tf.scatter_update(bias_cell_su, select_cells, tf.expand_dims(new_bias_cell_only,0))
    with tf.control_dependencies([update_bias2]):
      cell_counter_increment_bo = tf.assign(cell_counter, cell_counter+1)
    update_bias_only = tf.group(update_bias2, cell_counter_increment_bo)

    return model_update, select_cells, update_bias_only

  def initialize_model(self, save_location, folder_name, sess, feed_dict=None):
    """Setup model variables and saving information."""

    # TODO(bhaishahster): factor out 'session' from inside the library.

    # Make folder.
    self.initialize_folder(save_location, folder_name)

    # Initialize variables.
    self.initialize_variables(sess, feed_dict)


  def initialize_variables(self, sess, feed_dict=None):
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
      self.initialize_stim_var(sess, feed_dict)
      #self.initialize_b(sess)
      tf.logging.info('Variables initialized')

    writer = tf.summary.FileWriter(self.save_location + 'train', sess.graph)

    tf.logging.info('Loaded iteration: %d' % start_iter)

    self.saver_var = saver_var
    self.iter = start_iter
    self.writer = writer

  def initialize_stim_var(self, sess, feed_dict=None):
    # compute stimulus variance

    tf.logging.info('starting update stimulus variance')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tf.logging.info('threads started')

    stim_pix = tf.gather(tf.transpose(self.probes.stim),23)
    stim_var_pix = tf.reduce_mean(tf.mul(stim_pix, stim_pix))
    update_stim_var = tf.assign(self.variables.stim_var, stim_var_pix)

    sess.run(update_stim_var, feed_dict = feed_dict)
    tf.logging.info('Stim_var is %.3f' % sess.run(self.variables.stim_var))
    tf.logging.info('stimulus variance intiialized')


  def initialize_b(self,sess, n_batches_init=100):
    # initialize b based on <yexp(kx)>


    tf.logging.info('initializing b_cell_su')
    # setup data threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tf.logging.info('threads started')

    resp_expanded = tf.expand_dims(tf.expand_dims(self.probes.resp_select_cells, 1), 2)
    b_avg = tf.expand_dims(tf.reduce_mean(tf.mul(self.probes.su_act_select_cells,
                                                        resp_expanded), 0), 0)

    b_initialize = np.zeros((1, self.params.dimx, self.params.dimy, self.params.n_cells))
    for ibatch in range(n_batches_init):
      print('init b: %d' % ibatch)
      b_out, select_cells_np = sess.run([b_avg, self.probes.select_cells])
      b_initialize[:,:,:, select_cells_np] += b_out
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

  def write_summaries(self, sess):
    """Save variables and add summary."""

    #from IPython.terminal.embed import InteractiveShellEmbed
    #ipshell = InteractiveShellEmbed()
    #ipshell()

    # Save variables.
    latest_filename = self.short_filename + '_latest_fn'
    self.saver_var.save(sess, self.save_filename, global_step=self.iter,
                        latest_filename=latest_filename)
    tf.logging.info('Model saved')

    # Add summary.
    #summary = sess.run(self.summary_op)
    #self.writer.add_summary(summary, self.iter)
    #tf.logging.info('Summaries written, iteration: %d' % self.iter)

    # print
    #ls_train = sess.run(self.probes.loss)
    #tf.logging.info('Iter %d, train loss %.3f' % (self.iter, ls_train))



