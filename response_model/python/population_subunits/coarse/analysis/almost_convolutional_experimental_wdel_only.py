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
"""Model firing of a population using almost convolutional s.u. (experimental)

The declared class has attributes to train the model and
view the different components of the circuitry.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path

import numpy as np
import tensorflow as tf

from absl import gfile

class AlmostConvolutionalExperimentalWdelOnly(object):
  """Model firing rate for a population by almost convolutional subunits."""

  def __init__(self, loss_string, stim, resp, short_filename, window=2,
               stride=1, lam_w=0, step_sz=1, n_cells=107, taskid=0):

    """Build the graph to predict population response using subunits.

    Firing rate for cell c: lam_c = a_sfm_c'.relu(w.x + bias_su) + bias_cell,
    x: stimulus, lam_c: firing rate of cell
    bias_c and bias_su : cell and subunit bias
    a_sfm_c = softmax(a) : so a cell cannot be connected to
    all subunits equally well.

    where w_i are over a small window which are
    convolutionally related with each other.
    w_i = w_mother + w_del_i,
    where w_mother is common accross all 'windows' and
    w_del is different for different windows.

    Args:
      loss_string : type of loss to use
      stim: stimulus
      resp : response
      short_filename : filename to store results
      window: (2*window +1) is the convolutional window size
      stride: stride for convolutions
      lam_w : regularizing w_del
      step_sz : step size for SGD
      n_cells : total number of cells in response tensor.
    """

    # Add model specific names to filename.
    short_filename = ('model=almost_convolutional_expt_wdel_only_window=' +
                      str(window) + '_stride=' + str(stride) +
                      '_lam_w=' + str(lam_w) + short_filename)

    # Convolution parameters.
    model_params = collections.namedtuple('model_params',
                                          ['mask_tf', 'dimx', 'dimy',
                                           'n_pix', 'window', 'stride',
                                           'n_cells'])
    mask_tf, dimx, dimy, n_pix = get_windows(window, stride)
    model_pars = model_params(mask_tf, dimx, dimy, n_pix,
                              window, stride, n_cells)

    # Variables.
    model_vars = self.build_variables(model_pars, taskid)

    # Get firing rate.
    lam, su_act = self.build_firing_rate(model_vars, model_pars, stim)

    # Get loss according to specification.
    loss_unregularized = get_loss(loss_string, lam, resp)
    # Regularization skeeps 'delta' weights small.
    regularization = lam_w * tf.reduce_sum(tf.nn.l2_loss(model_vars.w_del))
    loss = loss_unregularized + regularization  # add regularization
    gradient_update = tf.train.AdagradOptimizer(step_sz).minimize(loss)

    # Projection to satisfy hard variable constraints.
    # Project only after gradient update.
    with tf.control_dependencies([gradient_update]):
      proj_ops= []
      if taskid % 2 == 0:
        bias_cell_project_positive = tf.assign(model_vars.bias_cell,
                                               tf.nn.relu(
                                                   model_vars.bias_cell))
        proj_ops += [bias_cell_project_positive]

      if np.floor(taskid/2) % 2 == 0 :
        scale_cell_project_positive = tf.assign(model_vars.scale_cell,
                                         tf.nn.relu(model_vars.scale_cell))
        proj_ops += [scale_cell_project_positive]

    # Make a combined model update op.
    #model_update = tf.group(gradient_update, b_project_positive, scale_cell_project_pos)
    model_update = tf.group(gradient_update, *proj_ops)

    # Make model probes.
    model_probes = collections.namedtuple('model_probes',
                                          ['su_act', 'lam', 'loss',
                                           'loss_unregularized'])
    model_prb = model_probes(su_act, lam, loss, loss_unregularized)

    self.stim = stim
    self.resp = resp
    self.params = model_pars
    self.update = model_update
    self.probes = model_prb
    self.variables = model_vars
    self.short_filename = short_filename
    self.build_summaries()

  def build_variables(self, model_pars, taskid):
    """Declare variables of the model."""

    # Get convolutional windows.
    dimx = model_pars.dimx
    dimy = model_pars.dimy
    n_pix = model_pars.n_pix
    window = model_pars.window
    n_cells = model_pars.n_cells

    # Build model variables.
    w_mother = tf.constant(np.array(np.zeros((2 * window + 1,
                                             2 * window + 1, 1, 1)),
                                    dtype='float32'), name='w_mother')
    w_del = tf.Variable(np.array(0.5 + 0.25*np.random.randn(dimx, dimy, n_pix),
                                 dtype='float32'), name='w_del')
    a = tf.Variable(np.array(np.zeros((dimx*dimy, n_cells)),
                             dtype='float32'), name='a')

    # declare bias_cell
    if taskid % 2 == 0:
      tf.logging.info('bias_cell is variable')
      bias_cell = tf.Variable(np.array(0.000001*np.ones(n_cells),
                                     dtype='float32'), name='bias_cell')
    else:
      tf.logging.info('bias_cell is constant')
      bias_cell = tf.constant(np.array(0.000001*np.ones(n_cells),
                                     dtype='float32'), name='bias_cell')

    # declare scale_cell
    if np.floor(taskid/2) % 2 == 0 :
      tf.logging.info('scale_cell is variable')
      scale_cell = tf.Variable(np.array(np.ones(n_cells),
                                     dtype='float32'), name='scale_cell')
    else:
      tf.logging.info('scale_cell is constant')
      scale_cell = tf.constant(np.array(np.ones(n_cells),
                                     dtype='float32'), name='scale_cell')

    # declare bias_su
    if np.floor(taskid/4) % 2 ==0:
      tf.logging.info('bias_su is variable')
      bias_su = tf.Variable(np.array(0.000001*np.random.randn(1, dimx, dimy),
                                   dtype='float32'), name='bias_su')
    else:
      tf.logging.info('bias_su is constant')
      bias_su = tf.constant(np.array(0.000001*np.random.randn(1, dimx, dimy),
                                   dtype='float32'), name='bias_su')

    # Collect model parameters.
    model_variables = collections.namedtuple('model_variables',
                                             ['w_mother', 'w_del', 'a',
                                              'bias_cell', 'bias_su',
                                              'scale_cell'])
    model_vars = model_variables(w_mother, w_del, a,
                                 bias_cell, bias_su, scale_cell)

    return model_vars

  def build_firing_rate(self, model_vars, model_pars, stim):
    """Compute the firing rate and subunit activations."""

    # Get model parameters.
    mask_tf = model_pars.mask_tf
    dimx = model_pars.dimx
    dimy = model_pars.dimy
    stride = model_pars.stride

    # Get model variables.
    a = model_vars.a
    w_mother = model_vars.w_mother
    w_del = model_vars.w_del
    bias_su = model_vars.bias_su
    bias_cell = model_vars.bias_cell
    scale_cell = model_vars.scale_cell
    k_smoothing = 0.00000001

    a_sfm = tf.transpose(tf.nn.softmax(tf.transpose(a)))
    stim_4d = tf.expand_dims(tf.reshape(stim, (-1, 40, 80)), 3)
    stim_convolved = tf.reduce_sum(tf.nn.conv2d(stim_4d, w_mother,
                                                strides=[1, stride, stride, 1],
                                                padding='VALID'), 3)
    stim_masked = tf.nn.conv2d(stim_4d, mask_tf, strides=[1, stride, stride, 1],
                               padding='VALID')
    stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)
    # Input from convolutional SU and delta SU.
    su_act = tf.nn.relu(stim_del + stim_convolved + bias_su)
    lam = (tf.matmul(tf.reshape(su_act, [-1, dimx*dimy]), a_sfm) * scale_cell +
           bias_cell + k_smoothing)

    return lam, su_act

  def build_summaries(self):
    """Add some summaries."""

    # Add mother subunit weights.
    w_mother = self.variables.w_mother
    mother_min = tf.reduce_min(w_mother)
    mother_max = tf.reduce_max(w_mother - mother_min)
    mother_rescaled = (w_mother - mother_min) / mother_max
    mother_rescaled = tf.transpose(mother_rescaled, [3, 0, 1, 2])
    tf.summary.image('mother', mother_rescaled)

    # Create summary writers.
    # Create histogram summary for all parameters which are learnt.
    for ivar in self.variables:
      tf.summary.histogram(ivar.name, ivar)

    # Loss summary.
    tf.summary.scalar('loss_total', self.probes.loss)

    # Loss without regularization summary.
    tf.summary.scalar('loss_unregularized', self.probes.loss_unregularized)
    # Merge all the summary writer ops into one op (this way,
    # calling one op stores all summaries)

    merged = tf.summary.merge_all()
    self.summary_op = merged
    tf.logging.info('summary OP set')

  def get_summary_op(self):
    """Return the summary op."""
    return self.summary_op

  def initialize_model(self, save_location, folder_name, sess,  feed_dict=None):
    """Setup model variables and saving information."""

    # TODO(bhaishahster): factor out 'session' from inside the library.

    # Make folder.
    self.initialize_folder(save_location, folder_name)

    # Initialize variables.
    self.initialize_variables(sess)

  def initialize_folder(self, save_location, folder_name):
    """Intialize saving location of the model."""

    parent_folder = os.path.join(save_location, folder_name)
    # Make folder if it does not exist.
    if not gfile.IsDirectory(parent_folder):
      gfile.MkDir(parent_folder)
    self.parent_folder = parent_folder

    save_location = os.path.join(parent_folder, self.short_filename)
    if not gfile.IsDirectory(save_location):
      gfile.MkDir(save_location)
    self.save_location = save_location

    self.save_filename = os.path.join(self.save_location, self.short_filename)

  def initialize_variables(self, sess):
    """Initialize variables or restore from previous fits."""

    sess.run(tf.group(tf.initialize_all_variables(),
                      tf.initialize_local_variables()))
    saver_var = tf.train.Saver(tf.all_variables(),
                               keep_checkpoint_every_n_hours=4)
    load_prev = False
    start_iter = 0
    try:
      # Restore previous fits if they are available
      # - useful when programs are preempted frequently on .
      latest_filename = self.short_filename + '_latest_fn'
      restore_file = tf.train.latest_checkpoint(self.save_location,
                                                latest_filename)
      # Restore previous iteration count and start from there.
      start_iter = int(restore_file.split('/')[-1].split('-')[-1])
      saver_var.restore(sess, restore_file)  # restore variables
      load_prev = True
    except:
      #tf.logging.info('Initializing variables from data')
      #self.initialize_variables_from_data(sess)
      tf.logging.info('No previous dataset')

    if load_prev:
      tf.logging.info('Previous results loaded from: ' + restore_file)
    else:
      tf.logging.info('Variables initialized')

    writer = tf.summary.FileWriter(self.save_location + 'train', sess.graph)

    tf.logging.info('Loaded iteration: %d' % start_iter)

    self.saver_var = saver_var
    self.iter = start_iter
    self.writer = writer

  def initialize_variables_from_data(self, sess, n_batches_init=20):
    """Initialize variables smartly by looking at some training data."""


    tf.logging.info('Initializing variables from data')

    # setup data threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tf.logging.info('data threads started')

    tf.logging.info('Initializing a from data')
    resp_expanded = tf.expand_dims(tf.expand_dims(self.resp, 1), 2)
    su_act_expanded = tf.expand_dims(self.probes.su_act, 3)
    a_avg = tf.expand_dims(tf.reduce_mean(tf.mul(su_act_expanded,
                                                 resp_expanded), 0), 0)

    a_initialize = np.zeros((1, self.params.dimx, self.params.dimy, self.params.n_cells))
    for ibatch in range(n_batches_init):
      print('init batch: %d' % ibatch)
      a_initialize += sess.run(a_avg)
    a_initialize /= n_batches_init

    a_max = np.max(np.reshape(a_initialize, [-1, self.params.n_cells]), axis=0)
    mask = a_initialize > a_max*0.7
    a_initial_masked = mask*np.log(a_max) - 40*(1-mask)
    a_initial_tf = tf.constant(a_initial_masked.astype(np.float32))

    a_init_tf = tf.assign(self.variables.a, tf.reshape(a_initial_tf, [-1, self.params.n_cells]))
    sess.run(a_init_tf)
    tf.logging.info('a initialized from data')

    from IPython.terminal.embed import InteractiveShellEmbed
    ipshell = InteractiveShellEmbed()
    ipshell()

    #coord.request_stop()
    #coord.join(threads)

    tf.logging.info('a initialzed based on average activity')


  def write_summaries(self, sess):
    """Save variables and add summary."""

    # Save variables.
    latest_filename = self.short_filename + '_latest_fn'
    self.saver_var.save(sess, self.save_filename, global_step=self.iter,
                        latest_filename=latest_filename)

    # Add summary.
    summary = sess.run(self.summary_op)
    self.writer.add_summary(summary, self.iter)
    tf.logging.info('Summaries written, iteration: %d' % self.iter)

    # print
    ls_train = sess.run(self.probes.loss)
    tf.logging.info('Iter %d, train loss %.3f' % (self.iter, ls_train))


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
  mask_tf = tf.constant(np.array(w_mask, dtype='float32'))

  # Number of windows in x and y dimensions.
  dimx = np.floor(1 + ((40 - (2 * window + 1))/stride)).astype('int')
  dimy = np.floor(1 + ((80 - (2 * window + 1))/stride)).astype('int')
  return mask_tf, dimx, dimy, n_pix


def get_loss(loss_string, lam, resp):
  """Compute Loss based on specification.

  Args:
    loss_string : str for type of loss
    lam : firing rate for cells over time
    resp : observed spiking response

  Returns :
    loss_unregularized : the loss which needs to be minimized to fit the model

  Raises :
    NameError : if the loss_string is not implemented
  """

  if loss_string == 'conditional_poisson':
    tf.logging.info('conditional poisson loss')
    loss_unregularized = -tf.reduce_sum(resp*tf.log(lam) -
                                        tf.reduce_sum(resp, 0)*
                                        tf.log(tf.reduce_sum(lam, 0)))

  elif loss_string == 'poisson':
    tf.logging.info('poisson loss')
    loss_unregularized = tf.reduce_mean(lam/120. -
                                        resp*tf.log(lam))  # poisson loss
  else:
    raise NameError('Loss not identified')

  return loss_unregularized
