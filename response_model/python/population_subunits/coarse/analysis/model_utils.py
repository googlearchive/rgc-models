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
""" Model utils 2.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path
import collections
import tensorflow as tf
from absl import app
from absl import flags
from absl import gfile
import cPickle as pickle
import cPickle as pickle
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import ndimage
import random




class Model(object):
  def __init__(self):
    pass

  def __init__(self, stim, resp, training_fcn, model_probes, model_params, short_filename):
    self.stim = stim
    self.resp = resp
    self.training_fcn = training_fcn
    self.model_probes = model_probes
    self.model_params = model_params
    self.short_filename = short_filename
    self.set_summaries()


  def set_summaries(self):
    ## Create summary writers.
    # Create histogram summary for all parameters which are learnt.
    for ivar in self.model_params:
      tf.histogram_summary(ivar.name, ivar)
    # Loss summary.
    tf.scalar_summary('loss_total',self.model_probes.loss)
    # Loss without regularization summary.
    tf.scalar_summary('loss_unregularized',self.model_probes.loss_unregularized)
    # Merge all the summary writer ops into one op (this way,
    # calling one op stores all summaries).

    merged = tf.merge_all_summaries()
    self.summary_op = merged
    tf.logging.info('summary OP set')

  def get_summaries(self):
    return self.summary_op

  def initialize_model(self, save_location, folder_name, sess):
    # Make folder.
    self.initialize_folder(save_location, folder_name)

    # Initialize variables.
    self.initialize_variables(sess)


  def initialize_folder(self, save_location, folder_name):

    parent_folder = os.path.join(save_location, folder_name)
    # make folder if it does not exist
    if not gfile.IsDirectory(parent_folder):
      gfile.MkDir(parent_folder)
    self.parent_folder = parent_folder

    save_location = os.path.join(parent_folder, self.short_filename)
    if not gfile.IsDirectory(save_location):
      gfile.MkDir(save_location)
    self.save_location = save_location

    self.save_filename = os.path.join(self.save_location, self.short_filename)


  def initialize_variables(self, sess):
    ## Initialize variables
    sess.run(tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables()))
    saver_var = tf.train.Saver(tf.all_variables(),
                               keep_checkpoint_every_n_hours=0.05)
    load_prev = False
    start_iter=0
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
      print('No previous dataset')

    if load_prev:
      print('Previous results loaded')
    else:
      print('Variables initialized')

    train_writer = tf.train.SummaryWriter(self.save_location + 'train', sess.graph)
    test_writer = tf.train.SummaryWriter(self.save_location + 'test', sess.graph)

    tf.logging.info('Starting iteration: %d' % start_iter)
    
    self.saver_var = saver_var
    self.iter = start_iter
    self.train_writer = train_writer
    self.test_writer = test_writer


  def write_summaries(self, sess, fd_train, fd_test):

    # save variables
    latest_filename = self.short_filename + '_latest_fn'
    self.saver_var.save(sess, self.save_filename, global_step=self.iter,
                     latest_filename = latest_filename)

    # add training summary
    summary = sess.run(self.summary_op, feed_dict=fd_train)
    self.train_writer.add_summary(summary,self.iter)

    # add testing summary
    summary = sess.run(self.summary_op, feed_dict=fd_test)
    self.test_writer.add_summary(summary,self.iter)
    tf.logging.info('Summaries written, iteration: %d' % self.iter)

    # print
    ls_train = sess.run(self.model_probes.loss,feed_dict = fd_train)
    ls_test = sess.run(self.model_probes.loss,feed_dict = fd_test)
    print(self.iter, ls_train, ls_test)
    tf.logging.info( 'Iter %d, train loss %.3f, test loss %.3f' % (self.iter, ls_train, ls_test))






def get_windows(window, stride):
  # get locations and arrangement of the convolutional windows

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
  dimx = np.floor(1 + ((40 - (2 * window + 1))/stride)).astype('int')
  dimy = np.floor(1 + ((80 - (2 * window + 1))/stride)).astype('int')
  return mask_tf, dimx, dimy, n_pix



def almost_convolutional(loss_string, sess, stim, resp, short_filename, window=2,
                         stride=1, lam_w=0, step_sz=1, n_cells=100):
  # firing rate for cell c: lam_c = a_sfm_c'.relu(w.x + bias_su) + bias_cell,
  # a_sfm_c = softmax(a) : so a cell cannot be connected to all subunits equally well.

  # where w_i are over a small window which are convolutionally related with each other.
  # w_i = w_mother + w_del_i,
  # where w_mother is common accross all 'windows' and w_del is different for different windows.

  # we know a>0 and for poisson loss, bias_cell>0
  # for poisson loss: small b added to prevent lam_c going to 0

  short_filename = ('model=' + 'almost_convolutional' + '_window=' +
                   str(window) + '_stride=' + str(stride) +
                  '_lam_w=' + str(lam_w) + short_filename)
  mask_tf, dimx, dimy, n_pix = get_windows(window, stride)


  # variables
  w_del = tf.Variable(np.array( 0.05*np.random.randn(dimx, dimy, n_pix), dtype='float32'), name='w_del')
  w_mother = tf.Variable(np.array(np.ones((2 * window + 1, 2 * window + 1, 1, 1)), dtype='float32'), name='w_mother')
  a = tf.Variable(np.array(np.random.randn(dimx*dimy, n_cells), dtype='float32'), name='a')
  a_sfm = tf.transpose(tf.nn.softmax(tf.transpose(a)))
  bias_cell = tf.Variable(np.array(0.000001*np.ones(n_cells),dtype='float32'), name='bias_cell')
  bias_su = tf.Variable(np.array(np.random.randn(1, dimx, dimy),dtype='float32'), name='bias_su')
  model_params = collections.namedtuple("model_params", ["w_mother", "w_del", "a", "bias_cell", "bias_su"])
  model_pars = model_params(w_mother, w_del, a, bias_cell, bias_su)

  # get firing rate
  stim4D = tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
  stim_convolved = tf.reduce_sum(tf.nn.conv2d(stim4D, w_mother,strides=[1, stride,stride, 1], padding="VALID"),3)
  stim_masked = tf.nn.conv2d(stim4D, mask_tf, strides=[1, stride, stride, 1], padding="VALID" )
  stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)
  su_act = tf.nn.relu(stim_del + stim_convolved + bias_su)
  lam = tf.matmul(tf.reshape(su_act, [-1, dimx*dimy]), a_sfm) + bias_cell + 0.00000001

  # regularization
  regularization = lam_w * tf.reduce_sum(tf.nn.l2_loss(w_del))

  # projection to satisfy hard variable constraints
  b_pos = tf.assign(bias_cell, (bias_cell + tf.abs(bias_cell))/2)
  def proj():
      sess.run(b_pos)

  # poisson loss
  loss_unregularized = get_loss(loss_string, lam, resp)
  loss = loss_unregularized + regularization # add regularization to get final loss function
  train_step = tf.train.AdagradOptimizer(step_sz).minimize(loss)

  model_probes = collections.namedtuple("model_probes", ["su_act", "lam", "loss", "loss_unregularized"])
  model_prb = model_probes(su_act, lam, loss, loss_unregularized)


  def training(inp_dict=None):
    _, loss_np = sess.run([train_step, loss], feed_dict=inp_dict) # one step of gradient descent
    proj() # model specific projection operations
    return loss_np

  # add some summaries
  # show mother subunit weights
  mother_min = tf.reduce_min(w_mother)
  mother_max = tf.reduce_max(w_mother - mother_min)
  mother_rescaled = (w_mother - mother_min) / mother_max
  mother_rescaled = tf.transpose(mother_rescaled, [3, 0, 1, 2])
  tf.image_summary('mother', mother_rescaled)
  # additional summaries will be made in Model.set_summaries()

  model_coll = Model(stim, resp, training, model_prb, model_pars, short_filename)

  return model_coll


def get_loss(loss_string, lam, resp):

  if loss_string == 'conditional_poisson':
    print('conditional poisson loss')
    loss_unregularized = -tf.reduce_sum(resp*tf.log(lam) - tf.reduce_sum(resp,0)*tf.log(tf.reduce_sum(lam ,0)))

  if loss_string == 'poisson':
    print('poisson loss')
    loss_unregularized = tf.reduce_mean(lam/120. - resp*tf.log(lam)) # poisson loss


  return loss_unregularized





def setup_response_model(model_id, *model_build_params):
  # based on model_id, build the appropriate model graph

  # get filename and make folder
  # build model
  if model_id == 'almost_convolutional':
    model_coll = almost_convolutional(*model_build_params)

  # add summary op
  _ = model_coll.get_summaries()


  return model_coll

