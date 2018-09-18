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
"""One-line documentation for jitter_model module.

A detailed description of jitter_model.
"""

import sys
import os.path
import collections
import tensorflow as tf
from absl import app
from absl import flags
from absl import gfile
from tensorflow.contrib.slim.model_deploy import DeploymentConfig, deploy
from tensorflow.python.profiler.model_analyzer import PrintModelAnalysis
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

# This is a horrible hack. I promise never to do this again.
DEVICE_COUNTER = 0

# Declare structure which would hold few TensorFlow object related
# to model
variables = collections.namedtuple("variables",
                                     ["w_mother", "w_del", "a"])
variables_lr = collections.namedtuple("variables_lr",
                                     ["w_mother", "w_del", "a", "w_stim_lr", "bias_su", "bias_cell", "time_course"])
dimensions = collections.namedtuple("dimensions", ["dimx", "dimy"])
ops = collections.namedtuple("ops", ["a_sfm"])
model = collections.namedtuple("model",
                                     ["train_step", "loss_inter", "loss",
                                      "merged_summary", "variables",
                                      "dimensions", "ops"])

dimensions_stimlr = collections.namedtuple("dimensions_stimlr", ["dimx_slr", "dimy_slr"])

model2 = collections.namedtuple("model2",
                                     ["train_step", "merged_summary",
                                      "variables_lr", "dimensions","dimensions_stimlr",
                                      "loss_inter"])

stim_collection = collections.namedtuple("stim_collection", ["stim_convolved","stim_time_filtered","stim_smooth_lr","stim_del"])


def approximate_conv_jitter(n_cells, lam_w, window, stride, step_sz,
                            tc_mean, su_channels):
  ## Used for finding flexible subunits - try on jitter data ?

  ## Define model

  # initialize stuff
  b_init = np.array(0.000001*np.ones(n_cells)) # a very small positive bias needed to avoid log(0) in poisson loss
  # RGB time filter
  tm4D = np.zeros((30,1,3,3)) # flags?
  for ichannel in range(3):
    tm4D[:,0,ichannel,ichannel] = tc_mean[:,ichannel]
  tc = tf.Variable((tm4D).astype('float32'),name = 'tc')

  d1=640
  d2=320
  colors=3

  # stimulus and response placeholders
  stim = tf.placeholder(tf.float32,shape=[None,d1,d2,colors],name='stim')
  resp = tf.placeholder(tf.float32,shape=[None,n_cells],name='resp')
  data_len = tf.to_float(tf.shape(stim)[0])

  # time convolution
  # time course should be time,d1,color,color
  # original stimulus is (time, d1,d2,color).
  # Permute it to (d2,time,d1,color) so that 1D convolution could be mimicked using conv_2d.
  stim_time_filtered = tf.transpose(tf.nn.conv2d(tf.transpose(stim,(2,0,1,3)),
                                                 tc, strides=[1,1,1,1],
                                                 padding='VALID'), (1,2,0,3))

  # learn almost convolutional model
  mask_tf, dimx, dimy, n_pix = get_windows(window, stride, n_channels=3)
  #print(dimx, dimy, n_pix)
  w_del = tf.Variable(np.array( 1*np.random.randn(dimx, dimy, n_pix),
                               dtype='float32'), name='w_del')
  # TODO(bhaishahster) instead of 0 initialization of w_del, multiply randn values by 1 or 0.06(old)

  w_mother = tf.Variable(np.array( np.ones((2 * window + 1, 2 * window + 1,
                                            su_channels, 1)),dtype='float32'),
                         name='w_mother')

  # Generate an image summary for the mother cell.
  mother_min = tf.reduce_min(w_mother)
  mother_max = tf.reduce_max(w_mother - mother_min)
  mother_rescaled = (w_mother - mother_min) / mother_max
  mother_rescaled = tf.transpose(mother_rescaled, [3, 0, 1, 2])
  tf.image_summary('mother', mother_rescaled)

  for channel in range(3):
    mother = w_mother[:, :, channel, 0]
    mother_min = tf.reduce_min(mother)
    mother_max = tf.reduce_max(mother - mother_min)
    mother_rescaled = (mother - mother_min) / mother_max
    mother_rescaled = tf.expand_dims(mother_rescaled, -1)
    mother_rescaled = tf.expand_dims(mother_rescaled, 0)
    tf.image_summary('mother_channel_%d' % channel, mother_rescaled)

  a = tf.Variable(np.array(np.random.randn(dimx*dimy, n_cells),
                           dtype='float32'), name='a')
  a_sfm = tf.transpose(tf.nn.softmax(tf.transpose(a)))

  a_sfm_expanded = tf.expand_dims(a_sfm, 0)
  a_sfm_expanded = tf.expand_dims(a_sfm_expanded, -1)
  tf.image_summary('a_sfm', a_sfm_expanded)

  b = tf.Variable(np.array(b_init,dtype='float32'), name='b')
  vars_fit = [w_mother, a, w_del] # which variables to fit
  # TODO(bhaishahster) add w_del to vars_fit, otherwise, just learning convolutional kernel
  #vars_fit = vars_fit + [b]

  # stimulus filtered with convolutional windows
  stim4D = stim_time_filtered#tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
  stim_convolved = tf.reduce_sum(tf.nn.conv2d(stim4D, w_mother,
                                              strides=[1, stride, stride, 1],
                                              padding="VALID"),3)
  stim_masked = tf.nn.conv2d(stim4D, mask_tf, strides=[1, stride, stride, 1],
                             padding="VALID" )
  stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)

  # activation of different subunits
  su_act = tf.nn.relu(stim_del + stim_convolved)

  # get firing rate
  lam = tf.matmul(tf.reshape(su_act, [-1, dimx*dimy]), a_sfm) + b

  # regularization
  regularization = lam_w * tf.reduce_sum(tf.nn.l2_loss(w_del))

  # projection to satisfy hard variable constraints
  #b_pos = tf.assign(b, (b + tf.abs(b))/2)

  loss_inter = (tf.reduce_sum(lam)/120. -
                tf.reduce_sum(resp*tf.log(lam))) / data_len
  loss = loss_inter + regularization # add regularization to get final loss function

  # training consists of calling training()
  # which performs a train step and project parameters to model specific constraints using proj()
  opt = tf.train.AdagradOptimizer(step_sz)
  if FLAGS.sync_replicas:
    if FLAGS.replicas_to_aggregate is None:
      replicas_to_aggregate = num_workers
    else:
      replicas_to_aggregate = FLAGS.replicas_to_aggregate

    opt = tf.train.SyncReplicasOptimizer(
          opt,
          replicas_to_aggregate=replicas_to_aggregate,
          total_num_replicas=num_workers,
          replica_id=FLAGS.task_index,
          name="mnist_sync_replicas")

  train_step = opt.minimize(loss, var_list=vars_fit,
                            global_step=tf.contrib.framework.get_global_step())

  ## Make summary for tensorboard
  # create summary writers
  # create histogram summary for all parameters which are learnt
  for ivar in vars_fit:
    tf.histogram_summary(ivar.name, ivar)
  # loss summary
  l_summary = tf.scalar_summary('loss',loss)
  # loss without regularization summary
  l_inter_summary = tf.scalar_summary('loss_inter',loss_inter)
  # Merge all the summary writer ops into one op (this way, calling one op stores all summaries)
  merged_summary = tf.merge_all_summaries()

  var= variables (w_mother, w_del, a)
  dims = dimensions(dimx, dimy)
  op = ops(a_sfm)
  return model(train_step, loss_inter, loss, merged_summary, var, dims, op), stim, resp



def get_windows(window,stride,n_channels=3, d1=640, d2=320):
    # use FLAGS to get convolutional 'windows' for convolutional models.
    n_pix = ((2* window + 1) ** 2)*n_channels  # number of pixels in the window
    w_mask = np.zeros((2 * window + 1, 2 * window + 1, n_channels, n_pix))
    icnt = 0

    # make mask_tf: weight (dimx X dimy X npix) for convolutional layer,
    # where each layer is 1 for a particular pixel in window and 0 for others.
    # this is used for flattening the pixels in a window,
    # so that different weights could be applied to each window
    for ichannel in range(n_channels):
        for ix in range(2 * window + 1):
            for iy in range(2 * window + 1):
                w_mask[ix, iy, ichannel, icnt] =1
                icnt = icnt + 1
            
    mask_tf = tf.constant(np.array(w_mask, dtype='float32'))
     # number of windows in x and y dimensions
    dimx = 1+ np.floor(((d1 - (2 * window + 1))/stride)).astype('int')
    dimy = 1+ np.floor(((d2 - (2 * window + 1))/stride)).astype('int')
    return mask_tf, dimx, dimy, n_pix




def build_model(n_cells, lam_w, window, stride, step_sz, tc_mean, su_channels,
                stim_cpu, resp_cpu, vars_l, num_towers):

  global DEVICE_COUNTER
  
  # initialize stuff
  b_init = np.array(0.000001*np.ones(n_cells)) # a very small positive bias needed to avoid log(0) in poisson loss
  # RGB time filter
  tm4D = np.zeros((30,1,3,3)) # flags?
  for ichannel in range(3):
    tm4D[:,0,ichannel,ichannel] = tc_mean[:,ichannel]
  #tc = tf.Variable((tm4D).astype('float32'),name = 'tc')
  tc = tf.constant((tm4D).astype('float32'),name = 'tc')

  d1=640
  d2=320
  colors=3

  # stimulus and response placeholders


  print('num towers: '+ str(num_towers))
  chunk_size = np.array(FLAGS.batchsz/num_towers).astype('int32')
  #chunk_size=50
  print('chunk size: ' + str(chunk_size))
  #chunk_size = tf.to_int32(tf.shape(resp_cpu)[0])/1#tf.constant(num_towers)
  #chunk_size = tf.to_int32(tf.constant(100))
  current_device = DEVICE_COUNTER

  #stim=tf.slice(stim_cpu, tf.constant([chunk_size * current_device,0,0,0]),tf.constant([chunk_size+29,-1,-1,-1]))
  #resp=tf.slice(resp_cpu, tf.constant([chunk_size * current_device,0]), tf.constant([chunk_size, -1]))
  stim = stim_cpu[chunk_size * current_device:
                  chunk_size * (current_device+1)+29, :, :, :]
  resp = resp_cpu[chunk_size * current_device:
                  chunk_size * (current_device+1), :]

  DEVICE_COUNTER += 1
  print("device counter = %d" %DEVICE_COUNTER)


  data_len = tf.to_float(tf.shape(resp)[0])

  # time convolution
  # time course should be time,d1,color,color
  # original stimulus is (time, d1,d2,color).
  # Permute it to (d2,time,d1,color) so that 1D convolution could be mimicked using conv_2d.
  stim_time_filtered = tf.transpose(tf.nn.conv2d(tf.transpose(stim,(2,0,1,3)),
                                                 tc, strides=[1,1,1,1],
                                                 padding='VALID'), (1,2,0,3))

  # learn almost convolutional model
  mask_tf, dimx, dimy, n_pix = get_windows(window, stride, n_channels=3)
  #print(dimx, dimy, n_pix)

  ## retrieve variables
  # w_mother = tf.Variable(np.array( np.ones((2 * window + 1,
  # 2 * window + 1, su_channels, 1)),dtype='float32'), name='w_mother')
  w_mother = vars_l.w_mother

  #w_del = tf.Variable(np.array( 1*np.random.randn(dimx, dimy, n_pix),dtype='float32'), name='w_del')
  w_del = vars_l.w_del
  # TODO(bhaishahster) instead of 0 initialization of w_del, multiply randn values by 1 or 0.06(old)

  #a = tf.Variable(np.array(np.random.randn(dimx*dimy, n_cells),dtype='float32'), name='a')
  a = vars_l.a

  b = tf.constant(np.array(b_init,dtype='float32'), name='b')


  # Generate an image summary for the mother cell.
  mother_min = tf.reduce_min(w_mother)
  mother_max = tf.reduce_max(w_mother - mother_min)
  mother_rescaled = (w_mother - mother_min) / mother_max
  mother_rescaled = tf.transpose(mother_rescaled, [3, 0, 1, 2])
  tf.image_summary('mother', mother_rescaled)

  for channel in range(3):
    mother = w_mother[:, :, channel, 0]
    mother_min = tf.reduce_min(mother)
    mother_max = tf.reduce_max(mother - mother_min)
    mother_rescaled = (mother - mother_min) / mother_max
    mother_rescaled = tf.expand_dims(mother_rescaled, -1)
    mother_rescaled = tf.expand_dims(mother_rescaled, 0)
    tf.image_summary('mother_channel_%d' % channel, mother_rescaled)


  a_sfm = tf.transpose(tf.nn.softmax(tf.transpose(a)))
  a_sfm_expanded = tf.expand_dims(a_sfm, 0)
  a_sfm_expanded = tf.expand_dims(a_sfm_expanded, -1)
  tf.image_summary('a_sfm', a_sfm_expanded)


  # stimulus filtered with convolutional windows
  stim4D = stim_time_filtered#tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
  stim_convolved = tf.reduce_sum(tf.nn.conv2d(stim4D, w_mother,
                                              strides=[1, stride, stride, 1],
                                              padding="VALID"),3)
  stim_masked = tf.nn.conv2d(stim4D, mask_tf, strides=[1, stride, stride, 1],
                             padding="VALID" )
  stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)

  # activation of different subunits
  su_act = tf.nn.relu(stim_del + stim_convolved)

  # get firing rate
  lam = tf.matmul(tf.reshape(su_act, [-1, dimx*dimy]), a_sfm) + b

  # regularization
  regularization = lam_w * tf.reduce_sum(tf.nn.l2_loss(w_del))

  # projection to satisfy hard variable constraints
  #b_pos = tf.assign(b, (b + tf.abs(b))/2)

  loss_inter = (tf.reduce_sum(lam)/120. -
                tf.reduce_sum(resp*tf.log(lam))) / data_len
  loss = tf.add_n([loss_inter, regularization], name="final_loss") # add regularization to get final loss function
  
  tf.get_default_graph().add_to_collection('losses', loss)

  ## Make summary for tensorboard
  # create summary writers
  # create histogram summary for all parameters which are learnt

  # loss summary
  l_summary = tf.scalar_summary('loss',loss)
  # loss without regularization summary
  l_inter_summary = tf.scalar_summary('loss_inter',loss_inter)


  return lam

def approximate_conv_jitter_multigpu(n_cells, lam_w, window,
                                     stride, step_sz, tc_mean,
                                     su_channels, config_params):
  ## find a different subunit in each convolutional window - use GPUs to get bigger batch sizes.

  # Build a configuration specifying multi-GPU and multi-replicas.
  config = DeploymentConfig.parse(config_params)

  print(config)
  ## Define model
  with tf.device(config.variables_device()):
    global_step = tf.contrib.framework.create_global_step()

  # Build the optimizer based on the device specification.
  with tf.device(config.optimizer_device()):
    opt = tf.train.AdagradOptimizer(step_sz)

  with tf.device(config.inputs_device()):
    d1=640
    d2=320
    colors=3
    stim_cpu = tf.placeholder(tf.float32,shape=[None,d1,d2,colors],name='stim_cpu')
    resp_cpu = tf.placeholder(tf.float32,shape=[None,n_cells],name='resp_cpu')

  with tf.device(config.variables_device()):
    # get window locations
    mask_tf, dimx, dimy, n_pix = get_windows(window, stride, n_channels=3)

    w_mother = tf.Variable(np.array( np.ones((2 * window + 1, 2 * window + 1,
                                              su_channels, 1)),dtype='float32'),
                           name='w_mother')
    w_del = tf.Variable(np.array( 1*np.random.randn(dimx, dimy, n_pix),
                                 dtype='float32'), name='w_del')
    a = tf.Variable(np.array(np.random.randn(dimx*dimy, n_cells),
                             dtype='float32'), name='a')
    vars_lst = variables(w_mother, w_del, a)
    for ivar in [w_mother, w_del, a]:
      tf.histogram_summary(ivar.name, ivar)


  # Build the model based on the desired configuration.
  tower_fn = build_model
  tower_args = (n_cells, lam_w, window, stride, step_sz,
                tc_mean, su_channels, stim_cpu, resp_cpu, vars_lst,
                config.num_towers)
  model_combined = deploy(config, tower_fn, optimizer=opt, args=tower_args)

  train_step = model_combined.train_op #opt.minimize(loss, var_list=vars_fit, global_step=tf.contrib.framework.get_global_step())

  # Merge all the summary writer ops into one op (this way, calling one op stores all summaries)
  #merged_summary = tf.merge_all_summaries()
  summary_op = model_combined.summary_op

  var= variables(w_mother, w_del, a)
  dims = dimensions(dimx, dimy)

  return model2(train_step, summary_op, var, dims, model_combined.total_loss), stim_cpu, resp_cpu, global_step



## An approximate convolutional model (wi = wmother+deltawi), where (jitter) stimulus is smoothened first using a 2D convolution
stim_tuple =[]
def build_model_stimlr(n_cells, lam_w,tc_mean, window, stride,
                       stim_downsample_window, stim_downsample_stride,
                       step_sz, su_channels, stim_cpu, resp_cpu,
                       vars_l, num_towers):

  global DEVICE_COUNTER
  



  d1=640
  d2=320
  colors=3

  # stimulus and response placeholders


  print('num towers: '+ str(num_towers))
  chunk_size = np.array(FLAGS.batchsz/num_towers).astype('int32')
  #chunk_size=50
  print('chunk size: ' + str(chunk_size))
  #chunk_size = tf.to_int32(tf.shape(resp_cpu)[0])/1#tf.constant(num_towers)
  #chunk_size = tf.to_int32(tf.constant(100))
  current_device = DEVICE_COUNTER

  #stim=tf.slice(stim_cpu, tf.constant([chunk_size * current_device,0,0,0]),tf.constant([chunk_size+29,-1,-1,-1]))
  #resp=tf.slice(resp_cpu, tf.constant([chunk_size * current_device,0]), tf.constant([chunk_size, -1]))
  stim = stim_cpu[chunk_size * current_device:
                  chunk_size * (current_device+1)+29, :, :, :]
  resp = resp_cpu[chunk_size * current_device:
                  chunk_size * (current_device+1), :]

  DEVICE_COUNTER += 1
  print("device counter = %d" %DEVICE_COUNTER)


  data_len = tf.to_float(tf.shape(resp)[0])

  # time convolution
  # time course should be time,d1,color,color
    # RGB time filter


  tm4D = np.zeros((30,1,3,3)) # flags?
  for ichannel in range(3):
    tm4D[:,0,ichannel,ichannel] = tc_mean[:,ichannel]
  tc = tf.constant((tm4D).astype('float32'),name = 'tc')

  '''
  tmc = vars_l.time_course
  tm4D = tf.zeros((30,1,3,3)) # flags?
  for ichannel in range(3):
    tm4D[:,0,ichannel,ichannel] = tmc[:,ichannel]
  tc = tm4D
  '''
  # original stimulus is (time, d1,d2,color).
  # Permute it to (d2,time,d1,color) so that 1D convolution could be mimicked using conv_2d.
  stim_time_filtered = tf.transpose(tf.nn.conv2d(tf.transpose(stim,(2,0,1,3)),
                                                 tc, strides=[1,1,1,1],
                                                 padding='VALID'),
                                    (1,2,0,3))

  # learn almost convolutional model
  _, dimx_lr, dimy_lr, _ = get_windows(stim_downsample_window,
                                       stim_downsample_stride, n_channels=1)
  w_stim_lr = vars_l.w_stim_lr
  print('dimx_lr %d, dimy_lr %d' % (dimx_lr, dimy_lr))

  # get window locations
  mask_tf, dimx, dimy, n_pix = get_windows(window, stride,
                                           n_channels=1, d1=dimx_lr, d2=dimy_lr)
  w_mother = vars_l.w_mother
  w_del = vars_l.w_del
  a = vars_l.a
  bias_su = vars_l.bias_su
  bias_cell = vars_l.bias_cell

  # Generate an image summary for the stimulus smoothening filter
  stim_lr_min = tf.reduce_min(w_stim_lr)
  stim_lr_max = tf.reduce_max(w_stim_lr - w_stim_lr)
  stim_lr_rescaled = (w_stim_lr - stim_lr_min) / stim_lr_max
  stim_lr_rescaled = tf.transpose(stim_lr_rescaled, [3, 0, 1, 2])
  tf.image_summary('stim_lr', stim_lr_rescaled)

  for channel in range(3):
    stim_lr = w_stim_lr[:, :, channel, 0]
    stim_lr_min = tf.reduce_min(stim_lr)
    stim_lr_max = tf.reduce_max(stim_lr - stim_lr_min)
    stim_lr_rescaled = (stim_lr - stim_lr_min) / stim_lr_max
    stim_lr_rescaled = tf.expand_dims(stim_lr_rescaled, -1)
    stim_lr_rescaled = tf.expand_dims(stim_lr_rescaled, 0)
    tf.image_summary('stim_lr_channel_%d' % channel, stim_lr_rescaled)


  # Generate an image summary for the mother cell.
  mother_min = tf.reduce_min(w_mother)
  mother_max = tf.reduce_max(w_mother - mother_min)
  mother_rescaled = (w_mother - mother_min) / mother_max
  mother_rescaled = tf.transpose(mother_rescaled, [3, 0, 1, 2])
  tf.image_summary('mother', mother_rescaled)

  for channel in range(1):
    mother = w_mother[:, :, channel, 0]
    mother_min = tf.reduce_min(mother)
    mother_max = tf.reduce_max(mother - mother_min)
    mother_rescaled = (mother - mother_min) / mother_max
    mother_rescaled = tf.expand_dims(mother_rescaled, -1)
    mother_rescaled = tf.expand_dims(mother_rescaled, 0)
    tf.image_summary('mother_channel_%d' % channel, mother_rescaled)


  a_sfm = tf.transpose(tf.nn.softmax(tf.transpose(a)))
  a_sfm_expanded = tf.expand_dims(a_sfm, 0)
  a_sfm_expanded = tf.expand_dims(a_sfm_expanded, -1)
  tf.image_summary('a_sfm', a_sfm_expanded)


  # stimulus filtered with convolutional windows
  stim4D = stim_time_filtered
  # smoothen the stimulus by a convolution
  stim_smooth_lr = tf.nn.conv2d(stim4D, w_stim_lr,
                                strides=[1, stim_downsample_stride,
                                         stim_downsample_stride, 1],
                                padding="VALID")

  # convolve with mother cell
  stim_convolved = tf.squeeze(tf.nn.conv2d(stim_smooth_lr, w_mother,
                                           strides=[1, stride, stride, 1],
                                           padding="VALID"))

  # extract stimulus in windows and multiply by delta weights
  stim_masked = tf.nn.conv2d(stim_smooth_lr, mask_tf,
                             strides=[1, stride, stride, 1],
                             padding="VALID" )
  stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)
  total_su_input = stim_del + stim_convolved

  # activation of different subunits
  su_act = tf.nn.relu(total_su_input + bias_su)
  su_act_flatten = tf.reshape(su_act, [-1, dimx*dimy])

  # get firing rate of cells
  lam = tf.nn.relu(tf.matmul(su_act_flatten, a_sfm) + bias_cell) + 0.000001

  # regularization
  regularization = lam_w * tf.reduce_sum(tf.nn.l2_loss(w_del))

  # projection to satisfy hard variable constraints
  #b_pos = tf.assign(b, (b + tf.abs(b))/2)

  loss_inter = (tf.reduce_sum(lam)/120. -
                tf.reduce_sum(resp*tf.log(lam))) / data_len
  loss = tf.add_n([loss_inter, regularization], name="final_loss") # add regularization to get final loss function

  tf.get_default_graph().add_to_collection('losses', loss)

  ## Make summary for tensorboard
  # create summary writers
  # create histogram summary for all parameters which are learnt

  # loss summary
  l_summary = tf.scalar_summary('loss',loss)
  # loss without regularization summary
  l_inter_summary = tf.scalar_summary('loss_inter',loss_inter)

  # intermediate stimulus tuple
  global stim_tuple
  stim_tuple = stim_collection(stim_convolved, stim_time_filtered, stim_smooth_lr, stim_del)

  return lam


def decode_op_stim_lr(sess, stim4D, sux, suy, vars_l, window, stride,
                       stim_downsample_window, stim_downsample_stride, dimx_lr, dimy_lr, dimx, dimy, n_cells, max_element='su'):
  w_stim_lr = vars_l.w_stim_lr

  # get window locations
  mask_tf, dimx, dimy, n_pix = get_windows(window, stride,
                                           n_channels=1, d1=dimx_lr, d2=dimy_lr)
  w_mother = vars_l.w_mother
  w_del = vars_l.w_del
  a = vars_l.a
  a_sfm = tf.transpose(tf.nn.softmax(tf.transpose(a)))
  bias_su = vars_l.bias_su
  bias_cell = vars_l.bias_cell

  stim_smooth_lr = tf.nn.conv2d(stim4D, w_stim_lr,
                                strides=[1, stim_downsample_stride,
                                         stim_downsample_stride, 1],
                                padding="VALID")

  # convolve with mother cell
  stim_convolved = tf.squeeze(tf.nn.conv2d(stim_smooth_lr, w_mother,
                                           strides=[1, stride, stride, 1],
                                           padding="VALID"))

  # extract stimulus in windows and multiply by delta weights
  stim_masked = tf.nn.conv2d(stim_smooth_lr, mask_tf,
                             strides=[1, stride, stride, 1],
                             padding="VALID" )
  stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)
  total_su_input = stim_del + stim_convolved

  # activation of different subunits
  su_act = total_su_input + bias_su
  su_act_flatten = tf.reshape(tf.nn.relu(su_act), [-1, dimx*dimy])

  # get firing rate of cells
  lam = tf.nn.relu(tf.matmul(su_act_flatten, a_sfm) + bias_cell) + 0.000001

  if max_element=='su':
    print('maximize for su')
    def maximize_stim_for_su():
      su_chosen_act = su_act[0, sux, suy]
      maximize_su = tf.train.AdagradOptimizer(0.001).minimize(-su_chosen_act,
                                                                   var_list=
                                                                   [stim4D])
      stim_normalize = tf.assign(stim4D, tf.nn.l2_normalize(stim4D, dim=[0,1,2,3]))
      sess.run(tf.initialize_all_variables())
      print('decoding stim for a subunit at', sux, suy)

      _ = sess.run(stim_normalize)
      for istep in range(500):

        #_,_,su_act_np = sess.run([maximize_su, stim_normalize,su_chosen_act])
        _,su_act_np, = sess.run([maximize_su, su_chosen_act])
        _ = sess.run(stim_normalize)
        s4d = sess.run(stim4D)
        #print('stim4d = %s (%f)' % (s4d, np.linalg.norm(s4d)))
        print(istep, su_act_np)
      return sess.run(stim4D), su_act_np


    return maximize_stim_for_su
  else:
    print('maximize for cell')
    def maximize_stim_for_cell():

      cell_act = lam[0, sux] # use sux as cell ID
      maximize_su = tf.train.AdagradOptimizer(0.001).minimize(-cell_act,
                                                                    var_list=
                                                                   [stim4D])
      stim_normalize = tf.assign(stim4D, tf.nn.l2_normalize(stim4D, dim=[0,1,2,3]))
      sess.run(tf.initialize_all_variables())
      print('decoding stim for a subunit at', sux, suy)

      _ = sess.run(stim_normalize)
      for istep in range(2500):
        _,cell_act_np, = sess.run([maximize_su, cell_act])
        _ = sess.run(stim_normalize)
        s4d = sess.run(stim4D)
        #print('stim4d = %s (%f)' % (s4d, np.linalg.norm(s4d)))
        print(istep, cell_act_np)
      return sess.run(stim4D), cell_act_np

    return maximize_stim_for_cell

def approximate_conv_jitter_multigpu_stim_lr(n_cells, lam_w, window, stride,
                                             step_sz, tc_mean, su_channels,
                                             config_params,
                                             stim_downsample_window,
                                             stim_downsample_stride):
  ## stimulus is first smoothened to lower dimensions,
  # then approximate convolutional architecture is applied.

  # Build a configuration specifying multi-GPU and multi-replicas.
  config = DeploymentConfig.parse(config_params)

  print(config)
  ## Define model
  with tf.device(config.variables_device()):
    global_step = tf.contrib.framework.create_global_step()

  # Build the optimizer based on the device specification.
  with tf.device(config.optimizer_device()):
    opt = tf.train.AdagradOptimizer(step_sz)

  with tf.device(config.inputs_device()):
    d1=640
    d2=320
    colors=3
    stim_cpu = tf.placeholder(tf.float32,shape=[None,d1,d2,colors],name='stim_cpu')
    resp_cpu = tf.placeholder(tf.float32,shape=[None,n_cells],name='resp_cpu')

  with tf.device(config.variables_device()):
    # get window for putting stimulus into lower dimensions
    _, dimx_lr, dimy_lr, _ = get_windows(stim_downsample_window,
                                         stim_downsample_stride, n_channels=1)
    w_stim_lr = tf.Variable(np.array(np.random.randn(2*stim_downsample_window+1,
                                              2*stim_downsample_window+1,3,1),
                                     dtype='float32'), name="w_stim_lr")
    print('dimx_lr %d, dimy_lr %d' % (dimx_lr, dimy_lr))


    # get window locations
    mask_tf, dimx, dimy, n_pix = get_windows(window, stride,
                                             n_channels=1, d1=dimx_lr,
                                             d2=dimy_lr)
    print('dimx %d, dimy %d' %(dimx, dimy))
    w_mother = tf.Variable(np.array( np.random.randn(2 * window + 1,
                                              2 * window + 1, 1, 1),
                                    dtype='float32'), name='w_mother')
    w_del = tf.Variable(np.array( 0.01*np.random.randn(dimx, dimy, n_pix),
                                 dtype='float32'), name='w_del')
    a = tf.Variable(np.array(np.random.randn(dimx*dimy, n_cells),
                             dtype='float32'), name='a')
    bias_su = tf.Variable(np.array(0.1*np.random.rand(dimx, dimy), dtype='float32'), name="bias_su")
    bias_cell = tf.Variable(np.array(0.1*np.random.rand(n_cells), dtype='float32'), name="bias_cells")
    time_course = tf.constant(np.array(tc_mean, dtype='float32'))

    vars_lst = variables_lr(w_mother, w_del, a, w_stim_lr, bias_su, bias_cell, time_course)
    for ivar in [w_mother, w_del, a, w_stim_lr]:
      tf.histogram_summary(ivar.name, ivar)


  # Build the model based on the desired configuration.
  tower_fn = build_model_stimlr
  tower_args = (n_cells, lam_w,tc_mean, window, stride, stim_downsample_window,
                stim_downsample_stride, step_sz,  su_channels,
                stim_cpu, resp_cpu, vars_lst, config.num_towers)
  model_combined = deploy(config, tower_fn, optimizer=opt, args=tower_args)
  train_step = model_combined.train_op #opt.minimize(loss, var_list=vars_fit, global_step=tf.contrib.framework.get_global_step())


  # compute stimulus to maximize output of a particular unit

  # Merge all the summary writer ops into one op
  # (this way, calling one op stores all summaries)
  #merged_summary = tf.merge_all_summaries()
  summary_op = model_combined.summary_op
  dims = dimensions(dimx, dimy)
  dims_slr = dimensions_stimlr(dimx_lr, dimy_lr)

  '''
  # Make functions which decodes stimulus to maximize the activation of diff. subunits
  decode_list2D = []
  stim4D = tf.Variable(np.zeros((1,640,320,3), dtype='float32'), name="decoded_stimulus")

  for sux in [23]:#np.arange(dimx):
    decode_f_list=[]
    for suy in np.arange(dimy):
      print(sux, suy)
      decode_f = decode_op_stim_lr(stim4D, sux, suy, vars_lst, window, stride,
                             stim_downsample_window, stim_downsample_stride,
                                              dimx_lr, dimy_lr, dimx, dimy, n_cells)
      decode_f_list.append(decode_f)

    decode_list2D.append(decode_f_list)
  '''

  global stim_tuple
  return model2(train_step, summary_op, vars_lst, dims,dims_slr, model_combined.total_loss,), stim_cpu, resp_cpu, global_step, stim_tuple


def main(argv):
  pass


if __name__ == '__main__':
  app.run()
