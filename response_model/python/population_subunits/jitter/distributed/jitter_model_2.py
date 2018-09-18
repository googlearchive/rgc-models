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
import retina.response_model.python.l1_projection_tf as l1_projection_tf
import retina.response_model.python.population_subunits.jitter.distributed.get_data_mat_fast as get_data_mat

FLAGS = flags.FLAGS

# This is a horrible hack. I promise never to do this again.
DEVICE_COUNTER = 0

# Declare structure which would hold few TensorFlow object related
# to model

# Tensorflow Variables
variables_lr = collections.namedtuple("variables_lr",
                                     ["w_mother", "w_del", "a", "w_stim_lr",
                                      "bias_su", "bias_cell", "time_course"])
# dimensions of model
dimensions = collections.namedtuple("dimensions", ["dimx", "dimy"])
dimensions_stimlr = collections.namedtuple("dimensions_stimlr",
                                           ["dimx_slr", "dimy_slr"])

# arbitrary ops (redundant)
ops = collections.namedtuple("ops", ["a_sfm"])

# relevant information about model - training, summary, variable, loss,
# projection and probe ops
model2 = collections.namedtuple("model2",
                                     ["train_step", "merged_summary",
                                      "variables_lr", "dimensions",
                                      "dimensions_stimlr",
                                      "loss_inter", "proj_ops", "probe_ops"])

# ops at different stages of stimulus processing - useful for debugging
stim_collection = collections.namedtuple("stim_collection",
                                         ["stim_convolved",
                                          "stim_time_filtered",
                                          "stim_smooth_lr",
                                          "stim_del"])


def get_windows(window,stride,n_channels=3, d1=640, d2=320):
    # Get parameters of convolutional 'windows'

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




global probe_nodes # dont want to make it 'global', but have to do it right now
def approximate_conv_jitter_multigpu_complex(n_cells, lam_w, window, stride,
                                             step_sz, tc_mean, su_channels,
                                             config_params,
                                             stim_downsample_window,
                                             stim_downsample_stride):
  ## Sets up the entire graph and summary ops.
  # Stimulus is first smoothened to lower dimensions
  # using convolution with w_stimlr and max pooling.
  # Followed by approximate convolutional architecture and poisson spiking.

  # An "approximate convolutional model" one where weights in
  # each convolutional window is sum of a common component (wmother) and
  # subunit specific modification (wdeltai)(wi = wmother+deltawi)

  ## Build a configuration specifying multi-GPU and multi-replicas.
  config = DeploymentConfig.parse(config_params)
  print(config)

  ## Start building the graph
  with tf.device(config.variables_device()):
    global_step = tf.contrib.framework.create_global_step()

  ## Build the optimizer based on the device specification.
  with tf.device(config.optimizer_device()):
    opt = tf.train.AdagradOptimizer(step_sz)


  ## Make stimulus and response placeholders
  with tf.device(config.inputs_device()):
    d1=640
    d2=320
    colors=3
    stim_cpu = tf.placeholder(tf.float32,shape=[None,d1,d2,colors],name='stim_cpu')
    resp_cpu = tf.placeholder(tf.float32,shape=[None,n_cells],name='resp_cpu')


  ## Set up variables
  with tf.device(config.variables_device()):

    ## Convert stimulus into lower resolution
    _, dimx_lr, dimy_lr, _ = get_windows(stim_downsample_window,
                                         stim_downsample_stride, n_channels=1)
    w_stim_lr = tf.Variable(np.array(0.05 +0*np.random.randn(2*stim_downsample_window+1,
                                              2*stim_downsample_window+1,3,1),
                                     dtype='float32'), name="w_stim_lr")
    print('dimx_lr %d, dimy_lr %d' % (dimx_lr, dimy_lr))

    # max pooling
    _, dimx_maxpool, dimy_maxpool, _ = get_windows(FLAGS.window_maxpool, FLAGS.stride_maxpool, n_channels=1, d1=dimx_lr, d2=dimy_lr)
    print('dimx_maxpool %d, dimy_maxpool %d' % (dimx_maxpool, dimy_maxpool))



    ## Set parameters for "almost convolutional model"
    # get window locations
    mask_tf, dimx, dimy, n_pix = get_windows(window, stride,
                                             n_channels=1, d1=dimx_maxpool,
                                             d2=dimy_maxpool)
    print('dimx %d, dimy %d' %(dimx, dimy))

    # mother subunit
    w_mother = tf.Variable(np.array( 0.05 + 0*np.random.randn(2 * window + 1,
                                              2 * window + 1, 1, 1),
                                    dtype='float32'), name='w_mother')

    # subunit specific modifications to each window
    w_del = tf.Variable(np.array( 0.001 * np.random.randn(dimx, dimy, n_pix),
                                 dtype='float32'), name='w_del')
    #w_del = tf.constant(np.array( 0 * np.random.randn(dimx, dimy, n_pix),
    #                             dtype='float32'), name='w_del')

    # weights from each subunit to each cell
    a = tf.Variable(np.array( 1* np.random.rand(dimx*dimy, n_cells),
                             dtype='float32'), name='a')

    # biases for each subunit and each cell
    bias_su = tf.Variable(np.array(0.00001*np.random.rand(dimx, dimy), dtype='float32'), name="bias_su")
    bias_cell = tf.Variable(np.array(0.00001*np.random.rand(n_cells), dtype='float32'), name="bias_cells")

    # time course derived from STA
    time_course = tf.constant(np.array(tc_mean, dtype='float32'))

    # make summary op for each parameter
    vars_lst = variables_lr(w_mother, w_del, a, w_stim_lr, bias_su, bias_cell, time_course)
    for ivar in [w_mother, w_del, a, w_stim_lr]:
      tf.histogram_summary(ivar.name, ivar)


    ## Compute which subunits will be connected to which cell
    su_cell_mask = get_su_cell_overlap(n_cells, window, stride,
                       stim_downsample_window, stim_downsample_stride)

    su_cell_mask_tf = tf.constant(np.array(su_cell_mask, dtype='float32'))


    # add projection op
    # Mixed norm (L2/L1) projection for W_del for block sparsity
    v_norm = tf.sqrt(tf.reduce_sum(tf.pow(w_del,2),2)) # if v_norm is 0, the it gives NaN
    scale = tf.clip_by_value(1 - lam_w*FLAGS.step_sz*tf.inv(v_norm), 0 , float('inf'))
    w_del_old = w_del
    # proj_wdel = tf.assign(w_del, tf.transpose(tf.mul(tf.transpose(w_del, (2, 0, 1)), scale), (1, 2, 0))) # mixed L2/L1 norm on w_del
    # proj_ops = [proj_wdel]
    # probe_ops = [v_norm, scale, w_del, w_del_old]

    # proximal step for L1 for sparsity in a
    #a_new = tf.nn.relu(a - FLAGS.lam_a) - tf.nn.relu(a - FLAGS.lam_a)
    #proj_a = tf.assign(a, a_new)
    #proj_ops = [proj_a]
    #probe_ops = [a]

    #proj_op_list=[]
    #for icell in np.arange(n_cells):
    ## old code
    ##  proj_op = l1_projection_tf.Project(a, 0, tf.constant(float(50)), tf.constant(0.01))
    ##  proj_op_list.append(proj_op)
    ##proj_ops = [tf.group(*proj_op_list)]

    #a_proj = tf.nn.relu(l1_projection_tf.Project(a, 0, tf.constant(float(FLAGS.rad_a)), tf.constant(0.01)))
    #a_proj_assign = tf.assign(a, a_proj)
    #proj_ops = a_proj_assign
    #probe_ops = []

    # if a is not passed through SFM, then make sure to keep a positive
    proj_ops = []
    if not(FLAGS.if_a_sfm):
      a_new = tf.nn.relu(a)
      proj_a_positive = tf.assign(a, a_new)
      proj_ops += [proj_a_positive]
      print('projections happening')


    # project a to have support determined by su_cell_mask
    if not(FLAGS.if_a_sfm):
      a_proj_support = tf.assign(a, tf.mul(a, su_cell_mask_tf))

    else:
      a_proj_support = tf.assign(a, (tf.mul(a, su_cell_mask_tf)
                                           - 40*(1-su_cell_mask_tf)))
    proj_ops += [a_proj_support]
    probe_ops = [a]
    print("a support is fixed")

    # make sure b_cell is non-negative
    bias_cell_new = tf.nn.relu(bias_cell)
    bias_cell_proj = tf.assign(bias_cell, bias_cell_new)
    proj_ops += [bias_cell_proj]
    print('Number of projection ops are: %d' % len(proj_ops))
    print("projection op made")


  ## Set up identical model on each tower (GPU) (based on user configuration)
  # to convert stimulus into firing rate across cell

  tower_fn = build_model_complex
  tower_args = (n_cells, lam_w,tc_mean, window, stride, stim_downsample_window,
                stim_downsample_stride, step_sz,  su_channels,
                stim_cpu, resp_cpu, vars_lst, config.num_towers)
  model_combined = deploy(config, tower_fn, optimizer=opt, args=tower_args)
  train_step = model_combined.train_op #opt.minimize(loss, var_list=vars_fit, global_step=tf.contrib.framework.get_global_step())
  global probe_nodes
  probe_ops.append(probe_nodes)



  ## compute stimulus to maximize output of a particular unit
  # Merge all the summary writer ops into one op
  # (this way, calling one op stores all summaries)
  #merged_summary = tf.merge_all_summaries()
  summary_op = model_combined.summary_op
  dims = dimensions(dimx, dimy)
  dims_slr = dimensions_stimlr(dimx_lr, dimy_lr)

  return model2(train_step, summary_op, vars_lst, dims,dims_slr, model_combined.total_loss, proj_ops, probe_ops), stim_cpu, resp_cpu, global_step



def build_model_complex(n_cells, lam_w,tc_mean, window, stride,
                       stim_downsample_window, stim_downsample_stride,
                       step_sz, su_channels, stim_cpu, resp_cpu,
                       vars_l, num_towers):

  ## Build the part of graph which goes from stimulus
  # to response (replicated on each tower)


  global DEVICE_COUNTER
  d1=640
  d2=320
  colors=3

  ## Extract part of data based on DEVICE_COUNTER
  print('num towers: '+ str(num_towers))
  chunk_size = np.array(FLAGS.batchsz/num_towers).astype('int32')
  print('chunk size: ' + str(chunk_size))
  current_device = DEVICE_COUNTER
  stim = stim_cpu[chunk_size * current_device:
                  chunk_size * (current_device+1), :, :, :] # BIG CHANGE
  resp = resp_cpu[chunk_size * current_device +29:
                  chunk_size * (current_device+1), :] # BIG CHANGE
  DEVICE_COUNTER += 1
  print("device counter = %d" %DEVICE_COUNTER)
  print("stim times range = %d, %d" %(chunk_size * current_device, chunk_size * (current_device+1)))
  print("resp times range = %d, %d" %(chunk_size * current_device +29, chunk_size * (current_device+1)))
  data_len = tf.to_float(tf.shape(resp)[0])


  ## Extract model parameters sitting on CPU
  w_stim_lr = vars_l.w_stim_lr
  w_mother = vars_l.w_mother
  w_del = vars_l.w_del
  a = vars_l.a

  if FLAGS.if_a_sfm:
    a_sfm = tf.transpose(tf.nn.softmax(tf.transpose(a)))
  else:
    a_sfm = a

  bias_su = vars_l.bias_su
  bias_cell = vars_l.bias_cell


  ## Extract filter for time convolution
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

  ## Filter RGB stimulus using time filter
  # original stimulus is (time, d1,d2,color).
  # Permute it to (d2,time,d1,color) so that 1D convolution could be mimicked using conv_2d.
  stim_time_filtered = tf.transpose(tf.nn.conv2d(tf.transpose(stim,(2,0,1,3)),
                                                 tc, strides=[1,1,1,1],
                                                 padding='VALID'),
                                    (1,2,0,3))

  ## Get stimulus parameters for getting low resolution version of stimulus
  _, dimx_lr, dimy_lr, _ = get_windows(stim_downsample_window,
                                       stim_downsample_stride, n_channels=1)

  # maxpool
  _, dimx_maxpool, dimy_maxpool, _ = get_windows(FLAGS.window_maxpool, FLAGS.stride_maxpool, n_channels=1, d1=dimx_lr, d2=dimy_lr)

  # get window locations
  mask_tf, dimx, dimy, n_pix = get_windows(window, stride,
                                             n_channels=1, d1=dimx_maxpool,
                                             d2=dimy_maxpool)



  ## Pass the stimulus through the "almost convolutional model" to get
  # lam(firing rate) across cells
  stim4D = stim_time_filtered
  lam, su_act_in = stim4D_to_lam_complex(stim4D, a_sfm, bias_su, bias_cell, w_stim_lr,
                                w_mother, w_del, stim_downsample_stride, stride, mask_tf, dimx, dimy)

  ## Compute total loss
  # compute spiking loss terms - simply add across cells
  if not(FLAGS.if_weighted_LL):
    print('Loss: Un-Weighed log-likelihood')
    loss_inter = (tf.reduce_sum(lam)/120. -
                 tf.reduce_sum(resp*tf.log(lam))) / data_len
  else:
    # compute spiking loss - weigh each cell according to the number of spikes it has
    print('Loss: Weighed log-likelihood')
    weights = get_data_mat.get_cell_weights()
    weights = np.squeeze(weights)
    print('Weights for difference cells are: ' + str(weights))
    weights_tf = tf.constant(np.array(weights, dtype='float32'))
    loss_inter = tf.reduce_sum(tf.mul((tf.reduce_sum(lam, 0)/120. -
                   tf.reduce_sum(resp*tf.log(lam), 0)), weights_tf)) / data_len

  # compute regularization terms
  # regularization = lam_w * tf.reduce_sum(tf.nn.l2_loss(w_del)) # should be lam_w
  regularization =  (lam_w * tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.pow(w_del,2),2))))
  if not(FLAGS.if_a_sfm):
    print('L1 loss for a')
    regularization += FLAGS.lam_a*tf.reduce_sum(tf.abs(a)) # should be lam_w

  if FLAGS.if_a_sfm:
    print('Entropy loss for a_sfm')
    regularization += FLAGS.lam_a*tf.reduce_sum(-tf.mul(a_sfm, tf.log(a_sfm)))

  # add regularization to get final loss function
  loss = tf.add_n([loss_inter, regularization], name="final_loss")
  # add loss term to graph collection. This helps optimizer in finding which loss to minimie
  tf.get_default_graph().add_to_collection('losses', loss)


  ## Make summary for tensorboard
  # create summary writers
  # create histogram summary for all parameters which are learnt

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


  a_sfm_expanded = tf.expand_dims(a_sfm, 0)
  a_sfm_expanded = tf.expand_dims(a_sfm_expanded, -1)
  tf.image_summary('a_sfm', a_sfm_expanded)

  # loss summary
  l_summary = tf.scalar_summary('loss',loss)
  # loss without regularization summary
  l_inter_summary = tf.scalar_summary('loss_inter',loss_inter)

  global probe_nodes
  probe_nodes = [stim_time_filtered, su_act_in]
  return lam



def stim4D_to_lam_complex(stim4D, a_sfm, bias_su, bias_cell, w_stim_lr,
                          w_mother, w_del, stim_downsample_stride, stride, mask_tf, dimx, dimy):

  ## Go from time filtered stimulus to firing rate of cells

  # smoothen the stimulus by a convolution with "w_stim_lr"
  stim_smooth_lr = tf.nn.conv2d(stim4D, w_stim_lr,
                                strides=[1, stim_downsample_stride,
                                         stim_downsample_stride, 1],
                                padding='VALID')

  # further reduce dimensionality with max-pooling
  stim_maxpool = tf.nn.max_pool(stim_smooth_lr,
                                [1, 2*FLAGS.window_maxpool+1, 2*FLAGS.window_maxpool+1, 1],
                                [1, FLAGS.stride_maxpool, FLAGS.stride_maxpool, 1],
                                padding='VALID')

  # convolve with mother cell
  stim_convolved = tf.squeeze(tf.nn.conv2d(stim_maxpool, w_mother,
                                           strides=[1, stride, stride, 1],
                                           padding="VALID"))

  # extract stimulus in windows and multiply by delta weights
  stim_masked = tf.nn.conv2d(stim_maxpool, mask_tf,
                             strides=[1, stride, stride, 1],
                             padding="VALID" )
  stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)
  total_su_input = stim_del + stim_convolved

  # activation of different subunits
  su_act_in = (total_su_input + bias_su)
  su_act_flatten = tf.reshape(tf.nn.relu(su_act_in), [-1, dimx*dimy])

  # get firing rate of cells
  lam = tf.matmul(su_act_flatten, a_sfm) + bias_cell + 0.000001

  return lam, su_act_in


def get_su_cell_overlap(n_cells, window, stride,
                       stim_downsample_window, stim_downsample_stride):
  ## Pass a short stimulus corresponding where
  # only pixels in the STA of a cell is high for a short time,
  # and see which subunits were activated.
  # Based on this, select which cells should be connected to which subunits.


  gra = tf.Graph()
  with gra.as_default():
    with tf.Session() as sess:


      ## Get stimulus parameters for getting low resolution version of stimulus
      _, dimx_lr, dimy_lr, _ = get_windows(stim_downsample_window,
                                       stim_downsample_stride, n_channels=1)

      # maxpool
      _, dimx_maxpool, dimy_maxpool, _ = get_windows(FLAGS.window_maxpool,
                                                 FLAGS.stride_maxpool,
                                                 n_channels=1,
                                                 d1=dimx_lr, d2=dimy_lr)

      # get window locations
      mask_tf, dimx, dimy, n_pix = get_windows(window, stride,
                                           n_channels=1, d1=dimx_maxpool,
                                             d2=dimy_maxpool)

      su_cell_mask  = np.zeros((dimx * dimy, n_cells))

      # variables in graph
      stim4D= tf.placeholder(tf.float32, shape=[1, 640, 320, 3])
      a_sfm = tf.constant(np.ones((dimx*dimy, n_cells), dtype='float32'))
      bias_su = tf.constant(np.array(0*np.random.rand(dimx, dimy), dtype='float32'), name="bias_su")
      bias_cell = tf.constant(np.array(0*np.random.rand(n_cells), dtype='float32'), name="bias_cells")
      w_mother = tf.constant(np.array( 1+ 0*np.random.randn(2 * window + 1,
                                              2 * window + 1, 1, 1),
                                    dtype='float32'), name='w_mother_dummy')

      # subunit specific modifications to each window
      w_del = tf.Variable(np.array( 0*np.random.randn(dimx, dimy, n_pix),
                                 dtype='float32'), name='w_del_dummy')
  
      w_stim_lr = tf.Variable(np.array(1+ 0*np.random.randn(2*stim_downsample_window+1,
                                              2*stim_downsample_window+1,3,1),
                                     dtype='float32'), name="w_stim_lr")

      _, su_act_in = stim4D_to_lam_complex(stim4D, a_sfm, bias_su, bias_cell,
                                             w_stim_lr, w_mother, w_del,
                                             stim_downsample_stride, stride,
                                             mask_tf, dimx, dimy)
      mask = get_data_mat.get_cell_masks()
      for icell in np.arange(n_cells):

        mask_up = -np.repeat(np.repeat(np.repeat(np.expand_dims(mask[:,:, icell]
                                                               ,2), 8, 0),8,1), 1, 2)
        mask_up[:,:,0:-10]=0
        mask_up = np.repeat(np.expand_dims(mask_up,3),3,3)
        mask_up = np.transpose(mask_up, (2, 0, 1,3))
        mask_up =np.array(mask_up, dtype='float32')
        feed_dict_mask = {stim4D : mask_up }
        sess.run(tf.initialize_all_variables())
        su_act_mask_np = sess.run(su_act_in, feed_dict_mask)
        su_mask = np.double(np.squeeze(np.ndarray.flatten(su_act_mask_np))>500000)

        print(icell, np.sum(np.ndarray.flatten(su_mask)))
        su_cell_mask[:, icell] = su_mask

        #plt.imshow(np.reshape(su_cell_mask[:, icell], (dimx, dimy)))
        #plt.show()
        #plt.draw()

        #from IPython.terminal.embed import InteractiveShellEmbed
        #ipshell = InteractiveShellEmbed()
        #ipshell()

      return su_cell_mask


def decode_op_complex(sess, stim4D, sux, suy, vars_l, window, stride,
                       stim_downsample_window, stim_downsample_stride, dimx_lr, dimy_lr, dimx, dimy, n_cells, max_element='su'):
  w_stim_lr = vars_l.w_stim_lr
  # Get normalized stimulus to maximize subunit (or cell) activation - will remove


  # maxpool
  _, dimx_maxpool, dimy_maxpool, _ = get_windows(FLAGS.window_maxpool, FLAGS.stride_maxpool, n_channels=1, d1=dimx_lr, d2=dimy_lr)

  # get window locations
  mask_tf, dimx, dimy, n_pix = get_windows(window, stride,
                                             n_channels=1, d1=dimx_maxpool,
                                             d2=dimy_maxpool)
  w_mother = vars_l.w_mother
  w_del = vars_l.w_del
  a = vars_l.a
  a_sfm = tf.transpose(tf.nn.softmax(tf.transpose(a)))
  bias_su = vars_l.bias_su
  bias_cell = vars_l.bias_cell

  lam, su_act_in = stim4D_to_lam_complex(stim4D, a_sfm, bias_su, bias_cell, w_stim_lr,
                                w_mother, w_del, stim_downsample_stride, stride, mask_tf, dimx, dimy)

  if max_element=='su':
    print('maximize for su')
    def maximize_stim_for_su():
      su_chosen_act = su_act_in[0, sux, suy]
      maximize_su = tf.train.AdagradOptimizer(0.1).minimize(-su_chosen_act,
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
      maximize_su = tf.train.AdagradOptimizer(0.1).minimize(-cell_act,
                                                                    var_list=
                                                                   [stim4D])
      stim_normalize = tf.assign(stim4D, tf.nn.l2_normalize(stim4D, dim=[0,1,2,3]))
      sess.run(tf.initialize_all_variables())
      print('decoding stim for a subunit at', sux, suy)

      _ = sess.run(stim_normalize)
      for istep in range(500):
        _ = sess.run(maximize_su)
        _ = sess.run(stim_normalize)
        s4d = sess.run(stim4D)
        cell_act_np = sess.run(cell_act)
        print('stim4d = %s (%f)' % (s4d, np.linalg.norm(s4d)))
        print(istep, cell_act_np)
      return sess.run(stim4D), cell_act_np

    return maximize_stim_for_cell



def calculate_STA_su_cell(a_fit, w_stim_lr_fit, w_mother_fit, w_del_fit,
                          bias_su_fit, bias_cell_fit,sta_batchsz = 100):
    #  STA calculation for ALL subunits and cells -
    # remake the graph using different batch size!
  print('compute STA for ALL subunits and cells')
  from datetime import datetime
  import time
  gra = tf.Graph()
  with gra.as_default():
    with tf.Session() as sess2:
      stim4D_artificial = tf.placeholder(tf.float32,
                                         shape=(sta_batchsz,640,320,3),
                                         name="artificial_wn")
      # make constants (for variables in previous model)
      a_fit_tf = tf.constant(np.array(a_fit,dtype='float32'))
      if (FLAGS.if_a_sfm):
        a_sfm_fit_tf = tf.transpose(tf.nn.softmax(tf.transpose(a_fit_tf)))
      else:
        a_sfm_fit_tf = a_fit_tf
      w_stim_lr_fit_tf = tf.constant(w_stim_lr_fit)
      w_mother_fit_tf = tf.constant(w_mother_fit)
      w_del_fit_tf = tf.constant(w_del_fit)
      bias_su_fit_tf = tf.constant(bias_su_fit)
      bias_cell_fit_tf = tf.constant(bias_cell_fit)
      ## Get stimulus parameters for getting low resolution version of stimulus
      _, dimx_lr, dimy_lr, _ = get_windows(FLAGS.stim_downsample_window,
                                       FLAGS.stim_downsample_stride, n_channels=1)

      # maxpool
      _, dimx_maxpool, dimy_maxpool, _ = get_windows(FLAGS.window_maxpool,
                                                 FLAGS.stride_maxpool,
                                                 n_channels=1,
                                                 d1=dimx_lr, d2=dimy_lr)

      # get window locations
      mask_tf, dimx, dimy, n_pix = get_windows(FLAGS.window, FLAGS.stride,
                                           n_channels=1, d1=dimx_maxpool,
                                             d2=dimy_maxpool)
      
      lam , su_act_in = stim4D_to_lam_complex(stim4D_artificial, a_sfm_fit_tf,
                                              bias_su_fit_tf, bias_cell_fit_tf,
                                               w_stim_lr_fit_tf, w_mother_fit_tf,
                                              w_del_fit_tf,
                                               FLAGS.stim_downsample_stride,
                                              FLAGS.stride,
                                               mask_tf, dimx, dimy)

      sta_su = tf.reshape(tf.matmul(tf.reshape(tf.transpose(stim4D_artificial, [1,2,3,0]),
                                            [640*320*3, sta_batchsz]),
                      tf.reshape(su_act_in, [-1, dimx*dimy])), [640, 320, 3, dimx, dimy])
      sta_su_np = np.zeros((640, 320, 3, dimx, dimy))

      sta_cell = tf.reshape(tf.matmul(tf.reshape(tf.transpose(stim4D_artificial, [1,2,3,0]),
                                      [640*320*3, sta_batchsz]),
                                      tf.reshape(lam, [-1, FLAGS.n_cells])),
                                      [640, 320, 3, FLAGS.n_cells])

      sta_cell_np = np.zeros((640, 320, 3, FLAGS.n_cells))
      n_batches= 500

      for ibatch in np.arange(n_batches):
        # generate random stimulus sample
        start_time = time.time()
        stim_np = np.array( np.random.randn(sta_batchsz , 640, 320, 3),
                           dtype='float32')
        duration = time.time() - start_time
        format_str = ('%s: generate_random_samples @ step %d, %.3f '
                      'sec/batch)')
        print(format_str % (datetime.now(), ibatch, duration))

        # Compute STA
        start_time = time.time()
        sta_su_np_batch, sta_cell_np_batch =  sess2.run([sta_su, sta_cell],
                                                       feed_dict=
                                                       {stim4D_artificial: stim_np})
        sta_su_np += sta_np_su_batch
        sta_cell_np += sta_cell_np_batch
        duration = time.time() - start_time
        format_str = ('%s: compute STA @ step %d, %.3f '
                     'sec/batch)')
        print(format_str % (datetime.now(), ibatch, duration))

      sta_su_np = sta_su_np / n_batches
      sta_cell_np = sta_cell_np / n_batches
      return sta_su_np, sta_cell_np

if __name__ == '__main__':
  app.run()
