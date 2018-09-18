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
"""One-line documentation for approximate_conv module.

A detailed description of approximate_conv.
"""

from datetime import datetime
import time
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
import retina.response_model.python.population_subunits.jitter.distributed.jitter_model as jitter_model
import retina.response_model.python.population_subunits.jitter.distributed.get_data_mat as get_data_mat
import random


FLAGS = flags.FLAGS
# flags for data locations
flags.DEFINE_string('folder_name', 'experiment_jitter',
                    'folder where to store all the data')
flags.DEFINE_string('save_location',
                    '/home/bhaishahster/distributed3/',
                    'where to store logs and outputs?');
flags.DEFINE_string('data_location',
                    '/home/retina_data/Google datasets/2016-04-21-1/data006(2016-04-21-1_data006_data006)/',
                    'where to take data from?')
flags.DEFINE_integer('batchsz', 240*4, 'batch size for training')
flags.DEFINE_integer('n_chunks',1793, 'number of data chunks') # should be 216
flags.DEFINE_integer('num_chunks_to_load', 2*6,
                     'number of chunks to load for 1 batch of data')
flags.DEFINE_integer('train_len', 216 - 21, 'how much training length to use?')
flags.DEFINE_float('step_sz', 20, 'step size for learning algorithm')
flags.DEFINE_integer('max_steps', 400000, 'maximum number of steps')

# random number generators initialized
# removes unneccessary data variabilities while comparing algorithms
flags.DEFINE_integer('np_randseed', 23, 'numpy RNG seed')
flags.DEFINE_integer('randseed', 65, 'python RNG seed')


# flags for model/loss specification
flags.DEFINE_string('model_id', 'relu_window_mother_sfm', 'which model to fit')
## list of models here, and quick explanation
flags.DEFINE_string('loss', 'poisson', 'which loss to use?')
# poisson, (conditional poisson - TODO), logistic or hinge

# model specific terms
# useful for convolution-like models
flags.DEFINE_string('architecture','1 layer',
                    'the architecture of model to be learnt')
# options : 1 layer, 2 layer_stimulus (stimulus put to lower dimensions),
# 2 layer_delta (two layered architecture of delta weights)
# stimulus downsampling options - if architecture = '2 layer_stimulus',
# then downsample stimulus with these windows and strides.
flags.DEFINE_integer('stim_downsample_window', 4,
                     'How to down sample the stimulus')
flags.DEFINE_integer('stim_downsample_stride',4,
                     'stride to use to downsample stimulus')



# weight windows on stimulus for subunits
flags.DEFINE_integer('window', 16,
                     'size of window for each subunit in relu_window model')
flags.DEFINE_integer('stride', 16,
                     'stride for relu_window')
flags.DEFINE_integer('su_channels', 3,
                     'number of color channels each subunit should take input from')

# some models need regularization of parameters
flags.DEFINE_float('lam_w', 0.000, 'sparsitiy regularization of w')
flags.DEFINE_float('lam_a', 0.000, 'sparsitiy regularization of a')

# dataset specific
flags.DEFINE_float('n_cells',1, 'number of cells in the dataset')

# distributed TF specific flags
flags.DEFINE_string("master", "local",
                           """BNS name of the TensorFlow master to use.""")
flags.DEFINE_integer("task", 0,
                            """Task id of the replica running the training.""")
flags.DEFINE_integer("ps_tasks", 0,
                            """Number of tasks in the ps job.
                            If 0 no ps job is used.""")

#flags.DEFINE_integer("is_eval", 0, """If this is eval worker""")

# specs for multi-gpu training
tf.app.flags.DEFINE_string('config_params', '',
                           """Deployment config params.""")

# parameters used for synchronous updating of gradients from multiple workers
flags.DEFINE_boolean("sync_replicas", False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")

flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")

# learn or analyze a particular fit?
flags.DEFINE_integer("learn",1,"""If we learn the model or analyse it""")
FLAGS = flags.FLAGS


def main(argv):
  RunComputation()


def RunComputation():


  # filename for saving file
  if FLAGS.architecture == '2 layer_stimulus':
    architecture_string = ('_architecture=' + str(FLAGS.architecture) +
                           '_stim_downsample_window=' +
                           str(FLAGS.stim_downsample_window) +
                           '_stim_downsample_stride=' +
                           str(FLAGS.stim_downsample_stride))
  else:
    architecture_string = ('_architecture=' + str(FLAGS.architecture))

  short_filename = ('model=' + str(FLAGS.model_id) + '_loss='+
                    str(FLAGS.loss) + '_batch_sz='+ str(FLAGS.batchsz) +
                    '_lam_w=' + str(FLAGS.lam_w) +
                    '_step_sz'+ str(FLAGS.step_sz) +
                    '_tlen=' + str(FLAGS.train_len) +
                    '_window='+str(FLAGS.window) +
                    '_stride='+str(FLAGS.stride) +
                    str(architecture_string) + '_jitter')


  # make a folder with name derived from parameters of the algorithm - it saves checkpoint files and summaries used in tensorboard
  parent_folder = FLAGS.save_location + FLAGS.folder_name + '/'
  # make folder if it does not exist
  if not gfile.IsDirectory(parent_folder):
    gfile.MkDir(parent_folder)
  FLAGS.save_location = parent_folder + short_filename + '/'
  print('Does the file exist?', gfile.IsDirectory(FLAGS.save_location))
  if not gfile.IsDirectory(FLAGS.save_location):
    gfile.MkDir(FLAGS.save_location)

  save_filename = FLAGS.save_location + short_filename


  """Main function which runs all TensorFlow computations."""
  with tf.Graph().as_default() as gra:
    with tf.device(tf.ReplicaDeviceSetter(FLAGS.ps_tasks)):
      print(FLAGS.config_params)
      tf.logging.info(FLAGS.config_params)
      # set up training dataset
      tc_mean = get_data_mat.init_chunks(FLAGS.n_chunks)

      '''
      # plot histogram of a training dataset
      stim_train, resp_train, train_len = get_data_mat.get_stim_resp('train',
                                                                     num_chunks=FLAGS.num_chunks_to_load)
      plt.hist(np.ndarray.flatten(stim_train[:,:,0:]))
      plt.show()
      plt.draw()
      '''
      # Create computation graph.
      #
      # Graph should be fully constructed before you create supervisor.
      # Attempt to modify graph after supervisor is created will cause an error.



      with tf.name_scope('model'):
        if FLAGS.architecture == '1 layer':
          # single GPU model
          if False:
            global_step = tf.contrib.framework.create_global_step()
            model, stim, resp = jitter_model.approximate_conv_jitter(FLAGS.n_cells,
                                                                     FLAGS.lam_w,
                                                                     FLAGS.window,
                                                                     FLAGS.stride,
                                                                     FLAGS.step_sz,
                                                                     tc_mean,
                                                                     FLAGS.su_channels)

          # multiGPU model
          if True:
            model, stim, resp, global_step = jitter_model.approximate_conv_jitter_multigpu(FLAGS.n_cells,
                                FLAGS.lam_w, FLAGS.window, FLAGS.stride, FLAGS.step_sz,
                                tc_mean, FLAGS.su_channels, FLAGS.config_params)

        if FLAGS.architecture == '2 layer_stimulus':
          # stimulus is first smoothened to lower dimensions, then same model is applied
          print(' put stimulus to lower dimenstions!')
          model, stim, resp, global_step, stim_tuple = jitter_model.approximate_conv_jitter_multigpu_stim_lr(FLAGS.n_cells,
                              FLAGS.lam_w, FLAGS.window, FLAGS.stride, FLAGS.step_sz,
                              tc_mean, FLAGS.su_channels, FLAGS.config_params,FLAGS.stim_downsample_window,FLAGS.stim_downsample_stride)


      # Print the number of variables in graph
      print('Calculating model size') # Hope we do not exceed memory
      PrintModelAnalysis(gra, max_depth=10)
      #import pdb; pdb.set_trace()

      # Builds our summary op.
      summary_op = model.merged_summary


      # Create a Supervisor.  It will take care of initialization, summaries,
      # checkpoints, and recovery.
      #
      # When multiple replicas of this program are running, the first one,
      # identified by --task=0 is the 'chief' supervisor.  It is the only one
      # that takes case of initialization, etc.
      is_chief = (FLAGS.task == 0) # & (FLAGS.learn==1)
      print(save_filename)
      if FLAGS.learn==1:
        # use supervisor only for learning,
        # otherwise it messes up data as it tries to store variables while you are doing analysis

        sv = tf.train.Supervisor(logdir=save_filename,
                           is_chief=is_chief,
                           saver=tf.train.Saver(),
                           summary_op=None,
                           save_model_secs=100,
                           global_step=global_step,
                           recovery_wait_secs=5)


        if (is_chief and FLAGS.learn==1):
          tf.train.write_graph(tf.get_default_graph().as_graph_def(),
                                 save_filename, 'graph.pbtxt')


        # Get an initialized, and possibly recovered session.  Launch the
        # services: Checkpointing, Summaries, step counting.
        #
        # When multiple replicas of this program are running the services are
        # only launched by the 'chief' replica.
        session_config = tf.ConfigProto(
                          allow_soft_placement=True,
                          log_device_placement=False)
        #import pdb; pdb.set_trace()
        sess = sv.PrepareSession(FLAGS.master, config=session_config)

        FitComputation(sv, sess, model, stim, resp, global_step, summary_op, stim_tuple)
        sv.Stop()

      else:
        # if not learn, then analyse


        session_config = tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False)
        with tf.Session(config=session_config) as sess:
          saver_var = tf.train.Saver(tf.all_variables(),
                             keep_checkpoint_every_n_hours=float('inf'))
          restore_file = tf.train.latest_checkpoint(save_filename)
          print(restore_file)
          start_iter = int(restore_file.split('/')[-1].split('-')[-1])
          saver_var.restore(sess, restore_file)

          if FLAGS.architecture == '2 layer_stimulus':
            AnalyseModel_lr(sess, model)
          else:
            AnalyseModel(sv, sess, model)




def FitComputation(sv, sess, model, stim, resp, global_step, summary_op, stim_tuple):

  def Test():
    # load a test chunk one at a time and compute average log-likelihood
    loss_batch = 0
    n_test_chunks = 2 # should be 8 #len(get_data_mat.test_chunks)
    for ichunk in range(n_test_chunks):
      if get_data_mat.test_counter >=n_test_chunks:
        get_data_mat.test_counter = 0
      stim_test, resp_test, test_len = get_data_mat.get_stim_resp(data_type='test')
      fd_test = {stim: np.array(stim_test,dtype='float32'),
               resp: np.array(resp_test,dtype='float32')}
      loss_batch += sess.run(model.loss_inter, feed_dict=fd_test)
    print_loss = loss_batch / n_test_chunks
    print('Test loss:%.3f' % print_loss)
    return print_loss


  # Run iterative computation in a loop.
  step = sess.run(global_step)
  is_chief = (FLAGS.task == 0)
  loss_avg = []
  while not sv.ShouldStop():
    #print('starting step')
    start_time = time.time()
    stim_train, resp_train, train_len = get_data_mat.get_stim_resp('train',
                                                                   num_chunks=FLAGS.num_chunks_to_load)
    duration = time.time() - start_time
    format_str = ('%s: get_data @ step %d, %.3f '
                  'sec/batch)')
    tf.logging.info(format_str % (datetime.now(), step, duration))
    print(format_str % (datetime.now(), step, duration))
    #print(resp)
    #import pdb; pdb.set_trace()
    fd_train = {stim: np.array(stim_train,dtype='float32'),
                resp: np.array(resp_train,dtype='float32')}
    #print('made dict')
    #print('running step')
    start_time = time.time()
    _, current_loss = sess.run([model.train_step, model.loss_inter], feed_dict=fd_train)
    #_ = sess.run([model.train_step], feed_dict=fd_train)
    #current_loss = 0
    loss_avg.append(current_loss)
    duration = time.time() - start_time
    format_str = ('%s: train @ step %d, %.3f '
                  'sec/batch) loss = %.3f')
    tf.logging.info(format_str % (datetime.now(), step, duration, np.mean(np.array(loss_avg))))
    print(format_str % (datetime.now(), step, duration, np.mean(np.array(loss_avg))))
    if len(loss_avg) > 10:
      loss_avg = loss_avg[1:]

    #print('finished running step.')

    '''
    from IPython.terminal.embed import InteractiveShellEmbed
    ipshell = InteractiveShellEmbed()
    ipshell()
    '''
    if step >= FLAGS.max_steps:
      break
    if is_chief and step % 10 == 0:
      mean_loss = np.mean(np.array(loss_avg))
      #print('making summary')
      start_time = time.time()
      summary_str = sess.run(summary_op, feed_dict=fd_train)
      sv.summary_computed(sess, summary_str)
      duration = time.time() - start_time
      format_str = ('%s: summary @ step %d, %.3f '
                    'sec/batch), loss: %.3f')
      #tf.logging.info(format_str % (datetime.now(), step, duration, loss_inter_summary))
      #print(format_str % (datetime.now(), step, duration, mean_loss))
      loss_avg = []

      # Test data loss

      test_loss = Test()
      '''
      test_summary = tf.Summary()
      value = test_summary.value.add()
      value.tag = 'Test loss'
      value.simple_value = test_loss
      print('Test loss %.3f' % value.simple_value)
      sv.summary_computed(sess, test_summary)
      #print('adding summary')
      '''
    step += 1
    #print('ending step')





def AnalyseModel(sv, sess, model):
  # analyse a 1 layered almost-convolutional model
  print('starting analysis')

  # get mother cell and print it
  w_fit_mother = sess.run(model.variables.w_mother)
  print(np.shape(w_fit_mother))
  for ichannel in range(3):
    plt.subplot(1,3,ichannel+1)
    print(np.squeeze(w_fit_mother[:,:,ichannel,0]))
    plt.imshow(np.squeeze(w_fit_mother[:,:,ichannel,0]), cmap='gray')
  plt.draw()
  plt.show()


  # print a_sfm
  '''
  a_sfm_eval = sess.run(model.ops.a_sfm)
  plt.plot(a_sfm_eval)
  print(np.shape(a_sfm_eval))
  print(np.sum(a_sfm_eval ,0))
  plt.show()
  plt.draw()
  '''

  # plot delta subunit for 'almost convolutional - model + delta models'

  w_del_e = np.squeeze(sess.run(model.variables.w_del))
  w_mot = sess.run(model.variables.w_mother)
  dimx = model.dimensions.dimx
  dimy = model.dimensions.dimy
  print(dimx, dimy)
  for icol in np.arange(3):
    icnt=1
    for idimx in np.arange(dimx):
      print(idimx)
      for idimy in np.arange(dimy):
        w_del_flatteni = np.squeeze(w_del_e[idimx, idimy, :])
        plt.subplot(dimx, dimy, icnt)
        wts = w_del_flatteni
        wh = 2*FLAGS.window+1

        wts = np.reshape(wts[wh*wh*icol:wh*wh*(icol+1)],(wh,wh))
        plt.imshow(np.squeeze(wts + np.squeeze(w_mot[:,:,icol])),cmap='gray')
        icnt=icnt+1
        plt.axis('off')
    print(icol)
    plt.draw()
    plt.show()

  # plot subunits for a chosen cell

  w_del_e = np.squeeze(sess.run(model.variables.w_del))
  w_mot = sess.run(model.variables.w_mother)
  a_sfm_eval = sess.run(model.ops.a_sfm)
  icell = 20
  icol = 1 # 1 for green
  a_wts = a_sfm_eval[:,icell]
  a_thr = np.percentile(a_wts, 99)
  sus = np.arange(a_sfm_eval.shape[0])
  chosen_su = sus[a_wts > a_thr]
  wh = 2 * FLAGS.window + 1
  dimx = model.dimensions.dimx
  dimy = model.dimensions.dimy
  icnt=-1
  isu = 0

  print(chosen_su)
  for idimx in np.arange(dimx):
    print(idimx)
    for idimy in np.arange(dimy):
      icnt=icnt+1
      if(a_wts[icnt]>=a_thr):
        print(icnt)
        # plot this subunit
        # compute 2D subunit
        w_del_flatteni = np.squeeze(w_del_e[idimx, idimy, :])
        wts = w_del_flatteni
        wts = np.reshape(wts[wh * wh * icol:wh * wh * (icol + 1)], (wh, wh))


        isu = isu + 1
        print(isu)
        plt.subplot(len(chosen_su), 2, (isu - 1) * 2 +  1)
        plt.imshow(np.squeeze(wts + np.squeeze(w_mot[:, :, icol])), cmap='gray')
        plt.title(str(a_wts[icnt]))

        plt.subplot(len(chosen_su), 2, (isu - 1) * 2 +  2)
        plt.imshow(np.squeeze(wts),cmap='gray')
        plt.title(str(a_wts[icnt]))


  plt.show()
  plt.draw()

def AnalyseModel_lr(sess, model):
   # analyse a 1 layered almost-convolutional model
  print('starting analysis')


  # get mother cell and print it
  w_fit_mother = sess.run(model.variables_lr.w_mother)
  print(np.shape(w_fit_mother))
  for ichannel in range(1):
    plt.subplot(1,1,ichannel+1)
    print(np.squeeze(w_fit_mother[:,:,ichannel,0]))
    plt.imshow(np.squeeze(w_fit_mother[:,:,ichannel,0]), cmap='gray',
               interpolation='nearest')
  plt.title('Mother subunit')
  plt.draw()
  plt.show()



  # see how the stimulus is put to lower dimensions - plot w_stim_lr
  w_fit_stim_lr = sess.run(model.variables_lr.w_stim_lr)
  print(np.shape(w_fit_stim_lr))
  for ichannel in range(3):
    plt.subplot(1,3,ichannel+1)
    print(np.squeeze(w_fit_stim_lr[:,:,ichannel,0]))
    plt.imshow(np.squeeze(w_fit_stim_lr[:,:,ichannel,0]), cmap='gray',
               interpolation='nearest')
  plt.title('w_stimlr: putting stimulus to lower dimensions')
  plt.draw()
  plt.show()



  # plot delta subunit for 'almost convolutional - model + delta models'
  '''
  w_del_e = np.squeeze(sess.run(model.variables_lr.w_del))
  w_mot = sess.run(model.variables_lr.w_mother)
  dimx = model.dimensions.dimx
  dimy = model.dimensions.dimy
  print(dimx, dimy)
  for icol in np.arange(1):
    icnt=1
    for idimx in np.arange(dimx):
      print(idimx)
      for idimy in np.arange(dimy):
        w_del_flatteni = np.squeeze(w_del_e[idimx, idimy, :])
        plt.subplot(dimx, dimy, icnt)
        #plt.subplot(6,6,icnt)
        wts = w_del_flatteni
        wh = 2*FLAGS.window+1

        wts = np.reshape(wts[wh*wh*icol:wh*wh*(icol+1)],(wh,wh))
        plt.imshow(np.squeeze(wts + np.squeeze(w_mot[:,:,icol])),cmap='gray',
                   interpolation='nearest')
        icnt=icnt+1
    plt.suptitle('w mother + w delta')
    print(icol)

    plt.draw()
    plt.show()
  '''

  # plot subunits for a chosen cell

  w_del_e = np.squeeze(sess.run(model.variables_lr.w_del))
  w_mot = sess.run(model.variables_lr.w_mother)
  a_model = sess.run(model.variables_lr.a)
  a_sfm_eval = a_model
  icell = 35
  icol = 0 # 1 for green
  a_wts = a_sfm_eval[:,icell]
  a_thr = np.percentile(a_wts, 99.9)
  sus = np.arange(a_sfm_eval.shape[0])
  chosen_su = sus[a_wts > a_thr]
  wh = 2 * FLAGS.window + 1
  dimx = model.dimensions.dimx
  dimy = model.dimensions.dimy
  icnt=-1
  isu = 0

  print(chosen_su)
  for idimx in np.arange(dimx):
    print(idimx)
    for idimy in np.arange(dimy):
      icnt=icnt+1
      if(a_wts[icnt]>=a_thr):

        good_sux=idimx
        good_suy=idimy
        print(icnt, idimx, idimy, a_wts[icnt])
        # plot this subunit
        # compute 2D subunit
        w_del_flatteni = np.squeeze(w_del_e[idimx, idimy, :])
        wts = w_del_flatteni
        wts = np.reshape(wts[wh * wh * icol:wh * wh * (icol + 1)], (wh, wh))

        isu = isu + 1
        print(isu)
        ax=plt.subplot(len(chosen_su), 2, (isu - 1) * 2 +  1)
        plt.imshow(np.squeeze(wts + np.squeeze(w_mot[:, :, icol])), cmap='gray',
                   interpolation='nearest')
        #plt.title(str(a_wts[icnt]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax=plt.subplot(len(chosen_su), 2, (isu - 1) * 2 +  2)
        plt.imshow(np.squeeze(wts),cmap='gray', interpolation='nearest')
        #plt.title(str(a_wts[icnt]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

  plt.show()
  plt.draw()


  ## Decode stimulus for each subunit
  print('Starting decode')

  w_stim_lr_fit = sess.run(model.variables_lr.w_stim_lr)
  w_mother_fit = sess.run(model.variables_lr.w_mother)
  a_fit = sess.run(model.variables_lr.a)
  w_del_fit = sess.run(model.variables_lr.w_del)
  bias_su_fit = sess.run(model.variables_lr.bias_su)
  bias_cell_fit = sess.run(model.variables_lr.bias_cell)
  tcmean_fit = sess.run(model.variables_lr.time_course)
  sux = good_sux
  g = tf.Graph()
  with g.as_default():
    with tf.Session() as sess2:
      for suy in [good_suy]:#np.arange(30):
        # plot a
        a = tf.constant(np.array(a_fit, dtype='float32'))
        a_sfm = tf.transpose(tf.nn.softmax(tf.transpose(a)))
        a_sfm_expanded = tf.expand_dims(a_sfm, 0)
        a_sfm_expanded = tf.expand_dims(a_sfm_expanded, -1)
        a_sfm_np = sess2.run(a_sfm_expanded)
        plt.imshow(np.squeeze(a_sfm_np), cmap='gray', interpolation='nearest')
        plt.show()
        plt.draw()

        # maximize stimulus for a particular su
        vars_lst = jitter_model.variables_lr(w_mother_fit, w_del_fit, a_fit, w_stim_lr_fit,bias_su_fit ,bias_cell_fit, tcmean_fit)
        np.random.seed(11111)
        stim4D = tf.Variable(np.array(np.random.randn(1,640,320,3), dtype='float32'), name="decoded_stimulus")
        decode_fcn = jitter_model.decode_op_stim_lr(sess2, stim4D, sux, suy, vars_lst, FLAGS.window, FLAGS.stride,
                          FLAGS.stim_downsample_window,
                          FLAGS.stim_downsample_stride,
                          model.dimensions_stimlr.dimx_slr,
                          model.dimensions_stimlr.dimy_slr,
                          model.dimensions.dimx, model.dimensions.dimy, FLAGS.n_cells)
        stim_decode, max_val = decode_fcn()

        print(np.shape(stim_decode))
        icol =0
        plt.subplot(1,2,1)
        plt.imshow(np.squeeze(stim_decode[0,:,:,icol]), cmap='gray', interpolation='nearest')

        xx = np.squeeze(stim_decode[0,:,:,icol])
        rc = np.nonzero(xx>0.8*np.max(np.ndarray.flatten(xx)))
        xxy = xx[np.min(rc[0]):np.max(rc[0]), np.min(rc[1]):np.max(rc[1])]
        plt.subplot(1,2,2)
        plt.imshow(xxy, cmap='gray', interpolation='nearest')
        plt.title('Max val: '+ str(max_val))
        plt.show()
        plt.draw()


        # maximize stimulus for a particular cell

        for mcellid in [20]:#np.arange(49): # which cell ID to plot
          np.random.seed(11111)
          stim4D = tf.Variable(np.array(np.random.randn(1,640,320,3), dtype='float32'), name="decoded_stimulus")
          decode_fcn = jitter_model.decode_op_stim_lr(sess2, stim4D, mcellid, -1, vars_lst, FLAGS.window, FLAGS.stride,
                            FLAGS.stim_downsample_window,
                            FLAGS.stim_downsample_stride,
                            model.dimensions_stimlr.dimx_slr,
                            model.dimensions_stimlr.dimy_slr,
                            model.dimensions.dimx, model.dimensions.dimy, FLAGS.n_cells, max_element='cell')
          stim_decode, max_val = decode_fcn()

          print(np.shape(stim_decode))
          icol =1
          #plt.subplot(7, 7, mcellid+1);
          plt.imshow(np.squeeze(stim_decode[0, :, :, icol]), cmap='gray', interpolation='nearest')
        plt.show()
        plt.draw()

  '''
  from IPython.terminal.embed import InteractiveShellEmbed
  ipshell = InteractiveShellEmbed()
  ipshell()
  '''

if __name__ == '__main__':
  app.run()
