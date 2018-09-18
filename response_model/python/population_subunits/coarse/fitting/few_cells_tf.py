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
"""Few cells model.
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

import numpy as np, h5py
import scipy.io as sio
from scipy import ndimage
import random

FLAGS = flags.FLAGS

flags.DEFINE_string('folder_name', 'experiment4', 'folder where to store all the data')

flags.DEFINE_string('save_location',
                    'home/bhaishahster/',
                    'where to store logs and outputs?');
flags.DEFINE_string('data_location',
                    '/home/bhaishahster/data_breakdown/',
                    'where to take data from?')
flags.DEFINE_integer('batchsz', 100, 'batch size for training')
flags.DEFINE_integer('n_chunks', 216, 'number of data chunks') # should be 216
flags.DEFINE_integer('n_b_in_c', 10, 'number of batches in one chunk of data')
flags.DEFINE_integer('np_randseed', 23, 'numpy RNG seed')
flags.DEFINE_integer('randseed', 65, 'python RNG seed')
flags.DEFINE_integer('ratio_SU', 2, 'ratio of subunits/cells')
flags.DEFINE_float('step_sz', 0.001, 'step size for learning algorithm')
flags.DEFINE_string('model_id', 'poisson', 'which model to fit')
flags.DEFINE_integer('train_len', 216 - 21, 'how much training length to use?')

FLAGS = flags.FLAGS

# global stimulus variables
stim_train_part = np.array([])
resp_train_part = np.array([])
chunk_order = np.array([])
cells_choose = np.array([])
chosen_mask = np.array([])

def get_test_data():
  # stimulus.astype('float32')[216000-1000: 216000-1, :]
  # response.astype('float32')[216000-1000: 216000-1, :]
  # length
  test_data_chunks = [FLAGS.n_chunks];
  for ichunk in test_data_chunks:
    filename = FLAGS.data_location + 'Off_par_data_' + str(ichunk) + '.mat'
    file_r = gfile.Open(filename, 'r')
    data = sio.loadmat(file_r)
    stim_part = data['maskedMovdd_part'].T
    resp_part = data['Y_part'].T
    test_len = stim_part.shape[0]
  #logfile.write('\nReturning test data')
  stim_part = stim_part[:, chosen_mask]
  resp_part = resp_part[:, cells_choose]
  return stim_part, resp_part, test_len


def get_next_training_batch(iteration):
  # stimulus.astype('float32')[tms[icnt: icnt+FLAGS.batchsz], :],
  # response.astype('float32')[tms[icnt: icnt+FLAGS.batchsz], :]
  # FLAGS.batchsz

  # we will use global stimulus and response variables
  global stim_train_part
  global resp_train_part
  global chunk_order

  togo = True
  while togo:
    if(iteration % FLAGS.n_b_in_c == 0):
    # load new chunk of data
      ichunk = (iteration / FLAGS.n_b_in_c) % (FLAGS.train_len-1 ) # last one chunks used for testing
      if (ichunk == 0): # shuffle training chunks at start of training data
        chunk_order = np.random.permutation(np.arange(FLAGS.train_len)) # remove first chunk - weired?
      #  if logfile != None :
      #    logfile.write('\nTraining chunks shuffled')

      if chunk_order[ichunk] + 1 != 1:
        filename = FLAGS.data_location + 'Off_par_data_' + str(chunk_order[ichunk] + 1) + '.mat'
        file_r = gfile.Open(filename, 'r')
        data = sio.loadmat(file_r)
        stim_train_part = data['maskedMovdd_part']
        resp_train_part = data['Y_part']

        ichunk = chunk_order[ichunk] + 1
        while stim_train_part.shape[1] < FLAGS.batchsz:
          #print('Need to add extra chunk')
          if (ichunk> FLAGS.n_chunks):
            ichunk = 2
          filename = FLAGS.data_location + 'Off_par_data_' + str(ichunk) + '.mat'
          file_r = gfile.Open(filename, 'r')
          data = sio.loadmat(file_r)
          stim_train_part = np.append(stim_train_part, data['maskedMovdd_part'], axis=1)
          resp_train_part = np.append(resp_train_part, data['Y_part'], axis=1)
          #print(np.shape(stim_train_part), np.shape(resp_train_part))
          ichunk = ichunk + 1

      #  if logfile != None:
      #    logfile.write('\nNew training data chunk loaded at: '+ str(iteration) + ' chunk #: ' + str(chunk_order[ichunk]))


    ibatch = iteration % FLAGS.n_b_in_c
    try:
      stim_train = np.array(stim_train_part[:,ibatch: ibatch + FLAGS.batchsz], dtype='float32').T
      resp_train = np.array(resp_train_part[:,ibatch: ibatch + FLAGS.batchsz], dtype='float32').T
      togo=False
    except:
      iteration = np.random.randint(1,100000)
      print('Load exception iteration: ' + str(iteration) + 'chunk: ' + str(chunk_order[ichunk]) + 'batch: ' + str(ibatch) )
      togo=True

  stim_train = stim_train[:, chosen_mask]
  resp_train = resp_train[:, cells_choose]
  return stim_train, resp_train, FLAGS.batchsz


def main(argv):
  print('\nCode started')

  global cells_choose
  global chosen_mask

  np.random.seed(FLAGS.np_randseed)
  random.seed(FLAGS.randseed)
  global chunk_order
  chunk_order = np.random.permutation(np.arange(FLAGS.n_chunks-1))

  ## Load data summary

  filename = FLAGS.data_location + 'data_details.mat'
  summary_file = gfile.Open(filename, 'r')
  data_summary = sio.loadmat(summary_file)
  cells = np.squeeze(data_summary['cells'])
  if FLAGS.model_id == 'poisson' or FLAGS.model_id == 'logistic' or FLAGS.model_id == 'hinge' or FLAGS.model_id == 'poisson_relu':
    cells_choose = (cells ==3287) | (cells ==3318 ) | (cells ==3155) | (cells ==3066)
  if FLAGS.model_id == 'poisson_full':
    cells_choose = np.array(np.ones(np.shape(cells)), dtype='bool')
  n_cells = np.sum(cells_choose)

  tot_spks = np.squeeze(data_summary['tot_spks'])
  total_mask = np.squeeze(data_summary['totalMaskAccept_log']).T
  tot_spks_chosen_cells = np.array(tot_spks[cells_choose] ,dtype='float32')
  chosen_mask = np.array(np.sum(total_mask[cells_choose,:],0)>0, dtype='bool')
  print(np.shape(chosen_mask))
  print(np.sum(chosen_mask))

  stim_dim = np.sum(chosen_mask)

  print(FLAGS.model_id)

  print('\ndataset summary loaded')
  # use stim_dim, chosen_mask, cells_choose, tot_spks_chosen_cells, n_cells

  # decide the number of subunits to fit
  n_su = FLAGS.ratio_SU*n_cells

  # saving details
  if FLAGS.model_id == 'poisson':
    short_filename = ('data_model=ASM_pop_batch_sz='+ str(FLAGS.batchsz) + '_n_b_in_c' + str(FLAGS.n_b_in_c) +
    '_step_sz'+ str(FLAGS.step_sz)+'_tlen=' + str(FLAGS.train_len) + '_bg')
  else:
    short_filename = ('data_model=' + str(FLAGS.model_id) + '_batch_sz='+ str(FLAGS.batchsz) + '_n_b_in_c' + str(FLAGS.n_b_in_c) +
    '_step_sz'+ str(FLAGS.step_sz) + '_tlen=' + str(FLAGS.train_len) + '_bg')

  parent_folder = FLAGS.save_location + FLAGS.folder_name + '/'
  if not gfile.IsDirectory(parent_folder):
    gfile.MkDir(parent_folder)
  FLAGS.save_location = parent_folder +short_filename + '/'
  print(gfile.IsDirectory(FLAGS.save_location))
  if not gfile.IsDirectory(FLAGS.save_location):
    gfile.MkDir(FLAGS.save_location)
  print(FLAGS.save_location)
  save_filename = FLAGS.save_location + short_filename


  with tf.Session() as sess:
    # Learn population model!
    stim = tf.placeholder(tf.float32, shape=[None, stim_dim], name='stim')
    resp = tf.placeholder(tf.float32, name='resp')
    data_len = tf.placeholder(tf.float32, name='data_len')

    if FLAGS.model_id == 'poisson' or FLAGS.model_id == 'poisson_full':
      # variables
      w = tf.Variable(np.array(0.01 * np.random.randn(stim_dim, n_su), dtype='float32'))
      a = tf.Variable(np.array(0.01 * np.random.rand(n_cells, 1, n_su), dtype='float32'))

      lam = tf.transpose(tf.reduce_sum(tf.exp(tf.matmul(stim, w) + a), 2))
      #loss_inter = (tf.reduce_sum(lam/tot_spks_chosen_cells)/120. - tf.reduce_sum(resp*tf.log(lam)/tot_spks_chosen_cells)) / data_len
      loss_inter = (tf.reduce_sum(lam)/120. - tf.reduce_sum(resp*tf.log(lam))) / data_len
      loss = loss_inter
      train_step = tf.train.AdagradOptimizer(FLAGS.step_sz).minimize(loss, var_list=[w, a])
      def training(inp_dict):
        sess.run(train_step, feed_dict=inp_dict)
      def get_loss(inp_dict):
        ls = sess.run(loss,feed_dict = inp_dict)
        return ls

    if FLAGS.model_id == 'poisson_relu':
      # variables
      w = tf.Variable(np.array(0.01 * np.random.randn(stim_dim, n_su), dtype='float32'))
      a = tf.Variable(np.array(0.01 * np.random.rand(n_su, n_cells), dtype='float32'))
      b_init =  np.log((tot_spks_chosen_cells)/(216000. - tot_spks_chosen_cells))
      b = tf.Variable(b_init,dtype='float32')
      f = tf.matmul(tf.exp(tf.nn.relu(stim, w)), a) + b     #loss_inter = (tf.reduce_sum(lam/tot_spks_chosen_cells)/120. - tf.reduce_sum(resp*tf.log(lam)/tot_spks_chosen_cells)) / data_len
      loss_inter = (tf.reduce_sum(f)/120. - tf.reduce_sum(resp*tf.log(f))) / data_len
      loss = loss_inter
      train_step = tf.train.AdagradOptimizer(FLAGS.step_sz).minimize(loss, var_list=[w, a])
      def training(inp_dict):
        sess.run(train_step, feed_dict=inp_dict)
      def get_loss(inp_dict):
        ls = sess.run(loss,feed_dict = inp_dict)
        return ls

    if FLAGS.model_id == 'logistic':
      w = tf.Variable(np.array(0.01 * np.random.randn(stim_dim, n_su), dtype='float32'))
      a = tf.Variable(np.array(0.01 * np.random.rand(n_su, n_cells), dtype='float32'))
      b_init =  np.log((tot_spks_chosen_cells)/(216000. - tot_spks_chosen_cells))
      b = tf.Variable(b_init,dtype='float32')
      f = tf.matmul(tf.nn.relu(tf.matmul(stim, w)), a) + b
      loss_inter = tf.reduce_sum(tf.nn.softplus(-2 * (resp - 0.5)*f)) / data_len
      loss = loss_inter
      train_step = tf.train.AdagradOptimizer(FLAGS.step_sz).minimize(loss, var_list=[w, a, b])
      a_pos = tf.assign(a, (a + tf.abs(a))/2)
      def training(inp_dict):
        sess.run(train_step, feed_dict=inp_dict)
        sess.run(a_pos)
      def get_loss(inp_dict):
        ls = sess.run(loss,feed_dict = inp_dict)
        return ls

    if FLAGS.model_id == 'hinge':
      w = tf.Variable(np.array(0.01 * np.random.randn(stim_dim, n_su), dtype='float32'))
      a = tf.Variable(np.array(0.01 * np.random.rand(n_su, n_cells), dtype='float32'))
      b_init =  np.log((tot_spks_chosen_cells)/(216000. - tot_spks_chosen_cells))
      b = tf.Variable(b_init,dtype='float32')
      f = tf.matmul(tf.nn.relu(tf.matmul(stim, w)), a) + b
      loss_inter = tf.reduce_sum(tf.nn.relu(1 -2 * (resp - 0.5)*f))  / data_len
      loss = loss_inter
      train_step = tf.train.AdagradOptimizer(FLAGS.step_sz).minimize(loss, var_list=[w, a, b])
      a_pos = tf.assign(a, (a + tf.abs(a))/2)
      def training(inp_dict):
        sess.run(train_step, feed_dict=inp_dict)
        sess.run(a_pos)
      def get_loss(inp_dict):
        ls = sess.run(loss,feed_dict = inp_dict)
        return ls



    # summaries
    l_summary = tf.scalar_summary('loss',loss)
    l_inter_summary = tf.scalar_summary('loss_inter',loss_inter)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.save_location + 'train',
                                      sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.save_location + 'test')

    print('\nStarting new code')

    sess.run(tf.initialize_all_variables())
    saver_var = tf.train.Saver(tf.all_variables(), keep_checkpoint_every_n_hours=0.05)
    load_prev = False
    start_iter=0
    try:
      latest_filename = short_filename + '_latest_fn'
      restore_file = tf.train.latest_checkpoint(FLAGS.save_location, latest_filename)
      start_iter = int(restore_file.split('/')[-1].split('-')[-1])
      saver_var.restore(sess, restore_file)
      load_prev = True
    except:
      print('No previous dataset')

    if load_prev:
      print('\nPrevious results loaded')
    else:
      print('\nVariables initialized')
    # Do the fitting

    icnt = 0
    stim_test,resp_test,test_length = get_test_data()
    fd_test = {stim: stim_test,
               resp: resp_test,
               data_len: test_length}

    #logfile.close()

    for istep in np.arange(start_iter,400000):

      print(istep)
      # get training data
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
        saver_var.save(sess, save_filename, global_step=istep, latest_filename = latest_filename)

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

