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
"""One-line documentation for few_cells_tf_analyse_roc module.

A detailed description of few_cells_tf_analyse_roc.
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
from matplotlib import pylab
import matplotlib.pyplot as plt

import numpy as np, h5py
import scipy.io as sio
from scipy import ndimage
import random
import re # regular expression matching

FLAGS = flags.FLAGS

flags.DEFINE_string('folder_name', 'experiment4', 'folder where to store all the data')

flags.DEFINE_string('save_location',
                    '/home/bhaishahster/',
                    'where to store logs and outputs?');
flags.DEFINE_string('data_location',
                    '/home/bhaishahster/data_breakdown/',
                    'where to take data from?')
flags.DEFINE_integer('batchsz', 500, 'batch size for training')
flags.DEFINE_integer('n_chunks', 216, 'number of data chunks') # should be 216
flags.DEFINE_integer('n_b_in_c', 2, 'number of batches in one chunk of data')
flags.DEFINE_integer('np_randseed', 23, 'numpy RNG seed')
flags.DEFINE_integer('randseed', 65, 'python RNG seed')
flags.DEFINE_integer('ratio_SU', 2, 'ratio of subunits/cells')
flags.DEFINE_float('step_sz', 1, 'step size for learning algorithm')
flags.DEFINE_string('model_id', 'poisson', 'which model to fit')
FLAGS = flags.FLAGS

# global vars
cells_choose = np.array([])
chosen_mask = np.array([])

def get_test_data():
  # stimulus.astype('float32')[216000-1000: 216000-1, :]
  # response.astype('float32')[216000-1000: 216000-1, :]
  # length
  global chosen_mask
  global cells_choose
  stim_part = np.array([]).reshape(0,np.sum(chosen_mask))
  resp_part = np.array([]).reshape(0,np.sum(cells_choose))

  test_data_chunks = np.arange(FLAGS.n_chunks-20, FLAGS.n_chunks+1);
  for ichunk in test_data_chunks:
    filename = FLAGS.data_location + 'Off_par_data_' + str(ichunk) + '.mat'
    file_r = gfile.Open(filename, 'r')
    data = sio.loadmat(file_r)

    s = data['maskedMovdd_part'].T
    r = data['Y_part'].T
    print(np.shape(s))
    print(np.shape(stim_part))
    stim_part = np.append(stim_part,s[:, chosen_mask] , axis=0)
    resp_part = np.append(resp_part,r[:, cells_choose] , axis=0)

    test_len = stim_part.shape[0]
  return stim_part, resp_part, test_len

def get_latest_file(save_location, short_filename): # get relevant files
  file_list = gfile.ListDirectory(save_location)
  print(save_location, short_filename)
  save_filename = save_location + short_filename
  print('\nLoading: ', save_filename)
  bin_files = []
  meta_files = []
  for file_n in file_list:
    if re.search(short_filename + '.', file_n):
      if re.search('.meta', file_n):
        meta_files += [file_n]
      else:
        bin_files += [file_n]
    # print(bin_files)
  print(len(meta_files), len(bin_files), len(file_list))

    # get iteration numbers
  iterations = np.array([])
  for file_name in bin_files:
    try:
      iterations = np.append(iterations, int(file_name.split('/')[-1].split('-')[-1]))
    except:
      print('Could not load filename: ' + file_name)
  iterations.sort()
  print(iterations)

  iter_plot = iterations[-1]
  print(int(iter_plot))
  restore_file = save_filename + '-' + str(int(iter_plot))
  return restore_file

def main(argv):
  print('\nCode started')

  np.random.seed(FLAGS.np_randseed)
  random.seed(FLAGS.randseed)
  global chosen_mask
  global cells_choose

  ## Load data summary

  filename = FLAGS.data_location + 'data_details.mat'
  summary_file = gfile.Open(filename, 'r')
  data_summary = sio.loadmat(summary_file)
  cells = np.squeeze(data_summary['cells'])
  cells_choose = (cells ==3287) | (cells ==3318 ) | (cells ==3155) | (cells ==3066)
  n_cells = np.sum(cells_choose)

  tot_spks = np.squeeze(data_summary['tot_spks'])
  total_mask = np.squeeze(data_summary['totalMaskAccept_log']).T
  tot_spks_chosen_cells = tot_spks[cells_choose]
  chosen_mask = np.array(np.sum(total_mask[cells_choose,:],0)>0, dtype='bool')
  print(np.shape(chosen_mask))
  print(np.sum(chosen_mask))

  stim_dim = np.sum(chosen_mask)
  # get test data
  stim_test,resp_test,test_length = get_test_data()

  print('\ndataset summary loaded')
  # use stim_dim, chosen_mask, cells_choose, tot_spks_chosen_cells, n_cells

  # decide the number of subunits to fit
  n_su = FLAGS.ratio_SU*n_cells

  # saving details
  #short_filename = 'data_model=ASM_pop_bg'
  #  short_filename = ('data_model=ASM_pop_batch_sz='+ str(FLAGS.batchsz) + '_n_b_in_c' + str(FLAGS.n_b_in_c) +
  #    '_step_sz'+ str(FLAGS.step_sz)+'_bg')

    # saving details
  batchsz = np.array([1000, 1000, 1000], dtype='int')
  n_b_in_c = np.array([1, 1, 1], dtype='int')
  step_sz = np.array([1, 1, 1], dtype='float32')
  folder_names = ['experiment25', 'experiment22', 'experiment23']
  roc_data = [[]] * n_cells
  for icnt, FLAGS.model_id in enumerate(['hinge', 'poisson', 'logistic']):
    # restore file
    FLAGS.batchsz = batchsz[icnt]
    FLAGS.n_b_in_c = n_b_in_c[icnt]
    FLAGS.step_sz = step_sz[icnt]
    folder_name = folder_names[icnt]
    if FLAGS.model_id == 'poisson':
      short_filename = ('data_model=ASM_pop_batch_sz='+ str(FLAGS.batchsz) + '_n_b_in_c' + str(FLAGS.n_b_in_c) +
      '_step_sz'+ str(FLAGS.step_sz)+'_bg')

    if FLAGS.model_id == 'poisson_full':
      short_filename = ('data_model=' + str(FLAGS.model_id) + '_batch_sz='+ str(FLAGS.batchsz) + '_n_b_in_c' + str(FLAGS.n_b_in_c) +
      '_step_sz'+ str(FLAGS.step_sz)+'_bg')

    if FLAGS.model_id == 'logistic' or FLAGS.model_id == 'hinge':
      short_filename = ('data_model='+ str(FLAGS.model_id) +'_batch_sz='+ str(FLAGS.batchsz) + '_n_b_in_c' + str(FLAGS.n_b_in_c) +
      '_step_sz'+ str(FLAGS.step_sz)+'_bg')


    print(FLAGS.model_id)
    parent_folder = FLAGS.save_location + folder_name + '/'
    save_location = parent_folder +short_filename + '/'
    restore_file = get_latest_file(save_location, short_filename)


    tf.reset_default_graph()
    with tf.Session() as sess:
      # Learn population model!
      stim = tf.placeholder(tf.float32, shape=[None, stim_dim], name='stim')
      resp = tf.placeholder(tf.float32, name='resp')
      data_len = tf.placeholder(tf.float32, name='data_len')


      # define models
      if FLAGS.model_id == 'poisson' or FLAGS.model_id == 'poisson_full':
        w = tf.Variable(np.array(0.01 * np.random.randn(stim_dim, n_su), dtype='float32'))
        a = tf.Variable(np.array(0.01 * np.random.rand(n_cells, 1, n_su), dtype='float32'))
        z = tf.transpose(tf.reduce_sum(tf.exp(tf.matmul(stim, w) + a), 2))
  
      if FLAGS.model_id == 'logistic' or FLAGS.model_id == 'hinge':
        w = tf.Variable(np.array(0.01 * np.random.randn(stim_dim, n_su), dtype='float32'))
        a = tf.Variable(np.array(0.01 * np.random.rand(n_su, n_cells), dtype='float32'))
        b_init =  np.random.randn(n_cells) #np.log((np.sum(response,0))/(response.shape[0]-np.sum(response,0)))
        b = tf.Variable(b_init,dtype='float32')
        z = tf.matmul(tf.nn.relu(tf.matmul(stim, w)), a) + b

      # restore variables
      # load tensorflow variables
      print(tf.all_variables())
      saver_var = tf.train.Saver(tf.all_variables())
      saver_var.restore(sess, restore_file)
      fd_test = {stim: stim_test,
                resp: resp_test,
                data_len: test_length}

      z_eval = sess.run(z, feed_dict=fd_test)
      print(z_eval[0:20, :])
      print(resp_test[0:20, :])

      for roc_cell in np.arange(n_cells):
        roc = get_roc(z_eval[:, roc_cell], resp_test[:, roc_cell])
        roc_data[roc_cell] = roc_data[roc_cell] + [roc]
        print(roc_cell)

  plt.figure()
  for icell in range(n_cells):
    plt.subplot(1, n_cells, icell+1)
    for icnt in np.arange(3):
      plt.plot(roc_data[icell][icnt][0], roc_data[icell][icnt][1])
      plt.hold(True)
      plt.xlabel('recall')
      plt.ylabel('precision')
    plt.legend(['hinge', 'poisson' ,'logistic'])
    cells_ch = cells[cells_choose]
    plt.title(cells_ch[icell])
  plt.show()
  plt.draw()


def get_roc(fr, resp):
  print(np.shape(fr), np.shape(resp))
  r_curve = np.array([])
  p_curve = np.array([])
  for iprctile in np.arange(0,100, 2):
    thr = np.percentile(fr, iprctile)
    recall = np.sum(np.bitwise_and(fr > thr, resp > 0).astype('double')) / np.sum((resp>0).astype('double'))
    precision = np.sum(np.bitwise_and(fr > thr, resp > 0).astype('double')) / np.sum((fr > thr).astype('double'))

    r_curve = np.append(r_curve, recall)
    p_curve = np.append(p_curve, precision)

  return [r_curve, p_curve]

if __name__ == '__main__':
  app.run()

