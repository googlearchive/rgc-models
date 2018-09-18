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
'''Analysis file.'''
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
flags.DEFINE_integer('n_b_in_c', 10, 'number of batches in one chunk of data')
flags.DEFINE_integer('np_randseed', 23, 'numpy RNG seed')
flags.DEFINE_integer('randseed', 65, 'python RNG seed')
flags.DEFINE_integer('ratio_SU', 2, 'ratio of subunits/cells')
flags.DEFINE_string('model_id', 'poisson', 'which model to fit')
FLAGS = flags.FLAGS

def main(argv):
  print('\nCode started')

  np.random.seed(FLAGS.np_randseed)
  random.seed(FLAGS.randseed)

  ## Load data summary

  filename = FLAGS.data_location + 'data_details.mat'
  summary_file = gfile.Open(filename, 'r')
  data_summary = sio.loadmat(summary_file)
  cells = np.squeeze(data_summary['cells'])
  if FLAGS.model_id == 'poisson' or FLAGS.model_id == 'logistic' or FLAGS.model_id == 'hinge':
    cells_choose = (cells ==3287) | (cells ==3318 ) | (cells ==3155) | (cells ==3066)
  if FLAGS.model_id == 'poisson_full':
    cells_choose = np.array(np.ones(np.shape(cells)), dtype='bool')
  n_cells = np.sum(cells_choose)

  tot_spks = np.squeeze(data_summary['tot_spks'])
  total_mask = np.squeeze(data_summary['totalMaskAccept_log']).T
  tot_spks_chosen_cells = tot_spks[cells_choose]
  chosen_mask = np.array(np.sum(total_mask[cells_choose,:],0)>0, dtype='bool')
  print(np.shape(chosen_mask))
  print(np.sum(chosen_mask))

  stim_dim = np.sum(chosen_mask)


  print('\ndataset summary loaded')
  # use stim_dim, chosen_mask, cells_choose, tot_spks_chosen_cells, n_cells

  # decide the number of subunits to fit
  n_su = FLAGS.ratio_SU*n_cells

  #batchsz = [100, 500, 1000,  100,   500,  1000,    100, 500, 1000, 1000, 1000, 5000, 10000, 5000, 10000]
  #n_b_in_c = [10,      2,      1,     10,   2,     1,  10, 2, 1,      1,   1,     1,    1,    1,     1 ]
  #step_sz = [0.0001, 0.0001, 0.0001, 0.01, 0.01, 0.01 , 1, 1, 1,      10,  100,   10,   10,   1,     1 ]
  batchsz = [100,     500,   1000,   5000,  1000,  100,    500,  1000, 5000, 10000, 100,  500, 1000, 5000, 10000, 100,  500, 1000, 5000, 10000]
  n_b_in_c = [10,      2,      1,     1,     1,     10,     2,     1,    1,   1,     10,   2,    1,    1,     1,  10,   2,    1,    1,     1 ]
  step_sz = [0.1,     0.1,    0.1, 0.1,   0.1,      1 ,      1,    1,    1,   1,     5,    5,   5,     5,     5,  10,   10,   10,   10,    10 ]


  with tf.Session() as sess:
    # Learn population model!
    stim = tf.placeholder(tf.float32, shape=[None, stim_dim], name='stim')
    resp = tf.placeholder(tf.float32, name='resp')
    data_len = tf.placeholder(tf.float32, name='data_len')

    # get filename
    if FLAGS.model_id == 'poisson' or FLAGS.model_id == 'poisson_full':
      w = tf.Variable(np.array(0.01 * np.random.randn(stim_dim, n_su), dtype='float32'))
      a = tf.Variable(np.array(0.1 * np.random.rand(n_cells, 1, n_su), dtype='float32'))
    if FLAGS.model_id == 'logistic' or FLAGS.model_id == 'hinge':
      w = tf.Variable(np.array(0.01 * np.random.randn(stim_dim, n_su), dtype='float32'))
      a = tf.Variable(np.array(0.01 * np.random.rand(n_su, n_cells), dtype='float32'))
      b_init =  np.random.randn(n_cells) #np.log((np.sum(response,0))/(response.shape[0]-np.sum(response,0)))
      b = tf.Variable(b_init,dtype='float32')

    plt.figure()
    for icnt, ibatchsz in enumerate(batchsz):
      in_b_in_c = n_b_in_c[icnt]
      istep_sz = np.array(step_sz[icnt],dtype='double')
      print(icnt)
      if FLAGS.model_id == 'poisson':
        short_filename = ('data_model=ASM_pop_batch_sz='+ str(ibatchsz) + '_n_b_in_c' + str(in_b_in_c) +
        '_step_sz'+ str(istep_sz)+'_bg')
      else:
        short_filename = ('data_model='+ str(FLAGS.model_id) +'_batch_sz='+ str(ibatchsz) + '_n_b_in_c' + str(in_b_in_c) +
        '_step_sz'+ str(istep_sz)+'_bg')

      parent_folder = FLAGS.save_location + FLAGS.folder_name + '/'
      save_location = parent_folder +short_filename + '/'
      print(gfile.IsDirectory(save_location))
      print(save_location)
      save_filename = save_location + short_filename


     #determine filelist
      file_list = gfile.ListDirectory(save_location)
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
      #print(bin_files)
      print(len(meta_files), len(bin_files), len(file_list))

  # get latest iteration
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

      # load tensorflow variables
      saver_var = tf.train.Saver(tf.all_variables())

      restore_file = save_filename + '-' + str(int(iter_plot))
      saver_var.restore(sess, restore_file)


      a_eval = a.eval()
      print(np.exp(np.squeeze(a_eval)))
      #print(np.shape(a_eval))

      # get 2D region to plot
      mask2D = np.reshape(chosen_mask, [40, 80])
      nz_idx = np.nonzero(mask2D)
      np.shape(nz_idx)
      print(nz_idx)
      ylim = np.array([np.min(nz_idx[0])-1, np.max(nz_idx[0])+1])
      xlim = np.array([np.min(nz_idx[1])-1, np.max(nz_idx[1])+1])
      w_eval = w.eval()

      #plt.figure()
      n_su = w_eval.shape[1]
      for isu in np.arange(n_su):
        xx = np.zeros((3200))
        xx[chosen_mask] = w_eval[:, isu]
        fig = plt.subplot(20, n_su, n_su * icnt + isu+1)

        plt.imshow(np.reshape(xx, [40, 80]), interpolation='nearest', cmap='gray')
        plt.ylim(ylim)
        plt.xlim(xlim)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        #if FLAGS.model_id == 'logistic' or FLAGS.model_id == 'hinge':
        #  plt.title(str(a_eval[isu, :]))
        #else:
        #  plt.title(str(np.squeeze(np.exp(a_eval[:, 0, isu]))), fontsize=12)
        if isu == 4:
          plt.title('Iteration:' + str(int(iter_plot)) + ' batchSz:' + str(ibatchsz) + ' step size:' + str(istep_sz), fontsize=18)
  plt.show()
  plt.draw()
    
    
if __name__ == '__main__':
  app.run()

