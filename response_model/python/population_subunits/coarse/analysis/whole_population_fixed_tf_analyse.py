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
"""Analyse the results of subunit fitting.
"""

import sys
import tensorflow as tf
from absl import app
from absl import flags
from absl import gfile

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pylab
import matplotlib.pyplot as plt

import numpy as np, h5py
import scipy.io as sio
from scipy import ndimage
import random
import re # regular expression matching

FLAGS = flags.FLAGS
flags.DEFINE_float('lam_w', 0.0, 'sparsitiy regularization of w')
flags.DEFINE_float('lam_a', 0.0, 'sparsitiy regularization of a')
flags.DEFINE_integer('ratio_SU', 7, 'ratio of subunits/cells')
flags.DEFINE_float('su_grid_spacing', 3, 'grid spacing')
flags.DEFINE_integer('np_randseed', 23, 'numpy RNG seed')
flags.DEFINE_integer('randseed', 65, 'python RNG seed')
flags.DEFINE_float('eta_w', 1e-3, 'learning rate for optimization functions')
flags.DEFINE_float('eta_a', 1e-2, 'learning rate for optimization functions')
flags.DEFINE_float('bias_init_scale', -1, 'bias initialized at scale*std')
flags.DEFINE_string('model_id', 'relu', 'which model to learn?');

flags.DEFINE_string('save_location',
                    '/home/bhaishahster/',
                    'where to store logs and outputs?');
flags.DEFINE_string('data_location',
                    '/cns/in-d/home/bhaishahster/data_breakdown/',
                    'where to take data from?')
flags.DEFINE_integer('batchsz', 100, 'batch size for training')
flags.DEFINE_integer('n_chunks', 216, 'number of data chunks') # should be 216
flags.DEFINE_integer('n_b_in_c', 10, 'number of batches in one chunk of data')

flags.DEFINE_string('folder_name', 'experiment31', 'folder where to store all the data')



def main(argv):
  #plt.ion() # interactive plotting

  # load model
  # load filename
  print(FLAGS.model_id)
  print(FLAGS.folder_name)
  if FLAGS.model_id == 'relu':
    # lam_c(X) = sum_s(a_cs relu(k_s.x)) , a_cs>0
    short_filename = ('data_model=' + str(FLAGS.model_id) +
                    '_lam_w=' + str(FLAGS.lam_w) +
                    '_lam_a='+str(FLAGS.lam_a) + '_ratioSU=' + str(FLAGS.ratio_SU) +
                    '_grid_spacing=' + str(FLAGS.su_grid_spacing) + '_normalized_bg')

  if FLAGS.model_id == 'exp':
    short_filename = ('data_model3=' + str(FLAGS.model_id) +
                    '_bias_init=' + str(FLAGS.bias_init_scale) + '_ratioSU=' + str(FLAGS.ratio_SU) +
                    '_grid_spacing=' + str(FLAGS.su_grid_spacing) + '_normalized_bg')

  if FLAGS.model_id == 'mel_re_pow2':
    short_filename = ('data_model3=' + str(FLAGS.model_id) +
                    '_lam_w=' + str(FLAGS.lam_w) +
                    '_lam_a='+str(FLAGS.lam_a) + '_ratioSU=' + str(FLAGS.ratio_SU) +
                    '_grid_spacing=' + str(FLAGS.su_grid_spacing) + '_normalized_bg')

  if FLAGS.model_id == 'relu_logistic':
    short_filename = ('data_model3=' + str(FLAGS.model_id) +
                    '_lam_w=' + str(FLAGS.lam_w) +
                    '_lam_a='+str(FLAGS.lam_a) + '_ratioSU=' + str(FLAGS.ratio_SU) +
                    '_grid_spacing=' + str(FLAGS.su_grid_spacing) + '_normalized_bg')

  if FLAGS.model_id == 'relu_proximal':
    short_filename = ('data_model3=' + str(FLAGS.model_id) +
                    '_lam_w=' + str(FLAGS.lam_w) +
                    '_lam_a='+str(FLAGS.lam_a) + '_eta_w=' + str(FLAGS.eta_w) + '_eta_a=' + str(FLAGS.eta_a) + '_ratioSU=' + str(FLAGS.ratio_SU) +
                    '_grid_spacing=' + str(FLAGS.su_grid_spacing) + '_proximal_bg')

  if FLAGS.model_id == 'relu_eg':
    short_filename = ('data_model3=' + str(FLAGS.model_id) +
                    '_lam_w=' + str(FLAGS.lam_w) +
                    '_eta_w=' + str(FLAGS.eta_w) + '_eta_a=' + str(FLAGS.eta_a) + '_ratioSU=' + str(FLAGS.ratio_SU) +
                    '_grid_spacing=' + str(FLAGS.su_grid_spacing) + '_eg_bg')


  # get relevant files
  parent_folder = FLAGS.save_location + FLAGS.folder_name + '/'
  FLAGS.save_location = parent_folder +short_filename + '/'
  file_list = gfile.ListDirectory(FLAGS.save_location)
  save_filename = FLAGS.save_location + short_filename

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

  # get iteration numbers
  iterations = np.array([])
  for file_name in bin_files:
    try:
      iterations = np.append(iterations, int(file_name.split('/')[-1].split('-')[-1]))
    except:
      print('Bad filename' + file_name)
  iterations.sort()
  print(iterations)

  iter_plot = iterations[-1]
  print(int(iter_plot))
  with tf.Session() as sess:
    # load tensorflow variables
    w = tf.Variable(np.array(np.random.randn(3200,749), dtype='float32'))
    a = tf.Variable(np.array(np.random.randn(749,107), dtype='float32'))
    saver_var = tf.train.Saver(tf.all_variables())
    restore_file = save_filename + '-' + str(int(iter_plot))
    saver_var.restore(sess, restore_file)

    # plot subunit - cell connections
    plt.figure()
    plt.cla()
    plt.imshow(a.eval(), cmap='gray', interpolation='nearest')
    plt.title('Iteration: ' + str(int(iter_plot)))
    plt.show()
    plt.draw()

    # plot a few subunits
    wts = w.eval()
    for isu in range(100):
      fig = plt.subplot(10, 10, isu+1)
      plt.imshow(np.reshape(wts[:, isu],[40, 80]), interpolation='nearest', cmap='gray')
    plt.title('Iteration: ' + str(int(iter_plot)))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
 
    plt.show()
    plt.draw()

if __name__ == '__main__':
  app.run()

