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
flags.DEFINE_float('lam_w', 0.0001, 'sparsitiy regularization of w')
flags.DEFINE_float('lam_a', 0.0001, 'sparsitiy regularization of a')
flags.DEFINE_integer('ratio_SU', 7, 'ratio of subunits/cells')
flags.DEFINE_float('su_grid_spacing', 3, 'grid spacing')
flags.DEFINE_integer('np_randseed', 23, 'numpy RNG seed')
flags.DEFINE_integer('randseed', 65, 'python RNG seed')
flags.DEFINE_float('eta_w', 1e-3, 'learning rate for optimization functions')
flags.DEFINE_float('eta_a', 1e-2, 'learning rate for optimization functions')
flags.DEFINE_float('bias_init_scale', -1, 'bias initialized at scale*std')
flags.DEFINE_string('model_id', 'relu_window', 'which model to learn?');

flags.DEFINE_string('save_location',
                    '/home/bhaishahster/',
                    'where to store logs and outputs?');
flags.DEFINE_string('data_location',
                    '/home/bhaishahster/data_breakdown/',
                    'where to take data from?')
flags.DEFINE_integer('batchsz', 100, 'batch size for training')
flags.DEFINE_integer('n_chunks', 216, 'number of data chunks') # should be 216
flags.DEFINE_integer('n_b_in_c', 10, 'number of batches in one chunk of data')

flags.DEFINE_integer('window', 3, 'size of window for each subunit in relu_window model')
flags.DEFINE_integer('stride', 3, 'stride for relu_window')
flags.DEFINE_string('folder_name', 'experiment4', 'folder where to store all the data')



def main(argv):
  #plt.ion() # interactive plotting
  window = FLAGS.window
  n_pix = (2* window + 1) ** 2
  dimx = np.floor(1 + ((40 - (2 * window + 1))/FLAGS.stride)).astype('int')
  dimy = np.floor(1 + ((80 - (2 * window + 1))/FLAGS.stride)).astype('int')
  nCells = 107
  # load model
  # load filename
  print(FLAGS.model_id)
  with tf.Session() as sess:
    if FLAGS.model_id == 'relu':
      # lam_c(X) = sum_s(a_cs relu(k_s.x)) , a_cs>0
      short_filename = ('data_model=' + str(FLAGS.model_id) +
                        '_lam_w=' + str(FLAGS.lam_w) +
                        '_lam_a='+str(FLAGS.lam_a) + '_ratioSU=' + str(FLAGS.ratio_SU) +
                        '_grid_spacing=' + str(FLAGS.su_grid_spacing) + '_normalized_bg')
      w = tf.Variable(np.array(np.random.randn(3200,749), dtype='float32'))
      a = tf.Variable(np.array(np.random.randn(749,107), dtype='float32'))


    if FLAGS.model_id == 'relu_window':
      short_filename = ('data_model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) + '_lam_w=' + str(FLAGS.lam_w) + '_bg')
      w = tf.Variable(np.array(0.1+ 0.05*np.random.rand(dimx, dimy, n_pix),dtype='float32')) # exp 5
      a = tf.Variable(np.array(np.random.rand(dimx*dimy, nCells),dtype='float32'))

    if FLAGS.model_id == 'relu_window_mother':
      short_filename = ('data_model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) + '_lam_w=' + str(FLAGS.lam_w) + '_bg')

      w_del = tf.Variable(np.array(0.1+ 0.05*np.random.rand(dimx, dimy, n_pix),dtype='float32'))
      w_mother = tf.Variable(np.array(np.ones((2 * window + 1, 2 * window + 1, 1, 1)),dtype='float32'))
      a = tf.Variable(np.array(np.random.rand(dimx*dimy, nCells),dtype='float32'))

    if FLAGS.model_id == 'relu_window_mother_sfm':
      short_filename = ('data_model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) + '_lam_w=' + str(FLAGS.lam_w) + '_bg')
      w_del = tf.Variable(np.array(0.1+ 0.05*np.random.rand(dimx, dimy, n_pix),dtype='float32'))
      w_mother = tf.Variable(np.array(np.ones((2 * window + 1, 2 * window + 1, 1, 1)),dtype='float32'))
      a = tf.Variable(np.array(np.random.rand(dimx*dimy, nCells),dtype='float32'))

    if FLAGS.model_id == 'relu_window_mother_sfm_exp':
      short_filename = ('data_model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) + '_lam_w=' + str(FLAGS.lam_w) + '_bg')
      w_del = tf.Variable(np.array(0.1+ 0.05*np.random.rand(dimx, dimy, n_pix),dtype='float32'))
      w_mother = tf.Variable(np.array(np.ones((2 * window + 1, 2 * window + 1, 1, 1)),dtype='float32'))
      a = tf.Variable(np.array(np.random.rand(dimx*dimy, nCells),dtype='float32'))

    if FLAGS.model_id == 'relu_window_exp':
      short_filename = ('data_model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) + '_lam_w=' + str(FLAGS.lam_w) + '_bg')
      w = tf.Variable(np.array(0.01+ 0.005*np.random.rand(dimx, dimy, n_pix),dtype='float32'))
      a = tf.Variable(np.array(0.02+np.random.rand(dimx*dimy, nCells),dtype='float32'))

    if FLAGS.model_id == 'relu_window_mother_exp':
      short_filename = ('data_model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) + '_lam_w=' + str(FLAGS.lam_w) + '_bg')
      w_del = tf.Variable(np.array(0.1+ 0.05*np.random.rand(dimx, dimy, n_pix),dtype='float32'))
      w_mother = tf.Variable(np.array(np.ones((2 * window + 1, 2 * window + 1, 1, 1)),dtype='float32'))
      a = tf.Variable(np.array(np.random.rand(dimx*dimy, nCells),dtype='float32'))


    if FLAGS.model_id == 'relu_window_a_support':
      short_filename = ('data_model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) + '_lam_w=' + str(FLAGS.lam_w) + '_bg')
      w = tf.Variable(np.array(0.001+ 0.0005*np.random.rand(dimx, dimy, n_pix),dtype='float32'))
      a = tf.Variable(np.array(0.002*np.random.rand(dimx*dimy, nCells),dtype='float32'))

    if FLAGS.model_id == 'exp_window_a_support':
      short_filename = ('data_model=' + str(FLAGS.model_id) + '_window=' +
                        str(FLAGS.window) + '_stride=' + str(FLAGS.stride) + '_lam_w=' + str(FLAGS.lam_w) + '_bg')
      w = tf.Variable(np.array(0.001+ 0.0005*np.random.rand(dimx, dimy, n_pix),dtype='float32'))
      a = tf.Variable(np.array(0.002*np.random.rand(dimx*dimy, nCells),dtype='float32'))

    parent_folder = FLAGS.save_location + FLAGS.folder_name + '/'
    FLAGS.save_location = parent_folder +short_filename + '/'

    # get relevant files
    file_list = gfile.ListDirectory(FLAGS.save_location)
    save_filename = FLAGS.save_location + short_filename
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

    # load tensorflow variables
    saver_var = tf.train.Saver(tf.all_variables())

    restore_file = save_filename + '-' + str(int(iter_plot))
    saver_var.restore(sess, restore_file)

    # plot subunit - cell connections
    plt.figure()
    plt.cla()
    plt.imshow(a.eval(), cmap='gray', interpolation='nearest')
    print(np.shape(a.eval()))
    plt.title('Iteration: ' + str(int(iter_plot)))
    plt.show()
    plt.draw()

    # plot all subunits on 40x80 grid
    try:
      wts = w.eval()
      for isu in range(100):
        fig = plt.subplot(10, 10, isu+1)
        plt.imshow(np.reshape(wts[:, isu],[40, 80]), interpolation='nearest', cmap='gray')
      plt.title('Iteration: ' + str(int(iter_plot)))
      fig.axes.get_xaxis().set_visible(False)
      fig.axes.get_yaxis().set_visible(False)
    except:
      print('w full does not exist? ')

    # plot a few subunits - wmother + wdel
    try:
      wts = w.eval()
      print('wts shape:', np.shape(wts))
      icnt=1
      for idimx in np.arange(dimx):
        for idimy in np.arange(dimy):
          fig = plt.subplot(dimx, dimy, icnt)
          plt.imshow(np.reshape(np.squeeze(wts[idimx, idimy, :]), (2*window+1,2*window+1)), interpolation='nearest', cmap='gray')
          icnt = icnt+1
          fig.axes.get_xaxis().set_visible(False)
          fig.axes.get_yaxis().set_visible(False)
      plt.show()
      plt.draw()
    except:
      print('w does not exist?')

    # plot wmother
    try:
      w_mot = np.squeeze(w_mother.eval())
      print(w_mot)
      plt.imshow(w_mot, interpolation='nearest', cmap='gray')
      plt.title('Mother subunit')
      plt.show()
      plt.draw()
    except:
      print('w mother does not exist')

    # plot wmother + wdel
    try:
      w_mot = np.squeeze(w_mother.eval())
      w_del = np.squeeze(w_del.eval())
      wts = np.array(np.random.randn(dimx, dimy, (2*window +1)**2))
      for idimx in np.arange(dimx):
        print(idimx)
        for idimy in np.arange(dimy):
          wts[idimx, idimy, :] = np.ndarray.flatten(w_mot) + w_del[idimx, idimy, :]
    except:
      print('w mother + w delta do not exist? ')
    '''
    try:
      
      icnt=1
      for idimx in np.arange(dimx):
        for idimy in np.arange(dimy):
          fig = plt.subplot(dimx, dimy, icnt)
          plt.imshow(np.reshape(np.squeeze(wts[idimx, idimy, :]), (2*window+1,2*window+1)), interpolation='nearest', cmap='gray')
          fig.axes.get_xaxis().set_visible(False)
          fig.axes.get_yaxis().set_visible(False)
    except:
      print('w mother + w delta plotting error? ')
    
    # plot wdel
    try:
      w_del = np.squeeze(w_del.eval())
      icnt=1
      for idimx in np.arange(dimx):
        for idimy in np.arange(dimy):
          fig = plt.subplot(dimx, dimy, icnt)
          plt.imshow( np.reshape(w_del[idimx, idimy, :], (2*window+1,2*window+1)), interpolation='nearest', cmap='gray')
          icnt = icnt+1
          fig.axes.get_xaxis().set_visible(False)
          fig.axes.get_yaxis().set_visible(False)
    except:
      print('w delta do not exist? ')
    plt.suptitle('Iteration: ' + str(int(iter_plot)))
    plt.show()
    plt.draw()
    '''
    # select a cell, and show its subunits.
    #try:

    ## Load data summary, get mask
    filename = FLAGS.data_location + 'data_details.mat'
    summary_file = gfile.Open(filename, 'r')
    data_summary = sio.loadmat(summary_file)
    total_mask = np.squeeze(data_summary['totalMaskAccept_log']).T
    stas = data_summary['stas']
    print(np.shape(total_mask))

    # a is 2D

    a_eval = a.eval()
    print(np.shape(a_eval))
    # get softmax numpy
    if FLAGS.model_id == 'relu_window_mother_sfm' or FLAGS.model_id == 'relu_window_mother_sfm_exp':
      b = np.exp(a_eval) / np.sum(np.exp(a_eval),0)
    else:
      b = a_eval

    plt.figure();
    plt.imshow(b, interpolation='nearest', cmap='gray')
    plt.show()
    plt.draw()

    # plot subunits for multiple cells.
    n_cells = 10
    n_plots_max = 20
    plt.figure()
    for icell_cnt, icell in enumerate(np.arange(n_cells)):
      mask2D = np.reshape(total_mask[icell,: ], [40, 80])
      nz_idx = np.nonzero(mask2D)
      np.shape(nz_idx)
      print(nz_idx)
      ylim = np.array([np.min(nz_idx[0])-1, np.max(nz_idx[0])+1])
      xlim = np.array([np.min(nz_idx[1])-1, np.max(nz_idx[1])+1])

      icnt = -1
      a_thr = np.percentile(np.abs(b[:, icell]), 99.5)
      n_plots = np.sum(np.abs(b[:, icell]) > a_thr)
      nx = np.ceil(np.sqrt(n_plots)).astype('int')
      ny = np.ceil(np.sqrt(n_plots)).astype('int')
      ifig=0
      ww_sum = np.zeros((40,80))

      for idimx in np.arange(dimx):
        for idimy in np.arange(dimy):
          icnt = icnt + 1
          if(np.abs(b[icnt,icell]) > a_thr):
            ifig = ifig + 1
            fig = plt.subplot(n_cells, n_plots_max, icell_cnt*n_plots_max + ifig + 2)
            ww = np.zeros((40,80))
            ww[idimx*FLAGS.stride: idimx*FLAGS.stride + (2*window+1),
               idimy*FLAGS.stride: idimy*FLAGS.stride + (2*window+1)] = b[icnt, icell] * (np.reshape(wts[idimx, idimy, :],
                                                                                                          (2*window+1,2*window+1)))
            plt.imshow(ww, interpolation='nearest', cmap='gray')
            plt.ylim(ylim)
            plt.xlim(xlim)
            plt.title(b[icnt,icell])
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

            ww_sum = ww_sum + ww

      fig = plt.subplot(n_cells, n_plots_max, icell_cnt*n_plots_max +  2)
      plt.imshow(ww_sum, interpolation='nearest', cmap='gray')
      plt.ylim(ylim)
      plt.xlim(xlim)
      fig.axes.get_xaxis().set_visible(False)
      fig.axes.get_yaxis().set_visible(False)
      plt.title('STA from model')

      fig = plt.subplot(n_cells, n_plots_max, icell_cnt*n_plots_max +  1)
      plt.imshow(np.reshape(stas[:, icell], [40, 80]), interpolation='nearest', cmap='gray')
      plt.ylim(ylim)
      plt.xlim(xlim)
      fig.axes.get_xaxis().set_visible(False)
      fig.axes.get_yaxis().set_visible(False)
      plt.title('True STA')

    plt.show()
    plt.draw()

    #except:
    #  print('a not 2D?')

    # using xlim and ylim, and plot the 'windows' which are relevant with their weights
    sq_flat = np.zeros((dimx, dimy))
    icnt = 0
    for idimx in np.arange(dimx):
      for idimy in np.arange(dimy):
        sq_flat[idimx, idimy] = icnt
        icnt = icnt + 1

    n_cells = 1
    n_plots_max = 10
    plt.figure()
    for icell_cnt, icell in enumerate(np.array([1, 2, 3, 4, 5])):#enumerate(np.arange(n_cells)):
      a_thr = np.percentile(np.abs(b[:, icell]), 99.5)
      mask2D = np.reshape(total_mask[icell,: ], [40, 80])
      nz_idx = np.nonzero(mask2D)
      np.shape(nz_idx)
      print(nz_idx)
      ylim = np.array([np.min(nz_idx[0])-1, np.max(nz_idx[0])+1])
      xlim = np.array([np.min(nz_idx[1])-1, np.max(nz_idx[1])+1])
      print(xlim, ylim)

      win_startx = np.ceil((xlim[0] - (2*window+1)) / FLAGS.stride)
      win_endx = np.floor((xlim[1]-1) / FLAGS.stride )
      win_starty = np.ceil((ylim[0] - (2*window+1)) / FLAGS.stride)
      win_endy = np.floor((ylim[1]-1) / FLAGS.stride )
      dimx_plot = win_endx - win_startx + 1
      dimy_plot =  win_endy - win_starty + 1
      ww_sum = np.zeros((40,80))
      for irow, idimy in enumerate(np.arange(win_startx, win_endx+1)):
        for icol, idimx in enumerate(np.arange(win_starty, win_endy+1)):
          fig = plt.subplot(dimx_plot+1, dimy_plot, (irow + 1) * dimy_plot + icol+1 )
          ww = np.zeros((40,80))
          ww[idimx*FLAGS.stride: idimx*FLAGS.stride + (2*window+1),
             idimy*FLAGS.stride: idimy*FLAGS.stride + (2*window+1)] =  (np.reshape(wts[idimx, idimy, :],
                                                                        (2*window+1,2*window+1)))
          plt.imshow(ww, interpolation='nearest', cmap='gray')
          plt.ylim(ylim)
          plt.xlim(xlim)
          if b[sq_flat[idimx, idimy],icell] > a_thr:
            plt.title(b[sq_flat[idimx, idimy],icell], fontsize=10, color='g')
          else:
            plt.title(b[sq_flat[idimx, idimy],icell], fontsize=10, color='r')
          fig.axes.get_xaxis().set_visible(False)
          fig.axes.get_yaxis().set_visible(False)

          ww_sum = ww_sum + ww * b[sq_flat[idimx, idimy],icell]

      fig = plt.subplot(dimx_plot+1, dimy_plot, 2)
      plt.imshow(ww_sum, interpolation='nearest', cmap='gray')
      plt.ylim(ylim)
      plt.xlim(xlim)
      fig.axes.get_xaxis().set_visible(False)
      fig.axes.get_yaxis().set_visible(False)
      plt.title('STA from model')

      fig = plt.subplot(dimx_plot+1, dimy_plot, 1)
      plt.imshow(np.reshape(stas[:, icell], [40, 80]), interpolation='nearest', cmap='gray')
      plt.ylim(ylim)
      plt.xlim(xlim)
      fig.axes.get_xaxis().set_visible(False)
      fig.axes.get_yaxis().set_visible(False)
      plt.title('True STA')



      plt.show()
      plt.draw()





if __name__ == '__main__':
  app.run()

