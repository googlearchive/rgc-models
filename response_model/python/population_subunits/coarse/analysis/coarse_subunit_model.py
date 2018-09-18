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
r"Coarse subunit model.

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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import ndimage
import random

from retina.response_model.python.population_subunits.coarse.fitting import data_utils
from retina.response_model.python.population_subunits.coarse.analysis import model_utils
import retina.response_model.python.population_subunits.coarse.analysis.analysis as analysis

# flags for data locations
flags.DEFINE_string('folder_name', 'experiment4', 'folder where to store all the data')

flags.DEFINE_string('save_location',
                    '/home/bhaishahster/',
                    'where to store logs and outputs?');

flags.DEFINE_string('data_location',
                    '/home/bhaishahster/data_breakdown/',
                    'where to take data from?')

# flags for stochastic learning
flags.DEFINE_integer('batchsz', 100, 'batch size for training')
flags.DEFINE_integer('max_iter', 400000, 'maximum number of iterations')
flags.DEFINE_float('step_sz', 0.1, 'step size for learning algorithm')
flags.DEFINE_float('learn', True, 'wheather to learn a model, or analyse a fitted one')

# random number generators initialized
# removes unneccessary data variabilities while comparing algorithms
flags.DEFINE_integer('np_randseed', 23, 'numpy RNG seed')
flags.DEFINE_integer('randseed', 65, 'python RNG seed')


# flags for model/loss specification
flags.DEFINE_string('model_id', 'almost_convolutional', 'which model to fit')
flags.DEFINE_string('loss_string', 'poisson', 'which loss to use?')
flags.DEFINE_string('masked_stimulus', False,
                    'use all pixels or only those inside RF of selected cells?')
flags.DEFINE_string('all_cells', True,
                    'learn model for all cells or a few chosen ones?')


# model specific terms
# useful for convolution-like models
flags.DEFINE_integer('window', 2,
                     'size of window for each subunit in relu_window model')
flags.DEFINE_integer('stride', 1, 'stride for relu_window')
# some models need regularization of parameters
flags.DEFINE_float('lam_w', 0.0, 'sparsitiy regularization of w')

FLAGS = flags.FLAGS

def get_filename():
  print('Save folder name: ' + str(FLAGS.folder_name) +
        '\nmodel: ' + str(FLAGS.model_id) +
        '\nmasked stimulus: ' + str(FLAGS.masked_stimulus) +
        '\nall_cells? ' + str(FLAGS.all_cells) +
        '\nbatch size ' + str(FLAGS.batchsz) +
        '\nstep size ' + str(FLAGS.step_sz))

  # saving details
  short_filename = ('_masked_stim=' + str(FLAGS.masked_stimulus) + '_all_cells='+
                    str(FLAGS.all_cells) + '_loss='+ str(FLAGS.loss_string) +
                    '_batch_sz='+ str(FLAGS.batchsz) +
                    '_step_sz'+ str(FLAGS.step_sz))

  return short_filename


def main(unused_argv):

  # set random seeds
  np.random.seed(FLAGS.np_randseed)
  random.seed(FLAGS.randseed)

  # set up datasets

  data = data_utils.CoarseDataUtils(FLAGS.data_location, np.int(FLAGS.batchsz),
                               FLAGS.all_cells, FLAGS.masked_stimulus, test_length=500)


  # get filename
  short_filename = get_filename()

  # setup model
  with tf.Session() as sess:
    stim_dim = 3200
    stim = tf.placeholder(tf.float32, shape=[None, stim_dim], name='stim')
    resp = tf.placeholder(tf.float32, name='resp')

    n_cells = np.sum(data.cells_choose)

   # setup model graph
    model = model_utils.setup_response_model(FLAGS.model_id, FLAGS.loss_string,
                                             sess, stim, resp, short_filename,
                                             FLAGS.window, FLAGS.stride,
                                             FLAGS.lam_w, FLAGS.step_sz,
                                             n_cells)



    # initialize model variables
    model.initialize_model(FLAGS.save_location, FLAGS.folder_name, sess)

    if FLAGS.learn :
      # do learning
      learn_model(data, model, sess)
    else:
      # do analysis
      analyse_model(data, model, sess)



def learn_model(data, model, sess):

  # Finally, do fitting
  # get test data and make test dictionary
  stim_test,resp_test = data.get_test_data()
  fd_test = {model.stim: stim_test,
             model.resp: resp_test}

  plt.ion()
  plt.figure()
  for istep in np.arange(model.iter,FLAGS.max_iter):
    model.iter = istep
    print(istep)
    # get training data and make test dictionary
    import time
    start_time = time.time()
    stim_train, resp_train = data.get_next_training_batch()
    fd_train = {model.stim: stim_train,
                model.resp: resp_train}
    print('Getting data @ %d took %.3f s' % (istep, time.time()-start_time))

    # take training step
    start_time = time.time()
    loss_train_np = model.training_fcn(fd_train)
    print('Gradient step @ %d took %0.3f s, loss %0.3f' % (istep,
                                                           time.time()-start_time,
                                                           loss_train_np))

    if istep%10 == 0:
      # compute training and testing losses
      model.write_summaries(sess, fd_train, fd_test)

      # plot w_mother
      plt.imshow(np.squeeze(sess.run(model.model_params.w_mother)),
                 interpolation='nearest', cmap='gray')
      plt.title('w mother, iteration: %d' % istep)

      plt.show()
      plt.draw()


def analyse_model(data, model, sess):
  w_mot = sess.run(model.model_params.w_mother)
  plt.imshow(np.squeeze(w_mot), interpolation='nearest', cmap='gray')
  plt.show()
  plt.draw()


  # plot delta subunit for 'almost convolutional - model + delta models'
  w_del = model.model_params.w_del

  w_del_e = np.squeeze(w_del.eval())
  dimx = w_del_e.shape[0]
  dimy = w_del_e.shape[1]
  print(w_del_e.shape)
  wts = np.array(0 * np.random.randn(dimx, dimy, (2*FLAGS.window +1)**2))

  for idimx in np.arange(dimx):
    print(idimx)
    for idimy in np.arange(dimy):
      wts[idimx, idimy, :] = (np.ndarray.flatten(w_mot) +
                                  w_del_e[idimx, idimy, :])

  ipshell = InteractiveShellEmbed()
  ipshell()

  a = model.model_params.a
  a_sfm = tf.transpose(tf.nn.softmax(tf.transpose(a)))
  a_sfm_eval = sess.run(a_sfm)

  stas = data.get_stas()

  analysis.plot_su_gaussian_spokes(a_sfm_eval, wts, dimx, dimy, stas)

if __name__ == '__main__':
  app.run()
