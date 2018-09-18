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
r"""Learn models for predicting population responses.

The WN stimulus is prefiltered in time using STA time course.
So, only spatial structure of the subunit is learnt

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import random

import numpy as np
import tensorflow as tf

from absl import app
from absl import flags
from absl import gfile

from retina.response_model.python.population_subunits.coarse.analysis import analysis_util
from retina.response_model.python.population_subunits.coarse.analysis import model_util
from retina.response_model.python.population_subunits.coarse.fitting import data_utils
from retina.response_model.python.population_subunits.coarse.fitting import train_util

# Flags for data locations.
flags.DEFINE_string('folder_name', 'experiment_tfrec',
                    'folder where to store all the data')

flags.DEFINE_string('save_location',
                    '/home/bhaishahster/coarse_experiments/',
                    'where to store logs and outputs?')

flags.DEFINE_string('data_location',
                    '/home/bhaishahster/data_breakdown/',
                    'where to take data from?')

flags.DEFINE_string('tmp_dir',
                    '/home/bhaishahster/data_breakdown/',
                    'temporary folder on machine for better I/O')

lags.DEFINE_integer('taskid', 0, 'Task ID')


# Flags for stochastic learning.
flags.DEFINE_integer('batchsz', 10000, 'batch size for training')
flags.DEFINE_integer('num_epochs', 40000, 'maximum number of iterations')
flags.DEFINE_float('step_sz', 10, 'step size for learning algorithm')
flags.DEFINE_float('learn', True,
                   'whether to learn a model, or analyse a fitted one')

# Random number generators initialized.
# Removes unneccessary data variabilities while comparing algorithms.
flags.DEFINE_integer('np_randseed', 23, 'numpy RNG seed')
flags.DEFINE_integer('randseed', 65, 'python RNG seed')

# Flags for model/loss specification.
flags.DEFINE_string('model_id', 'almost_convolutional',
                    'model for firing rate: almost_convolutional')
flags.DEFINE_string('loss_string', 'poisson',
                    'which loss to use: poisson or conditional_poisson')
flags.DEFINE_string('masked_stimulus', False,
                    'use all pixels or only those inside RF of selected cells?')
flags.DEFINE_string('chosen_cells', None,
                    'learn model for which cells? if None, learn of all cells')
flags.DEFINE_integer('n_cells', 107, 'number of cells in the dataset')

# Model specific terms.
# Useful for convolution-like models.
flags.DEFINE_integer('window', 2,
                     'size of window for each subunit in relu_window model')
flags.DEFINE_integer('stride', 1, 'stride for relu_window')
# Some models need regularization of parameters.
flags.DEFINE_float('lam_w', 0.0, 'sparsitiy regularization of w')


FLAGS = flags.FLAGS


def get_filename():
  """"Generate partial filename using FLAGS for running the code."""

  if FLAGS.chosen_cells is None:
    all_cells = True
  else:
    all_cells = False

  tf.logging.info('Save folder name: ' + str(FLAGS.folder_name) +
                  '\nmodel: ' + str(FLAGS.model_id) +
                  '\nmasked stimulus: ' + str(FLAGS.masked_stimulus) +
                  '\nall_cells? ' + str(all_cells) +
                  '\nbatch size ' + str(FLAGS.batchsz) +
                  '\nstep size ' + str(FLAGS.step_sz) +
                  '\ntaskid:' + str(FLAGS.taskid))

  # saving details
  short_filename = ('_masked_stim=' + str(FLAGS.masked_stimulus) +
                    '_all_cells='+ str(all_cells) +
                    '_loss='+ str(FLAGS.loss_string) +
                    '_batch_sz='+ str(FLAGS.batchsz) +
                    '_step_sz='+ str(FLAGS.step_sz) +
                    '_taskid=' + str(FLAGS.taskid))

  return short_filename


def main(unused_argv):

  # Set random seeds.
  np.random.seed(FLAGS.np_randseed)
  random.seed(FLAGS.randseed)

  # Set up dataset as a local tfrecords file.

  # load .tfrecords file
  """
  src = os.path.join(FLAGS.data_location, 'coarse_data.tfrecords')
  if not gfile.Exists(src) or not FLAGS.learn:
    tf.logging.info('Loading data')
    tf.logging.info('Source ' + src)
    data = data_utils.CoarseDataUtils(FLAGS.data_location,
                                      np.int(FLAGS.batchsz),
                                      masked_stimulus=FLAGS.masked_stimulus,
                                      chosen_cells=FLAGS.chosen_cells,
                                      test_length=500)

  if not gfile.Exists(src):
    data.convert_to_TFRecords('coarse_data.tfrecords', FLAGS.data_location)
    tf.logging.info('TF records created')

  tf.logging.info('Will copy .tfrecords to: ' + str(FLAGS.tmp_dir))

  # Copy .tfrecords file to local disk to avoid data I/O latency.
  dst = os.path.join(FLAGS.tmp_dir, 'coarse_data.tfrecords')
  if not gfile.IsDirectory(FLAGS.tmp_dir):
    gfile.MkDir(FLAGS.tmp_dir)
  if not gfile.Exists(dst):
    tf.logging.info('Copying TF records locally')
    gfile.Copy(src, dst)
  tf.logging.info('TFRecords copied to local folder')


  

  # load tfrecords chunks
  src = os.path.join(FLAGS.data_location, 'coarse_data_chunks')
  if not gfile.Exists(src) or not FLAGS.learn:
    tf.logging.info('Loading data')
    tf.logging.info('Source ' + src)
    data = data_utils.CoarseDataUtils(FLAGS.data_location,
                                      np.int(FLAGS.batchsz),
                                      masked_stimulus=FLAGS.masked_stimulus,
                                      chosen_cells=FLAGS.chosen_cells,
                                      test_length=500)

  if not gfile.Exists(src):
    data.convert_to_TFRecords_chunks('coarse_data_chunks', FLAGS.data_location)
    tf.logging.info('TF records created')

  tf.logging.info('Will copy .tfrecords chunks to: ' + str(FLAGS.tmp_dir))

  # Copy .tfrecords file to local disk to avoid data I/O latency over cns.
  dst = os.path.join(FLAGS.tmp_dir, 'coarse_data_chunks')
  if not gfile.IsDirectory(FLAGS.tmp_dir):
    gfile.MkDir(FLAGS.tmp_dir)
  if not gfile.Exists(dst):
    tf.logging.info('Copying TF records locally')
    gfile.Copy(src, dst)
  tf.logging.info('TFRecords copied to local folder')
  """

  # Get filename.
  short_filename = get_filename()

  # Setup graph.
  with tf.Session() as sess:

    # TODO(bhaishahster) In future, get this from a lightweight data header.

    # Setup input nodes.
    # uses tfrecords chunks
    if not FLAGS.model_id == 'almost_convolutional_mel_dropout_only_wdelta':
      # uses .tfrecords
      stim, resp = data_utils.inputs(name='coarse_data.tfrecords',
                                   data_location=FLAGS.tmp_dir,
                                   batch_size=FLAGS.batchsz,
                                   num_epochs=FLAGS.num_epochs,
                                   stim_dim=3200, resp_dim=FLAGS.n_cells)
      '''
      stim, resp = data_utils.inputs(name='coarse_data_chunks',
                                   data_location=FLAGS.tmp_dir,
                                   batch_size=FLAGS.batchsz,
                                   num_epochs=FLAGS.num_epochs,
                                   stim_dim=3200, resp_dim=FLAGS.n_cells)
      '''
      tf.logging.info('stim and resp queue runners made')


    # load data in memory
    fdict=None
    if FLAGS.model_id == 'almost_convolutional_mel_dropout_only_wdelta':
      data = data_utils.CoarseDataUtils(FLAGS.data_location,
                                      np.int(FLAGS.batchsz),
                                      masked_stimulus=FLAGS.masked_stimulus,
                                      chosen_cells=FLAGS.chosen_cells,
                                      test_length=500)

      stim = tf.placeholder(tf.float32, name='stim')
      resp = tf.placeholder(tf.float32, name='resp')
      fdict = {stim: data.stimulus.astype(np.float32), resp: data.response.astype(np.float32)}
      tf.logging.info('stim and resp loaded in memory')

    # Setup model graph.
    model = model_util.setup_response_model(FLAGS.model_id, FLAGS.loss_string,
                                            stim, resp, short_filename,
                                            FLAGS.window, FLAGS.stride,
                                            FLAGS.lam_w, FLAGS.step_sz,
                                            FLAGS.n_cells, FLAGS.taskid)

    # Initialize model variables.
    # model.initialize_model(FLAGS.save_location, FLAGS.folder_name, sess)
    model.initialize_model(FLAGS.save_location, FLAGS.folder_name, sess, feed_dict=fdict)
    if FLAGS.learn:
      # Do learning.
      # TODO(bhaishahster): not setup for distributed training yet
      train_util.train_model(model, sess, summary_frequency=107, feed_dict=fdict)
    else:
      # Do analysis.
      # raise NotImplementedError()
      # TODO(bhaishahster) implement analyse_model(model, sess)
      if (FLAGS.model_id == 'almost_convolutional_experimental' or
          FLAGS.model_id == 'almost_convolutional' or
          FLAGS.model_id == 'almost_convolutional_experimental_wdel_only'):
        analysis_util.analyse_almost_convolutional_model(model, sess, data)

      if (FLAGS.model_id == 'almost_convolutional_softmax' or
          FLAGS.model_id == 'almost_convolutional_exp_dropout_scaling' or
          FLAGS.model_id == 'almost_convolutional_exponential_dropout' or
          FLAGS.model_id == 'almost_convolutional_exp_dropout_only_wdelta' or
          FLAGS.model_id == 'almost_convolutional_exp_one_cell_only_wdelta' or
          FLAGS.model_id == 'almost_convolutional_mel_dropout_only_wdelta'):
        analysis_util.analyse_almost_convolutional_softmax(model, sess, data)

if __name__ == '__main__':
  app.run()
