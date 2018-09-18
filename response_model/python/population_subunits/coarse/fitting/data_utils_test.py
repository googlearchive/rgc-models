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
"""Tests for retina.response_model.python.population_subunits.coarse.fitting.data_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path
import tensorflow as tf
from absl import app
from absl import flags
from absl import gfile

import cPickle as pickle
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import scipy.io as sio
from scipy import ndimage
import random
import time
from retina.response_model.python.population_subunits.coarse.fitting import data_utils

FLAGS = flags.FLAGS
# flags for data location
flags.DEFINE_string('data_location',
                    '/home/bhaishahster/data_breakdown/',
                    'where to take data from?')
flags.DEFINE_integer('batchsz', 1000, 'batch size for training')


def main(argv):

  # load choose few cells this time
  #data = data_utils.CoarseDataUtils(FLAGS.data_location, FLAGS.batchsz,
  #                                   all_cells=False, masked_stimulus=True,
  #                                   chosen_cells = [3287, 3318, 3155, 3066])


  # setup dataset

  data = data_utils.CoarseDataUtils(FLAGS.data_location, FLAGS.batchsz)
  """
  # get testing data
  stim_test, resp_test = data.get_test_data()
  print('Got testing data')

  # get training data
  for idata in range(500):
    stim_train, resp_train = data.get_next_training_batch()
  print('Got training data')
  """
  # convert data to TF records
  # data.convert_to_TFRecords('coarse_data',
  #                          save_location='/home/bhaishahster/tmp')

  # convert data to TF chunks, and stores data in small chunks

  # from IPython.terminal.embed import InteractiveShellEmbed
  # ipshell = InteractiveShellEmbed()
  # ipshell()
  data.convert_to_TFRecords_chunks('coarse_data_chunks',
                            save_location='/home/bhaishahster/data_breakdown')

  print('converting to TF records done')

  # get queue

  num_epochs=100

  stim, resp = data_utils.inputs(name='coarse_data_chunks',
                      data_location='/home/bhaishahster/data_breakdown',
                      batch_size=FLAGS.batchsz, num_epochs=num_epochs,
                      stim_dim=3200, resp_dim=107)
  """
  stim, resp = data_utils.inputs(name='coarse_data',
                      data_location='/home/bhaishahster/data_breakdown',
                      batch_size=FLAGS.batchsz, num_epochs=num_epochs,
                      stim_dim=3200, resp_dim=107)
  """
  print('stim and response queue runners made')

  # The op for initializing the variables.
  init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())
  print('init op made')

  sess=tf.Session()
  sess.run(init_op)
  print('initialized')

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  print('threads started')

  # now, dequeue batches using sess.run() to verify that it works ok
  try:
    step = 0
    while not coord.should_stop():
      start_time = time.time()

      # get a batch of stimulus and response
      stim_np, resp_np = sess.run([stim,resp])
      duration = time.time() - start_time

      # Print an overview fairly often.
      print(step, duration)
      step += 1
  except tf.errors.OutOfRangeError:
    print('Done training for %d epochs, %d steps.' % (num_epochs, step))
  finally:
    # When done, ask the threads to stop.
    coord.request_stop()


  # Wait for threads to finish.
  coord.join(threads)
  sess.close()
  print('all threads stopped')

if __name__ == '__main__':
  app.run()
