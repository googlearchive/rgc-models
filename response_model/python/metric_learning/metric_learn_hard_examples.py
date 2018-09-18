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
r""""Learn a simple metric by mining hard examples.

Args:
--save_suffix='_hard_examples' --lam_l1=0.001 --data_train='example_long_wn_2rep_ON_OFF.mat' --triplet_type='a' --model='quadratic'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from google3.pyglib import app
import retina.response_model.python.metric_learning.config as config
import retina.response_model.python.metric_learning.data_util as du
import retina.response_model.python.metric_learning.score_fcns.quadratic_score as quad
import retina.response_model.python.metric_learning.score_fcns.mrf as mrf

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=()):

  # get details to store model
  model_savepath, model_filename = config.get_filepaths()
  print('lam l1 is: '+ str(FLAGS.lam_l1))

  # load responses to two trials of long white noise
  data_wn = du.DataUtilsMetric(os.path.join(FLAGS.data_path, FLAGS.data_train))

  # quadratic score function
  with tf.Session() as sess:

    # Initialize the model.
    tf.logging.info('Model : %s ' % FLAGS.model)

    if FLAGS.model == 'quadratic':
      score = quad.QuadraticScore(sess, model_savepath,
                                  model_filename,
                                  n_cells=data_wn.n_cells,
                                  time_window=FLAGS.time_window,
                                  lr=FLAGS.learning_rate,
                                  lam_l1=FLAGS.lam_l1)

    if FLAGS.model == 'mrf':
      score = mrf.MRFScore(sess, model_savepath, model_filename,
                           n_cells=data_wn.n_cells,
                           time_window=FLAGS.time_window,
                           lr=FLAGS.learning_rate,
                           lam_l1=FLAGS.lam_l1,
                           cell_centers=data_wn.get_centers(),
                           neighbor_threshold=FLAGS.neighbor_threshold)

    # Learn the metric.
    # Set triplet type.
    if FLAGS.triplet_type == 'a':
      triplet_fcn = data_wn.get_triplets

    if FLAGS.triplet_type == 'b':
      triplet_fcn = data_wn.get_tripletsB

    # Get test data.
    outputs = triplet_fcn(batch_size=FLAGS.batch_size_train,
                          time_window=FLAGS.time_window)
    anchor_test = outputs[0]
    pos_test = outputs[1]
    neg_test = outputs[2]
    triplet_test = [anchor_test, pos_test, neg_test]

    # Learn the metric.
    loss_test_log = []
    loss_train_log = []
    hard_iters = 200  # when to start showing hard examples.

    # plt.ion()

    for score.iter in range(score.iter, FLAGS.max_iter):

      # Get new training batch.
      if score.iter > hard_iters:
        batch_scale = 3
      else:
        batch_scale = 1
      outputs = triplet_fcn(batch_size=FLAGS.batch_size_train*batch_scale,
                            time_window=FLAGS.time_window)
      anchor_batch = outputs[0]
      pos_batch = outputs[1]
      neg_batch = outputs[2]

      if score.iter > hard_iters:
        dd_neg = score.get_distance(anchor_batch, neg_batch)
        dd_pos = score.get_distance(anchor_batch, pos_batch)
        dd_diff = dd_pos - dd_neg
        top_examples = np.argsort(dd_diff)[::-1]
        choose_examples = top_examples[:FLAGS.batch_size_train]

        anchor_batch = anchor_batch[choose_examples, :, :]
        pos_batch = pos_batch[choose_examples, :, :]
        neg_batch = neg_batch[choose_examples, :, :]
        # from IPython import embed; embed()

      triplet_batch = [anchor_batch, pos_batch, neg_batch]
      loss_train = score.update(triplet_batch)

      if score.iter % 10 == 0:
        # Run tests regularly.
        loss_test = sess.run(score.loss, {score.anchor:
                                          triplet_test[0],
                                          score.pos: triplet_test[1],
                                          score.neg: triplet_test[2]})
        loss_test_log += [loss_test]  # This log is unused right now.
        loss_train_log += [loss_train]

      if score.iter % 10 == 0:
        tf.logging.info('Iteration: %d, Loss : %.3f, Loss test : %.3f' %
                        (score.iter, loss_train, loss_test))
        '''
        plt.clf()
        plt.subplot(1, 3, 1)
        plt.plot(loss_train_log, 'k')
        plt.title('Train')

        plt.subplot(1, 3, 2)
        plt.plot(loss_test_log, 'k')
        plt.title('Test')

        plt.subplot(1, 3, 3)
        plt.imshow(sess.run(score.A_symm), interpolation='nearest', cmap='gray')
        plt.title('A')
        plt.show()
        plt.draw()
        plt.pause(0.1)
        '''

      if score.iter % 1000 == 0:
        score.save_model()
        tf.logging.info('Model saved')

if __name__ == '__main__':
  app.run()
