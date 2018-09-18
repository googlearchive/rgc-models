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
r""""Learn a simple metric learning with hard negatives."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import tensorflow as tf
from absl import app
import retina.response_model.python.metric_learning.config as config
import retina.response_model.python.metric_learning.data_util as du
import retina.response_model.python.metric_learning.score_fcns.quadratic_score as quad
import retina.response_model.python.metric_learning.score_fcns.mrf as mrf
import retina.response_model.python.metric_learning.analyse_metric as analyse

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
    # get test data

    # triplet A
    outputs = data_wn.get_triplets(batch_size=FLAGS.batch_size_train,
                                   time_window=FLAGS.time_window)
    anchor_test, pos_test, neg_test, _, _ = outputs
    triplet_test_a = [anchor_test, pos_test, neg_test]
    tf.logging.info('triplet A tested')

    # triplet B
    outputs = data_wn.get_tripletsB(batch_size=FLAGS.batch_size_train,
                                    time_window=FLAGS.time_window)
    anchor_test, pos_test, neg_test, _, _, _ = outputs
    triplet_test_b = [anchor_test, pos_test, neg_test]
    tf.logging.info('triplet B tested')

    # triplet C
    outputs = data_wn.get_tripletsC(batch_size=FLAGS.batch_size_train,
                                    time_window=FLAGS.time_window)
    anchor_test, pos_test, neg_test, _, _, _ = outputs
    triplet_test_c = [anchor_test, pos_test, neg_test]
    tf.logging.info('triplet C tested')

    # triplet D
    outputs = data_wn.get_tripletsD(batch_size=FLAGS.batch_size_train,
                                    time_window=FLAGS.time_window)
    anchor_test, pos_test, neg_test, _, _, _ = outputs
    triplet_test_d = [anchor_test, pos_test, neg_test]
    tf.logging.info('triplet D tested')

    loss_test_log = []
    for score.iter in range(score.iter, FLAGS.max_iter):
      # Get new training batch.

      outputs = data_wn.get_triplets_mix(batch_size=FLAGS.batch_size_train,
                                         time_window=FLAGS.time_window,
                                         score=score)
      anchor_batch, pos_batch, neg_batch, _, _, _ = outputs
      triplet_batch = [anchor_batch, pos_batch, neg_batch]
      # from IPython import embed; embed()

      loss_train = score.update(triplet_batch)

      if score.iter % 10 == 0:
        # Run tests regularly.

        # Test A
        loss_test_a = sess.run(score.loss, {score.anchor: triplet_test_a[0],
                                            score.pos: triplet_test_a[1],
                                            score.neg: triplet_test_a[2]})
        # Test B
        loss_test_b = sess.run(score.loss, {score.anchor: triplet_test_b[0],
                                            score.pos: triplet_test_b[1],
                                            score.neg: triplet_test_b[2]})

        # Test C
        loss_test_c = sess.run(score.loss, {score.anchor: triplet_test_c[0],
                                            score.pos: triplet_test_c[1],
                                            score.neg: triplet_test_c[2]})
        # Test A
        loss_test_d = sess.run(score.loss, {score.anchor: triplet_test_d[0],
                                            score.pos: triplet_test_d[1],
                                            score.neg: triplet_test_d[2]})

        loss_test = [loss_test_a, loss_test_b, loss_test_c, loss_test_d]
        loss_test_log += [loss_test]  # This log is unused right now.

      if score.iter % 100 == 0:
        tf.logging.info('Iteration: %d, Losses : %.3f %.3f %.3f %.3f' %
                        (score.iter, loss_test[0], loss_test[1],
                         loss_test[2], loss_test[3]))

      if score.iter % 1000 == 0:
        score.save_model()
        tf.logging.info('Model saved')

if __name__ == '__main__':
  app.run()
