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
r""""Learn a response metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import tensorflow as tf
from tensorflow.python.profiler import PrintModelAnalysis
from absl import app
import retina.response_model.python.metric_learning.config as config
import retina.response_model.python.metric_learning.data_util as du

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=()):

  # get details to store model
  model_savepath, model_filename = config.get_filepaths()

  # load responses to two trials of long white noise
  data_train = du.DataUtilsMetric(os.path.join(FLAGS.data_path,
                                               FLAGS.data_train))
  data_test = du.DataUtilsMetric(os.path.join(FLAGS.data_path,
                                              FLAGS.data_test))

  with tf.Session() as sess:

    # Initialize the model.
    tf.logging.info('Model : %s ' % FLAGS.model)
    score = config.get_model(sess, model_savepath, model_filename,
                             data_train, True)

    # print model analysis
    PrintModelAnalysis(tf.get_default_graph())

    # setup summary writer
    summary_writers = []
    for loc in ['train', 'test']:
      summary_location = os.path.join(model_savepath, model_filename,
                                      'summary_' + loc)
      summary_writer = tf.summary.FileWriter(summary_location, sess.graph)
      summary_writers += [summary_writer]

    # setup triplet functions
    triplet_fcn_train = config.get_triplet_fcn(data_train,
                                               FLAGS.batch_size_train)
    triplet_fcn_test = config.get_triplet_fcn(data_test,
                                              FLAGS.batch_size_test)

    # Learn the metric.
    for score.iter in range(score.iter, FLAGS.max_iter):
      # Get new training batch.
      anchor_batch, pos_batch, neg_batch, _, _, _ = triplet_fcn_train()
      triplet_batch = [anchor_batch, pos_batch, neg_batch]

      # Update model
      loss_train = score.update(triplet_batch)

      if score.iter % 10 == 0:

        # write train summary
        anchor_train, pos_train, neg_train, _, _, _ = triplet_fcn_train()
        feed_dict_train = {score.anchor: anchor_train,
                           score.pos: pos_train,
                           score.neg: neg_train}
        loss_train, summary_train = sess.run([score.loss, score.summary_op],
                                             feed_dict=feed_dict_train)
        summary_writers[0].add_summary(summary_train, score.iter)

        # write test summary
        anchor_test, pos_test, neg_test, _, _, _ = triplet_fcn_test()
        feed_dict_test = {score.anchor: anchor_test,
                          score.pos: pos_test,
                          score.neg: neg_test}
        loss_test, summary_test = sess.run([score.loss, score.summary_op],
                                           feed_dict=feed_dict_test)
        summary_writers[1].add_summary(summary_test, score.iter)

        tf.logging.info('Iteration: %d, Loss : %.3f, Loss test : %.3f' %
                        (score.iter, loss_train, loss_test))

        tf.logging.info('Summary written')

      if score.iter % 100 == 0:
        score.save_model()
        tf.logging.info('Model saved')

if __name__ == '__main__':
  app.run()
