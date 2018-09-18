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
"""Utils for training models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf


def train_model(model, sess, summary_frequency=100, feed_dict=None):
  """Train a model using by  sess.run(model.update) in a loop.

  Stores summary by at a certain 'summary_frequency'
  It also manages data queues

  Args:
    model : The tensorflow graph
    sess : Tensorflow session to run graph in
    summary_frequency : Write summary and store graph at this frequency
  """

  # Setup data threads.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  tf.logging.info('threads started')

  # Train the model.
  try:
    while not coord.should_stop():
      start_time = time.time()

      # Run one step of the model.
      # _, loss_train_np = sess.run([model.update, model.probes.loss])
      sess.run([model.update], feed_dict=feed_dict)
      duration = time.time() - start_time

      # Print an overview fairly often.
      #tf.logging.info('training @ %d took %.3fs loss: %.3f' % (model.iter,
      #                                                         duration,
      #                                                         loss_train_np))

      tf.logging.info('training @ %d took %.3fs cell: %d' % (model.iter,
                                                             duration,
                                                             sess.run(model.probes.select_cells)))

      # In distributed setting model.iter counts the
      # number of iterations by a single worker.
      model.iter += 1

      if model.iter % summary_frequency == 0:
        # Compute training and testing losses.
        model.write_summaries(sess)

  except tf.errors.OutOfRangeError:
    tf.logging.info('Done training for %d steps.' % (model.iter))
  finally:
    # When done, ask the threads to stop.
    coord.request_stop()

  # Wait for threads to finish.
  coord.join(threads)
  sess.close()
  tf.logging.info('all threads stopped')
