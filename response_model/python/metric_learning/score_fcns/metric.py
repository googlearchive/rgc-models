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
"""Learn metric from response triplets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import tensorflow as tf

gfile = tf.gfile


class Metric(object):
  """Learn a metric using triplets.

  Given exmaples of response triplets {(anchor, positive, negative)},
  learn a metric such that d(anchor, positive) < d(anchor, negative).
  """

  def __init__(self):
    """Set up model to learn the metric."""
    pass

  def get_parameters(self):
    """Return parameters of the metric."""
    pass

  def update(self, triplet_batch):
    """Given a batch of training data, update metric parameters.

    Args:
        triplet_batch : List [anchor, positive, negative], each with shape:
                        (batch x cells x time_window)
    """
    pass

  def get_distance(self, resp1, resp2):
    """Give distances between pairs in two sets of responses.

    The two response arrays are (batch x cells x time_window) return distances
    between corresponding pairs of responses in resp1, resp2.

    Args:
        resp1 : Binned responses (each : batch x cells x time_window).
        resp2 : same as resp 1.

    Returns:
        distances : evaluated distances of size (batch).
    """
    pass

  def initialize_model(self, save_folder, file_name, sess):
    """Setup model variables and saving information.

    Args:
      save_folder (string) : Folder to store model.
                             Makes one if it does not exist.
      filename (string) : Prefix of model/checkpoint files.
      sess : Tensorflow session.
    """

    # Make folder.
    self.initialize_folder(save_folder, file_name)

    # Initialize variables.
    self.initialize_variables(sess)

  def initialize_folder(self, save_folder, file_name):
    """Intialize saving location of the model."""

    # Make folder if it does not exist.
    if not gfile.IsDirectory(save_folder):
      gfile.MkDir(save_folder)

    self.save_folder = save_folder
    self.short_filename = file_name
    self.long_filename = os.path.join(save_folder, file_name)

  def initialize_variables(self, sess):
    """Initialize variables or restore from previous fits."""

    sess.run(tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer()))
    saver_var = tf.train.Saver(tf.global_variables(),
                               keep_checkpoint_every_n_hours=5)
    load_prev = False
    start_iter = 0
    try:
      # Restore previous fits if they are available
      # - useful when programs are preempted frequently on .
      latest_filename = self.short_filename + '_latest_fn'
      restore_file = tf.train.latest_checkpoint(self.save_folder,
                                                latest_filename)
      # Restore previous iteration count and start from there.
      start_iter = int(restore_file.split('/')[-1].split('-')[-1])
      saver_var.restore(sess, restore_file)  # restore variables
      load_prev = True
    except:
      tf.logging.info('No previous dataset')

    if load_prev:
      tf.logging.info('Previous results loaded from: ' + restore_file)
    else:
      tf.logging.info('Variables initialized')

    tf.logging.info('Loaded iteration: %d' % start_iter)

    self.saver_var = saver_var
    self.iter = start_iter

  def save_model(self):
    """Save model variables."""
    latest_filename = self.short_filename + '_latest_fn'
    self.saver_var.save(self.sess, self.long_filename, global_step=self.iter,
                        latest_filename=latest_filename)

  def get_embedding(self, resp):
    """Get embedding for responses.

    Args :
      resp : responses (Batch size x n_cells x time_window)
    """
    pass
