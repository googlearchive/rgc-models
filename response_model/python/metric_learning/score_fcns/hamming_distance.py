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
r""""Learn a score function of form s(x,y) = ||x-y||_{2}^{2}.

For binary response vectors, squared euclidean distance is
same as hamming distance.

A way to run the model:
--logtostderr --model='hamming' \
--data_test='data012_lr_15_cells_groupb_with_stimulus_test.mat' \
--data_train='data012_lr_15_cells_groupb_with_stimulus_train.mat' \
--data_path='/cns/oi-d/home/bhaishahster/metric_learning/examples_pc2017_04_25_1/' \
--save_suffix='_2017_04_25_1_cells_15_groupb_train' --gfs_user='foam-brain-gpu'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from retina.response_model.python.metric_learning.score_fcns import metric

gfile = tf.gfile


class HammingScore(metric.Metric):
  """Setup a metric that computes hamming distances."""

  def __init__(self, save_folder):
    """Set up the saving directory."""
    # Make folder if it does not exist.
    if not gfile.IsDirectory(save_folder):
      gfile.MkDir(save_folder)

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

    return np.linalg.norm(resp1 - resp2, ord='fro', axis=(1, 2))


