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
""" Decode stimulus for a given stimulus.

s_r_embedding_multiple_retina_refractor --logtostderr --stim_layers='1, 5, 1, 3, 128, 1, 3, 128, 1, 3, 128, 1, 3, 128, 2, 3, 128, 2, 3, 1, 1' --resp_layers='3, 128, 1, 3, 128, 1, 3, 128, 1, 3, 128, 2, 3, 128, 2, 3, 1, 1' --batch_norm=True --save_folder='/cns/in-d/home/bhaishahster/end_to_end_refrac_2' --save_suffix='_stim-resp_wn_nsem' --is_test=1 --taskid=24
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os.path
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import gfile
import numpy as np, h5py,numpy
import scipy.io as sio
from numpy.random import RandomState
import pickle

FLAGS = tf.app.flags.FLAGS

def predict_stimulus(resp_batch, sr_graph, resp): # responses[iretina]
  '''Predict responses'''

  resp_iters  = []

  n_cells = resp['responses'].shape[1]
  t_len = resp_batch.shape[0]

  stim_np = np.zeros((t_len, 80, 40, 30)).astype(np.float32)
  step_sz = 0.001
  eps = 1e-1
  dist_prev = np.inf
  d_log = []
  # theta_np_log = []

  grad_stim = tf.gradients(sr_graph.d_s_r_pos, sr_graph.stim_tf)[0]

  for iiter in range(200):
    feed_dict ={sr_graph.stim_tf: stim_np.astype(np.float32),
                sr_graph.anchor_model.responses_tf: resp_batch,
                sr_graph.anchor_model.map_cell_grid_tf: resp['map_cell_grid'],
                sr_graph.anchor_model.cell_types_tf: resp['ctype_1hot'],
                sr_graph.anchor_model.mean_fr_tf: resp['mean_firing_rate']}

    dist_np, grad_stim_np = sr_graph.sess.run([sr_graph.d_s_r_pos, grad_stim], feed_dict=feed_dict)

    # update theta
    stim_np = stim_np - step_sz * grad_stim_np

    # theta_np_log += [theta_np]
    if np.sum(np.abs(dist_prev - dist_np)) < eps:
      break
    print(iiter, np.sum(dist_np), np.sum(np.abs(dist_prev - dist_np)))
    dist_prev = dist_np
    d_log += [np.sum(dist_np)]

  return stim_np


def plot_decoding(stimulus_target, stim_decode, n_targets=10):

  batch_size = stimulus_target.shape[0]

  plt.figure()
  plt_tm = np.random.randint(0, batch_size, n_targets)
  n_cols = 5
  for itarget in range(n_targets):
    plt.subplot(n_targets, n_cols, itarget * n_cols + 1)
    plt.imshow(stimulus_target[plt_tm[itarget], :, :, 4].T, cmap='gray', interpolation='nearest')
    if itarget == 0:
      plt.title('Target, t-4')
    plt.axis('off')

    plt.subplot(n_targets, n_cols, itarget * n_cols + 2)
    s_target = stimulus_target[plt_tm[itarget], :, :, 4].T
    s_target_blur =scipy.ndimage.filters.gaussian_filter(s_target, 2)
    plt.imshow(s_target_blur, cmap='gray', interpolation='nearest')
    if itarget == 0:
      plt.title('Smoothened target, t-4')
    plt.axis('off')

    plt.subplot(n_targets, n_cols, itarget * n_cols + 3)
    plt.imshow(stim_decode[plt_tm[itarget], :, :, 4].T, cmap='gray', interpolation='nearest')
    if itarget == 0:
      plt.title('decoded t-4')
    plt.axis('off')

    plt.subplot(n_targets, n_cols, itarget * n_cols + 4)
    plt.imshow(stim_decode[plt_tm[itarget], :, :, 7].T, cmap='gray', interpolation='nearest')
    if itarget == 0:
      plt.title('decoded t-7')
    plt.axis('off')

    plt.subplot(n_targets, n_cols, itarget * n_cols + 5)
    plt.imshow(stim_decode[plt_tm[itarget], :, :, 10].T, cmap='gray', interpolation='nearest')
    if itarget == 0:
      plt.title('decoded t-10')
    plt.axis('off')
  plt.show()
