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
"""Response prediction."""

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
from absl import app
from absl import gfile
import numpy as np, h5py,numpy
import scipy.io as sio
from numpy.random import RandomState
import pickle


FLAGS = tf.app.flags.FLAGS


def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)


def predict_responses(stim_batch, sr_graph, resp): # responses[iretina]
  '''Predict responses'''

  resp_iters  = []

  n_cells = resp['responses'].shape[1]
  t_len = stim_batch.shape[0]
  # theta is log odds log(p/(1-p))
  theta_np = np.random.randn(t_len, n_cells)
  step_sz = 1
  eps = 1e-1
  dist_prev = np.inf
  d_log = []
  # theta_np_log = []

  # gumble softmax for sampling relaxed responses
  temp_tf = tf.placeholder(tf.float32, shape=())
  theta = tf.placeholder(tf.float32, shape=(t_len, n_cells))

  t1 = (theta + sample_gumbel(tf.shape(theta)))
  t2 = (sample_gumbel(tf.shape(theta)))
  resp_sample = tf.nn.softmax(tf.stack([t1, t2])/temp_tf, dim=0)
  resp_batch = tf.expand_dims(tf.gather(resp_sample, 0), 2)

  with tf.control_dependencies([resp_batch]):
    grad_theta = tf.gradients(resp_batch, theta)[0]

  grad_resp = tf.gradients(sr_graph.d_s_r_pos, sr_graph.anchor_model.responses_tf)[0]
  grad_resp_remove_last = tf.gather(tf.transpose(grad_resp, [2, 0, 1]), 0)

  temperature = 10

  for iiter in range(200):
    print(iiter)
    if iiter % 30 == 0:
      temperature *= 0.9
    # sample response, dr/dtheta
    resp_batch_np, grad_theta_np = sr_graph.sess.run([resp_batch, grad_theta],
                                            feed_dict ={theta : theta_np,
                                                        temp_tf : temperature})
    # d(distance)/dr
    feed_dict = {sr_graph.stim_tf: stim_batch,
                 sr_graph.anchor_model.map_cell_grid_tf: resp['map_cell_grid'],
                 sr_graph.anchor_model.cell_types_tf: resp['ctype_1hot'],
                 sr_graph.anchor_model.mean_fr_tf: resp['mean_firing_rate'],
                 sr_graph.anchor_model.responses_tf: resp_batch_np,
                 }
    dist_np, grad_resp_np = sr_graph.sess.run([sr_graph.d_s_r_pos, grad_resp_remove_last], feed_dict=feed_dict)

    # update theta
    theta_np = theta_np - step_sz * grad_resp_np * grad_theta_np

    # get scale firing rates!
    print('scaling mean firing rate')
    prob_np = np.exp(theta_np)
    prob_np = prob_np / (1 + prob_np)
    for _ in range(100):
      prob_np = prob_np * (resp['mean_firing_rate'] / prob_np.mean(0))

      # project using L2 distance
      # mean firing rate
      # prob_np = prob_np - prob_np.mean(0) + responses[iretina]['mean_firing_rate']
      # variance
      # prob_np = prob_np * np.sqrt((responses[iretina]['mean_firing_rate']**2 + responses[iretina]['mean_firing_rate']) / (prob_np**2).mean(0))

      prob_np = np.minimum(prob_np, 1 - 1e-1)
      prob_np = np.maximum(prob_np, 1e-2)

    theta_np = np.log(prob_np / (1 - prob_np))

    # theta_np_log += [theta_np]
    if np.sum(np.abs(dist_prev - dist_np)) < eps:
      break
    print(iiter, np.sum(dist_np), np.sum(np.abs(dist_prev - dist_np)))
    dist_prev = dist_np
    d_log += [np.sum(dist_np)]

  r_s = []
  for  sample_iters in range(100):
    r_ss = sr_graph.sess.run(resp_batch, feed_dict ={theta : theta_np,
                                           temp_tf : 1e-6})
    r_s += [r_ss]
  r_s = np.double(np.array(r_s) > 0.5).squeeze()

  return theta_np, r_s
