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
"""Stimulation algorithm for prosthesis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy
import pickle
import copy
import os
import cvxpy
import tensorflow as tf
from tensorflow.python.platform import gfile

tf.flags.DEFINE_integer('taskid', 0,
                    'Task ID determines what dictionary to use')

tf.flags.DEFINE_string('save_dir',
                    '/home/bhaishahster/spatial_locality/',
                    'Directory to store results.')

FLAGS = tf.flags.FLAGS


def main(unused_argv=()):

  #
  err_eps = 1e-1
  delta_m = 1

  task_list = []
  for d in [8, 10, 20, 30, 50]:
    for s in [2, 3, 4, 6, 8, 10, 14, 18, 20, 30]:
      if 2*s + 1 < d:
        task_list += [[d, s]]

  d, s = task_list[FLAGS.taskid]
  print('task id : %d, d %d, s %s' % (FLAGS.taskid, d, s))


  # set truth
  x = np.zeros((d, d))
  x[int(d/2 - 2): int(d/2 + s + 1), int(d/2 - s): int(d/2 + s + 1)] = 1
  xx = np.ndarray.flatten(x)
  l1_val = np.sum(np.abs(xx))

  neighbor_mat = get_neighbormat(np.ones((d, d)), nbd=1)
  eps = 0.1
  lnl1_val  = np.sum(np.abs(xx) / (neighbor_mat.dot(np.abs(xx)) + eps))
  rl1_val = np.sum(np.abs(xx) / (np.eye(d * d).dot(np.abs(xx)) + eps))

  l1_fn = lambda d, m, xx, l1_val : l1_norm_estimate(d, m, xx, l1_val)
  lnl1_fn = lambda d, m, xx, lnl1_val : lnl1_norm_estimate(d, m, xx, lnl1_val, eps, neighbor_mat)
  rl1_fn = lambda d, m, xx, l1_val : lnl1_norm_estimate(d, m, xx, l1_val, eps, np.eye(d * d))

  ifcn = -1
  for fcn in [l1_fn, lnl1_fn, rl1_fn]:
    ifcn += 1
    if ifcn == 0:
      l_val = l1_val
    if ifcn == 1:
      l_val = lnl1_val
    if ifcn == 2:
      l_val = rl1_val

    error_log = {}

    # L1 norm
    # Min
    m_min = np.floor(d/4).astype(np.int)
    err_min = fcn(d, m_min, xx, l_val)
    error_log.update({m_min: err_min})
    print('%d: %.4f' % (m_min, err_min))

    # Max
    m_max = np.int(10 * d)
    err_max = fcn(d, m_max, xx, l_val)
    error_log.update({m_max: err_max})
    print('%d: %.4f' % (m_max, err_max))

    while(True):
      if m_max - m_min <= delta_m:
        print('Range too close, stopping... ')
        break

      m_mid = np.int((m_min + m_max) / 2)
      err_mid = fcn(d, m_mid, xx, l_val)
      error_log.update({m_mid: err_mid})

      if err_mid < err_eps:
        m_max = m_mid

      if err_mid > err_eps:
        m_min = m_mid

      print('%d: %.4f, min: %d, max: %d' % (m_mid, err_mid, m_min, m_max))

    if ifcn == 0:
      error_l1_d_log = error_log
    if ifcn == 1:
      error_lnl1_d_log = error_log
    if ifcn == 2:
      error_rl1_d_log = error_log

  save_dict = {'error_l1_d_log': error_l1_d_log,
               'error_lnl1_d_log': error_lnl1_d_log,
               'error_rl1_d_log': error_rl1_d_log, 'd': d, 's': s}

  pickle.dump(save_dict, gfile.Open(os.path.join(FLAGS.save_dir,
                                                 'd_%d_s_%d.pkl' % (d, s)), 'w'))

def l1_norm_estimate(d, m, xx, l1_val):

  err_log = []
  for irep in range(10):
    A = np.random.randn(m, d*d)
    y = A.dot(xx)

    x_cvxpy = cvxpy.Variable(d**2)
    objective = cvxpy.Minimize(cvxpy.sum_squares(y - A * x_cvxpy))
    constraints = [cvxpy.sum_entries(cvxpy.abs(x_cvxpy)) <= l1_val]
    prob = cvxpy.Problem(objective, constraints)
    # The optimal objective is returned by prob.solve().
    result = prob.solve(verbose=False)
    x_val_prev = np.array(x_cvxpy.value)
    err_log += [np.sum(np.abs(np.squeeze(xx) - np.squeeze(x_val_prev))**2)]
  return np.mean(err_log)


def lnl1_norm_estimate(d, m, xx, lnl1_val, eps, neighbor_mat):

  # LNL1
  err_log = []
  for irep in range(10):
    A = np.random.randn(m, d*d)
    y = A.dot(xx)
    x_val_prev = np.expand_dims(np.random.rand(d ** 2), 1)

    error_prev = np.inf
    for iiter in range(500):
      x_cvxpy = cvxpy.Variable(d**2)
      wts = 1 / (neighbor_mat.dot(np.abs(x_val_prev)) + eps)
      objective = cvxpy.Minimize(cvxpy.sum_squares(y - A * x_cvxpy))
      constraints = [wts.T * cvxpy.abs(x_cvxpy) <= lnl1_val]
      prob = cvxpy.Problem(objective, constraints)
      # The optimal objective is returned by prob.solve().
      result = prob.solve(verbose=False)
      x_val_prev = np.array(x_cvxpy.value)
      error = np.sum(np.abs(np.squeeze(xx) - np.squeeze(x_val_prev)))
      if np.abs(error - error_prev) < 1e-6:
        break
      error_prev = error

    err_log += [error_prev]

  return np.mean(err_log)

def get_neighbormat(mask_matrix, nbd=1):
  mask = np.ndarray.flatten(mask_matrix)>0

  x = np.repeat(np.expand_dims(np.arange(mask_matrix.shape[0]), 1), mask_matrix.shape[1], 1)
  y = np.repeat(np.expand_dims(np.arange(mask_matrix.shape[1]), 0), mask_matrix.shape[0], 0)
  x = np.ndarray.flatten(x)
  y = np.ndarray.flatten(y)

  idx = np.arange(len(mask))
  iidx = idx[mask]
  xx = np.expand_dims(x[mask], 1)
  yy = np.expand_dims(y[mask], 1)

  distance = (xx - xx.T)**2 + (yy - yy.T)**2
  neighbor_mat = np.double(distance <= nbd)

  return neighbor_mat

if __name__ == '__main__':
  app.run(main)
