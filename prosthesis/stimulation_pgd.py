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

tf.flags.DEFINE_string('Algorithm', 'simultaneous_planning',
                       'Planning algorithm to use')
  
tf.flags.DEFINE_float('learning_rate',
                      100,
                      'Learning rate for optimization.')

tf.flags.DEFINE_integer('t_max',
                        20,
                        'Maximum number of stimulations')

tf.flags.DEFINE_integer('delta',
                        5,
                        'Maximum number of stimulations')

tf.flags.DEFINE_string('normalization',
                       'C',
                       'Normalization ')


tf.flags.DEFINE_string('save_dir',
                       '/home/bhaishahster/stimulation_algos/pgd/',
                       'Directory to store results.')



FLAGS = flags.FLAGS

def main(unused_argv=()):
  src = '/home/bhaishahster/Stimulation_data.pkl'

  data = pickle.load(gfile.Open(src, 'r'))

  S_collection = data['S']  # Target
  A = data['A']  # Decoder
  D = data['D'].T  # Dictionary

  for itarget in range(S_collection.shape[1]):
    
    S = S_collection[:, itarget]

    # Run Greedy first to initialize

    if FLAGS.Algorithm == 'greedy':
      x_greedy = greedy_stimulation(S, A, D, max_stims = FLAGS.t_max * FLAGS.delta,
                         file_suffix='%d' % itarget, save=True, save_dir=FLAGS.save_dir)

    if FLAGS.Algorithm == 'simultaneous_planning':
      x_greedy = greedy_stimulation(S, A, D, max_stims = FLAGS.t_max * FLAGS.delta,
                       file_suffix='%d' % itarget, save=False, save_dir=FLAGS.save_dir)

      # Plan for multiple time points
      x_init = np.zeros((x_greedy.shape[0], FLAGS.t_max))
      #from IPython import embed; embed()
      for it in range(FLAGS.t_max):
        print((it + 1) * FLAGS.delta - 1)
        x_init[:, it] = x_greedy[:, (it + 1) * FLAGS.delta - 1]
      simultaneous_planning(S, A, D, t_max=FLAGS.t_max, lr=FLAGS.learning_rate,
                          delta=FLAGS.delta, normalization=FLAGS.normalization,
                          file_suffix='%d' % itarget, x_init=x_init, save_dir=FLAGS.save_dir)


    if FLAGS.Algorithm == 'simultaneous_planning_cvx':
      simultaneous_planning_cvx(S, A, D, t_max=FLAGS.t_max,
                                delta=FLAGS.delta,
                                file_suffix='%d' % itarget, save_dir=FLAGS.save_dir)




def greedy_stimulation(S, A, D, save_dir='', max_stims = 100, file_suffix='', save=False):
  '''Greedily select stimulation pattern for each step.'''

  n_dict_elem = D.shape[1]

  # compute variance of dictionary elements.
  stas_norm = np.expand_dims(np.sum(A ** 2, 0) ,0)  # 1 x # cells
  var_dict = np.squeeze(np.dot(stas_norm, D * (1 - D)))  # # dict
  AD = A.dot(D)
  x = np.zeros(n_dict_elem)
  current_mean_percept = A.dot(D.dot(x))

  x_chosen = np.zeros((n_dict_elem, max_stims))
  for istim in range(max_stims):
    print(istim)
    errs = np.sum((np.expand_dims(S - current_mean_percept, 1) - AD) ** 2, 0) + var_dict
    chosen_dict = np.argmin(errs)
    min_e_d = errs[chosen_dict]
    '''
    # Compute objective value
    min_e_d = np.inf
    for idict in range(n_dict_elem):
      diff = S - current_mean_percept - AD[:, idict]
      error = np.sum(diff ** 2, 0) + var_dict[idict]
      if error < min_e_d:
        chosen_dict = idict
        min_e_d = error
    '''
    x[chosen_dict] += 1
    current_mean_percept = A.dot(D.dot(x))
    x_chosen[chosen_dict, istim] = 1

  # Final Error
  x_chosen = np.cumsum(x_chosen, 1)
  error_curve = compute_error(S, A, D, var_dict, x_chosen)
  if save:
    save_dict = {'error_curve': error_curve, 'x_chosen': x_chosen, 'x': x}
    pickle.dump(save_dict,
                gfile.Open(os.path.join(save_dir,
                                        'greedy_%d_%s.pkl' %
                                        (max_stims, file_suffix)),
                           'w'))

  return x_chosen


def compute_error(S, A, D, var_dict, x_chosen):
  diff = np.expand_dims(S, 1) - A.dot(D.dot(x_chosen))
  return np.sum(diff ** 2, 0) + np.dot(var_dict, x_chosen)


def simultaneous_planning_cvx(S, A, D, t_max = 2000, delta = 5,
                          file_suffix='', save_dir=''):

  # Setup problem parameters
  # make p_tau uniform between 500 and 2000
  p_tau = np.ones(t_max)
  p_tau[:5] = 0
  p_tau = p_tau / np.sum(p_tau)

  n_dict_elem = D.shape[1]

  # compute variance of dictionary elements.
  stas_norm = np.expand_dims(np.sum(A ** 2, 0) ,0)  # 1 x # cells
  var_dict = np.squeeze(np.dot(stas_norm, D * (1 - D)))  # # dict

  # Construct the problem.

  y = cvxpy.Variable(n_dict_elem, t_max)

  x = cvxpy.cumsum(y, 1)
  S_expanded = np.repeat(np.expand_dims(S, 1), t_max, 1)
  objective = cvxpy.Minimize((cvxpy.sum_entries((S_expanded - A * (D * x))**2, 0) + var_dict * x) * p_tau)
  constraints = [0 <= y, cvxpy.sum_entries(y, 0).T <= delta * np.ones((1, t_max)).T]
  prob = cvxpy.Problem(objective, constraints)

  # The optimal objective is returned by prob.solve().
  result = prob.solve(verbose=True)
  # The optimal value for x is stored in x.value.
  print(x.value)
  # The optimal Lagrange multiplier for a constraint
  # is stored in constraint.dual_value.
  print(constraints[0].dual_value)


def simultaneous_planning(S, A, D, save_dir='', t_max = 2000, lr=0.01,
                          normalization='T-i', delta = 5,
                          file_suffix='', x_init=None):
  ''' Solve the simultaneous planning constrained optimization problem.

  Let xi be the set of electrodes played till time i.
  Let the distribution of saccades be p(tau).

  Optimization problem.
  Min E_tau ||S - ADx_tau||^2
  subject to -
    x_{i+1} >= x_{i}   forall i
    |x_i|_1 <= i   forall i
    x_i >= 0  forall i.

  Solve using projected gradient descent.

  '''

  # Compute expanded quantities
  S_expanded = np.repeat(np.expand_dims(S, 1), t_max, 1)

  if normalization == 'T-i':
    normalizing_factors =  np.array([t_max - i  for i in range(t_max)])

  if normalization == 'sqrt(T-i)':
    normalizing_factors =  np.sqrt(np.array([t_max - i  for i in range(t_max)]))

  if normalization == 'C':
    normalizing_factors = (t_max / 2)  + 0 * np.array([t_max - i  for i in range(t_max)])

  # make p_tau uniform between 500 and 2000
  p_tau = np.ones(t_max)

   # TODO(bhaishahster): Dont hardcode p_tau!!
  p_tau[:5] = 0
  p_tau = p_tau / np.sum(p_tau)

  n_dict_elem = D.shape[1]
  # TODO(bhaishahster): Find better initialization.

  # Initialize
  if x_init is not None:
    y_init_normalized =  np.zeros_like(x_init)  # successive difference of x.
    y_init_normalized[:, 0] = x_init[:, 0]
    for iy in np.arange(1, y_init_normalized.shape[1]):
      y_init_normalized[:, iy] = x_init[:, iy] - x_init[:, iy - 1]
    y_init = y_init_normalized * np.expand_dims(normalizing_factors, 0)  #

  else:
    # Zero initialization
    y_init = np.zeros((n_dict_elem, t_max))

  # Smarter initialization
  #x_init =  np.linalg.pinv(D).dot(np.linalg.pinv(A).dot(S_expanded))
  #x_init = project_constraints(x_init)

  #
  # Do projected gradient descent
  y = y_init.copy()
  f_log = []

  # compute variance of dictionary elements.
  stas_norm = np.expand_dims(np.sum(A ** 2, 0) ,0)  # 1 x # cells
  var_dict = np.squeeze(np.dot(stas_norm, D * (1 - D)))  # # dict

  radii_normalized = (np.ones(y.shape[1]))  * delta
  radii = np.multiply(normalizing_factors, radii_normalized )
  x_log = []
  y_best = []
  f_min = np.inf
  for iiter in range(4000):

    if iiter % 500 == 499:
      lr = lr * 0.3

    # compute x from y
    y_normalized = y / np.expand_dims(normalizing_factors, 0)
    x = np.cumsum(y_normalized, 1)

    x_log += [x]
    # Compute objective value
    diff = S_expanded - A.dot(D.dot(x))
    errors = np.sum(diff ** 2, 0) + np.dot(var_dict, x)
    f = errors.dot(p_tau)

    print('Iterate: %d, Function value : %.3f' % (iiter, f))
    f_log += [f]
    if f < f_min:
      f_min = f
      y_best = y

    # Gradients step
    grad = (D.T.dot(A.T.dot((S_expanded - A.dot(D.dot(x))))) - np.expand_dims(var_dict, 1)) * np.expand_dims(p_tau, 0)
    # collect gradient for each y. - new formulation that Kunal suggested.
    grad_y = np.cumsum(grad[:, ::-1], 1)
    grad_y = grad_y[:, ::-1] / np.expand_dims(normalizing_factors, 0)
    # y = y + (lr / np.sqrt(iiter + 1)) *  grad_y
    y = y + (lr) * grad_y


    # Project to constraint set
    y = project_l1_pos(y, radii)

    '''
    if iiter > 2:
      if np.abs(f_log[-2] - f_log[-1]) < 1e-5:
        # compute x from y
        y_normalized = y / np.expand_dims(normalizing_factors, 0)
        x = np.cumsum(y_normalized, 1)
        break
    '''

  y = y_best
  y_normalized = y / np.expand_dims(normalizing_factors, 0)
  x = np.cumsum(y_normalized, 1)
  diff = S_expanded - A.dot(D.dot(x))
  errors = np.sum(diff ** 2, 0) + np.dot(var_dict, x)

  # Randomized rounding
  x_rr_discrete = randomized_rounding(x)
  errors_rr_discrete = compute_error(S, A, D, var_dict, x_rr_discrete)

  # Hard thresholding
  y_ht_discrete = hard_thresholding(y, radii)
  y_ht_discrete_normalized = y_ht_discrete / np.expand_dims(normalizing_factors, 0)
  x_ht_discrete = np.cumsum(y_ht_discrete_normalized, 1)
  errors_ht_discrete = compute_error(S, A, D, var_dict, x_ht_discrete)

  x_log = np.array(x_log)
  x_decrease = np.sum((x_log - x_log[-1, :, :]) ** 2, 1)
  x_dec_best = np.sum((x_log - x[:, :]) ** 2, 1)
  x_last = x_log[-1]

  save_dict = {'x': x, 'x_rr_discrete': x_rr_discrete,
               'x_ht_discrete': x_ht_discrete,
               'errors': errors, 'x_decrease': x_decrease,
               'x_dec_best': x_dec_best, 'x_last': x_last,
               'errors_rr_discrete': errors_rr_discrete,
               'errors_ht_discrete': errors_ht_discrete,
               'radii_normalized': radii_normalized,
               'radii': radii, 'normalizing_factors': normalizing_factors,
               'f_log': f_log, 'y': y, 'y_ht_discrete': y_ht_discrete, 'S': S,
               'A': A, 'D': D}

  pickle.dump(save_dict,
              gfile.Open(os.path.join(save_dir,
                                      'pgd_%d_%.6f_%s_%d_%s.pkl' %(t_max, lr,
                                                                   normalization,
                                                                   delta,
                                                                   file_suffix)),
                         'w'))


def randomized_rounding(x):
  '''Randomized rounding.'''

  # Discretize
  thresholds = np.random.rand(x.shape[0])
  x_discrete = np.zeros_like(x)
  for idict in range(x.shape[0]):
    for itime in range(x.shape[1]):
      x_discrete[idict, itime] = np.ceil(x[idict, itime] - thresholds[idict])

  return x_discrete


def hard_thresholding(y, radii):
  '''Hard thresholding of y.'''
  y_discrete = np.zeros_like(y)
  for t in range(y.shape[1]):
    l1_radius = radii[t]
    idx = np.argsort(y[:, t])[::-1]
    for iidx in idx:
      y_discrete[iidx, t] = np.ceil(y[iidx, t])
      if y_discrete[:, t].sum() >= l1_radius:
        break
  return y_discrete


def project_l1_pos(x, radii):
  '''Numpy implementation of L1 projection.'''

  # Project to Positvity constrain
  x = np.maximum(x, 0)

  # numpy implementation of L1 projection
  for t in range(x.shape[1]):
    l1_radius = radii[t]
    if np.sum(x[:, t]) < l1_radius:
      continue
    vals = np.sort(x[:, t])
    F = np.cumsum(vals[::-1])[::-1]  # Compute inverse cumulative value to efficiently search of lambda.
    prev_v = np.min(x[:, t])
    for iiv, iv in enumerate(vals):
      if iv == 0:
        continue

      if F[iiv]  - (vals.shape[0] - iiv) * iv < l1_radius :
        break
      prev_v = iv
    vals = np.maximum(x[:, t] - prev_v, 0)
    violation = np.sum(vals) - l1_radius
    nnz = np.sum(vals > 0)
    shift = violation / nnz
    x[:, t] = np.maximum(vals - shift, 0)
  return x

def simultaneous_planning_interleaved_discretization(S, A, D, save_dir='', t_max = 2000, lr=0.01,
                          normalization='T-i', delta = 5,
                          file_suffix='', x_init=None, freeze_freq=200, steps_max=3999):
  ''' Solve the simultaneous planning constrained opt problem with interleaved discretization.

  Let xi be the set of electrodes played till time i.
  Let the distribution of saccades be p(tau).

  Optimization problem.
  Min E_tau ||S - ADx_tau||^2
  subject to -
    x_{i+1} >= x_{i}   forall i
    |x_i|_1 <= i   forall i
    x_i >= 0  forall i.

  Solve using projected gradient descent.

  '''

  # Compute expanded quantities
  S_expanded = np.repeat(np.expand_dims(S, 1), t_max, 1)
  S_normalize = np.sum(S ** 2)

  if normalization == 'T-i':
    normalizing_factors =  np.array([t_max - i  for i in range(t_max)])

  if normalization == 'sqrt(T-i)':
    normalizing_factors =  np.sqrt(np.array([t_max - i  for i in range(t_max)]))

  if normalization == 'C':
    normalizing_factors = (t_max / 2)  + 0 * np.array([t_max - i  for i in range(t_max)])

  # make p_tau uniform between 500 and 2000
  p_tau = np.ones(t_max)

   # TODO(bhaishahster): Dont hardcode p_tau!!
  p_tau[:5] = 0
  p_tau = p_tau / np.sum(p_tau)

  n_dict_elem = D.shape[1]
  # TODO(bhaishahster): Find better initialization.

  # Initialize
  if x_init is not None:
    y_init_normalized =  np.zeros_like(x_init)  # successive difference of x.
    y_init_normalized[:, 0] = x_init[:, 0]
    for iy in np.arange(1, y_init_normalized.shape[1]):
      y_init_normalized[:, iy] = x_init[:, iy] - x_init[:, iy - 1]
    y_init = y_init_normalized * np.expand_dims(normalizing_factors, 0)  #

  else:
    # Zero initialization
    y_init = np.zeros((n_dict_elem, t_max))

  # Smarter initialization
  #x_init =  np.linalg.pinv(D).dot(np.linalg.pinv(A).dot(S_expanded))
  #x_init = project_constraints(x_init)

  #
  # Do projected gradient descent
  y = y_init.copy()
  f_log = []

  # compute variance of dictionary elements.
  stas_norm = np.expand_dims(np.sum(A ** 2, 0) ,0)  # 1 x # cells
  var_dict = np.squeeze(np.dot(stas_norm, D * (1 - D)))  # # dict

  radii_normalized = (np.ones(y.shape[1]))  * delta
  radii = np.multiply(normalizing_factors, radii_normalized )
  x_log = []
  y_best = []
  f_min = np.inf
  training_indices = np.arange(y.shape[1])
  #grad_sq_log = np.zeros_like(y) + 0.001
  to_freeze = False
  for iiter in range(steps_max):

    '''
    if iiter % 500 == 499:
      lr = lr * 0.3
    '''

    if (iiter % freeze_freq == freeze_freq - 1) or to_freeze:
      if training_indices.size == 0:
        print('Everything frozen, Exiting..')
        break

      frozen_index = training_indices[0]
      print('Freezing %d' % frozen_index)
      training_indices = training_indices[1:]
      y[:, frozen_index] = np.squeeze(hard_thresholding(np.expand_dims(y[:, frozen_index], 1), np.array([radii[frozen_index]])))
      #grad_sq_log = np.zeros_like(y) + 0.001

      to_freeze = False
      '''
      try:
        plt.ion()
        plt.plot(f_log)
        plt.show()
        plt.draw()
        plt.pause(0.5)
      except:
        pass
      '''

    # compute x from y
    y_normalized = y / np.expand_dims(normalizing_factors, 0)
    x = np.cumsum(y_normalized, 1)

    # x_log += [x]
    # Compute objective value
    diff = S_expanded - A.dot(D.dot(x))
    errors = np.sum(diff ** 2, 0) + np.dot(var_dict, x)
    f = errors.dot(p_tau)

    print('Iterate: %d, Function value : %.7f' % (iiter, f / S_normalize))
    f_log += [f]
    if f < f_min:
      f_min = f
      y_best = np.copy(y)

    # from IPython import embed; embed()

    # Gradients step
    grad = (D.T.dot(A.T.dot((S_expanded - A.dot(D.dot(x))))) - np.expand_dims(var_dict, 1)) * np.expand_dims(p_tau, 0)
    # collect gradient for each y.
    grad_y = np.cumsum(grad[:, ::-1], 1)
    grad_y = grad_y[:, ::-1] / np.expand_dims(normalizing_factors, 0)
    # y = y + (lr / np.sqrt(iiter + 1)) *  grad_y
    #y[:, training_indices] = y[:, training_indices] + (lr / np.sqrt((iiter % freeze_freq) + 1)) * grad_y[:, training_indices]
    y[:, training_indices] = y[:, training_indices] + (lr) * grad_y[:, training_indices]  # fixed learning rate!!
    #
    # Adagrad
    #grad_sq_log += grad_y ** 2
    #y[:, training_indices] = y[:, training_indices] + (lr) * (grad_y[:, training_indices] / np.sqrt(grad_sq_log))


    # Project to constraint set
    if len(y[:, training_indices].shape) > 1:
      y[:, training_indices] = project_l1_pos(y[:, training_indices], np.array(radii[training_indices]))
    else :
      y[:, training_indices] = np.squeeze(project_l1_pos(np.expand_dims(y[:, training_indices], 1), np.array([radii[training_indices]])))

    if iiter > 2:
      if np.abs(f_log[-2] - f_log[-1]) < 1e-5:
        if freeze_freq < np.inf:
          to_freeze = True

    '''
    if iiter > 2:
      if np.abs(f_log[-2] - f_log[-1]) < 1e-5:
        # compute x from y
        y_normalized = y / np.expand_dims(normalizing_factors, 0)
        x = np.cumsum(y_normalized, 1)
        break
    '''

  if freeze_freq == np.inf:
    print('taking the best value')
    y = y_best  # -> Take the best value.
  y_normalized = y / np.expand_dims(normalizing_factors, 0)
  x = np.cumsum(y_normalized, 1)
  diff = S_expanded - A.dot(D.dot(x))
  errors = np.sum(diff ** 2, 0) + np.dot(var_dict, x)

  # Randomized rounding
  x_rr_discrete = randomized_rounding(x)
  errors_rr_discrete = compute_error(S, A, D, var_dict, x_rr_discrete)

  # Hard thresholding
  y_ht_discrete = hard_thresholding(y, radii)
  y_ht_discrete_normalized = y_ht_discrete / np.expand_dims(normalizing_factors, 0)
  x_ht_discrete = np.cumsum(y_ht_discrete_normalized, 1)
  errors_ht_discrete = compute_error(S, A, D, var_dict, x_ht_discrete)

  #x_log = np.array(x_log)
  #x_decrease = np.sum((x_log - x_log[-1, :, :]) ** 2, 1)
  #x_dec_best = np.sum((x_log - x[:, :]) ** 2, 1)
  #x_last = x_log[-1]
  save_dict = {'x': x, 'x_rr_discrete': x_rr_discrete,
               'x_ht_discrete': x_ht_discrete,
               # 'x_decrease': x_decrease,
               # 'x_dec_best': x_dec_best, 'x_last': x_last,
               'errors': errors,
               'errors_rr_discrete': errors_rr_discrete,
               'errors_ht_discrete': errors_ht_discrete,
               'radii_normalized': radii_normalized,
               'radii': radii, 'normalizing_factors': normalizing_factors,
               'f_log': f_log, 'y': y, 'y_ht_discrete': y_ht_discrete, 'S': S,
               'A': A, 'D': D}

  normalize = np.sum(S ** 2)
  #plt.plot(f_log/normalize)
  #plt.axhline(errors_ht_discrete[5:].mean()/normalize, color='g')
  pickle.dump(save_dict,
                gfile.Open(os.path.join(save_dir,
                                        'pgd_%d_%.6f_%s_%d_%s.pkl' %(t_max, lr,
                                                                   normalization,
                                                                   delta,
                                                                   file_suffix)),
                           'w'))

def simultaneous_planning_interleaved_discretization_exp_gradient(S, A, D, save_dir='', t_max = 2000, lr=0.01,
                          normalization='T-i', delta = 5,
                          file_suffix='', x_init=None, freeze_freq=200, steps_max=3999):
  ''' Solve the simultaneous planning constrained opt problem with interleaved discretization.

  Let xi be the set of electrodes played till time i.
  Let the distribution of saccades be p(tau).

  Optimization problem.
  Min E_tau ||S - ADx_tau||^2
  subject to -
    x_{i+1} >= x_{i}   forall i
    |x_i|_1 <= i   forall i
    x_i >= 0  forall i.

  Solve using projected gradient descent.

  '''

  # Compute expanded quantities
  S_expanded = np.repeat(np.expand_dims(S, 1), t_max, 1)
  S_normalize = np.sum(S ** 2)

  # make p_tau uniform between 500 and 2000
  p_tau = np.ones(t_max)

   # TODO(bhaishahster): Dont hardcode p_tau!!
  p_tau[:5] = 0
  p_tau = p_tau / np.sum(p_tau)

  n_dict_elem = D.shape[1]
  # TODO(bhaishahster): Find better initialization.

  # Initialize
  if x_init is not None:
    y_init =  np.zeros_like(x_init)  # successive difference of x.
    y_init[:, 0] = x_init[:, 0]
    for iy in np.arange(1, y_init.shape[1]):
      y_init[:, iy] = x_init[:, iy] - x_init[:, iy - 1]

    # Add a small delta to every element
    y_init_ones = np.ones((n_dict_elem, t_max))
    y_init_ones = delta * y_init_ones / y_init_ones.sum(0)
    alpha = 0.9
    y_init = alpha * y_init + (1-alpha) * y_init_ones
    y_min = np.min(0.0001 * (1 - alpha) * y_init_ones)
  else:
    # One initialization
    y_init = np.ones((n_dict_elem, t_max))
    y_init = delta * y_init / y_init.sum(0)


  # Do exponential weighing
  y = y_init.copy()
  f_log = []

  # compute variance of dictionary elements.
  stas_norm = np.expand_dims(np.sum(A ** 2, 0) ,0)  # 1 x # cells
  var_dict = np.squeeze(np.dot(stas_norm, D * (1 - D)))  # # dict

  radii = (np.ones(y.shape[1]))  * delta
  x_log = []
  y_best = []
  f_min = np.inf
  training_indices = np.arange(y.shape[1])

  to_freeze = False
  for iiter in range(steps_max):

    if (iiter % freeze_freq == freeze_freq - 1) or to_freeze:
      if training_indices.size == 0:
        print('Everything frozen, Exiting..')
        break

      frozen_index = training_indices[0]
      print('Freezing %d' % frozen_index)
      training_indices = training_indices[1:]
      y[:, frozen_index] = np.squeeze(hard_thresholding(np.expand_dims(y[:, frozen_index], 1), np.array([radii[frozen_index]])))

      # refresh non-frozen entries
      #y_init_ones = np.ones((n_dict_elem, t_max))
      #y_init_ones = delta * y_init_ones / y_init_ones.sum(0)
      #alpha = 0.9
      #y[:, training_indices] = alpha * y[:, training_indices] + (1-alpha) * y_init_ones[:, training_indices]

      to_freeze = False
      '''
      try:
        plt.ion()
        plt.plot(f_log)
        plt.show()
        plt.draw()
        plt.pause(0.5)
      except:
        pass
      '''

    # compute x from y
    x = np.cumsum(y, 1)

    # x_log += [x]
    # Compute objective value
    diff = S_expanded - A.dot(D.dot(x))
    errors = np.sum(diff ** 2, 0) + np.dot(var_dict, x)
    f = errors.dot(p_tau)

    print('Iterate: %d, y_min : %.9f, Function value : %.7f' % (iiter, y_min, f / S_normalize))
    f_log += [f]
    if f < f_min:
      f_min = f
      y_best = np.copy(y)

    # Gradients step
    grad = (D.T.dot(A.T.dot((S_expanded - A.dot(D.dot(x))))) - np.expand_dims(var_dict, 1)) * np.expand_dims(p_tau, 0)
    # collect gradient for each y.
    grad_y = np.cumsum(grad[:, ::-1], 1)
    grad_y = grad_y[:, ::-1]
    y[:, training_indices] = y[:, training_indices] * np.exp(lr * grad_y[:, training_indices] / delta)
    # Keep small elements from going to -inf
    y[:, training_indices] = delta * y[:, training_indices] / np.sum(y[:, training_indices], 0)  # Keeps y normalized
    y[:, training_indices] = np.maximum(y[:, training_indices], y_min)
    if iiter > 2:
      if np.abs(f_log[-2] - f_log[-1]) < 1e-8:
        if freeze_freq < np.inf:
          to_freeze = True

  if freeze_freq == np.inf:
    print('taking the best value')
    y = y_best  # -> Take the best value.
  # use last value of y.
  x = np.cumsum(y, 1)
  diff = S_expanded - A.dot(D.dot(x))
  errors = np.sum(diff ** 2, 0) + np.dot(var_dict, x)

  # Randomized rounding
  x_rr_discrete = randomized_rounding(x)
  errors_rr_discrete = compute_error(S, A, D, var_dict, x_rr_discrete)

  # Hard thresholding
  y_ht_discrete = hard_thresholding(y, radii)
  x_ht_discrete = np.cumsum(y_ht_discrete, 1)
  errors_ht_discrete = compute_error(S, A, D, var_dict, x_ht_discrete)

  save_dict = {'x': x, 'x_rr_discrete': x_rr_discrete,
               'x_ht_discrete': x_ht_discrete,
               'errors': errors,
               'errors_rr_discrete': errors_rr_discrete,
               'errors_ht_discrete': errors_ht_discrete,
               'radii': radii,
               'f_log': f_log, 'y': y, 'y_ht_discrete': y_ht_discrete, 'S': S,
               'A': A, 'D': D}


  normalize = np.sum(S ** 2)
  #plt.plot(f_log/normalize)
  #plt.axhline(errors_ht_discrete[5:].mean()/normalize, color='g')
  pickle.dump(save_dict,
                gfile.Open(os.path.join(save_dir,
                                        'pgd_%d_%.6f_%s_%d_%s.pkl' %(t_max, lr,
                                                                   normalization,
                                                                   delta,
                                                                   file_suffix)),
                           'w'))


if __name__ == '__main__':
  app.run(main)
