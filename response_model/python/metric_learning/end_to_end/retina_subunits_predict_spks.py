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
r"""Predict responses for subunits for many cells.

# pylint: disable-line-too-long

# pylint: enable-line-too-long
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import numpy as np
import random
from absl import app
from absl import gfile
# pylint: disable-unused-import
import retina.response_model.python.metric_learning.end_to_end.config as config  # defines all the flags
# pylint: enable-unused-import
import retina.response_model.python.metric_learning.end_to_end.data_util as data_util
import tensorflow as tf
import scipy.optimize
import pickle
from retina.response_model.python.ASM.su_fit_nov import su_model


tf.app.flags.DEFINE_bool('is_null',
                         False,
                         'Project stimulus to null space of receptive fields?')

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=()):

  #np.random.seed(23)
  #tf.set_random_seed(1234)
  #random.seed(50)

  # 1. Load stimulus-response data.
  # Collect population response across retinas in the list 'responses'.
  # Stimulus for each retina is indicated by 'stim_id',
  # which is found in 'stimuli' dictionary.
  datasets = gfile.ListDirectory(FLAGS.src_dir)
  stimuli = {}
  responses = []
  for icnt, idataset in enumerate(datasets):

    fullpath = os.path.join(FLAGS.src_dir, idataset)
    if gfile.IsDirectory(fullpath):
      key = 'stim_%d' % icnt
      op = data_util.get_stimulus_response(FLAGS.src_dir, idataset, key,
                                           boundary=FLAGS.valid_cells_boundary)
      stimulus, resp, dimx, dimy, _ = op

      stimuli.update({key: stimulus})
      responses += resp

  # 2. Do response prediction for a retina
  iretina = FLAGS.taskid
  subunit_fit_loc = FLAGS.save_folder
  subunits_datasets = gfile.ListDirectory(subunit_fit_loc)

  piece = responses[iretina]['piece']
  matched_dataset = [ifit for ifit in subunits_datasets if piece[:12] == ifit[:12]]
  if matched_dataset == []:
    raise ValueError('Could not find subunit fit')

  subunit_fit_path = os.path.join(subunit_fit_loc, matched_dataset[0])
  stimulus = stimuli[responses[iretina]['stimulus_key']]

  # sample test data
  stimulus_test = stimulus[FLAGS.test_min: FLAGS.test_max, :, :]

  # Optionally, create a null stimulus for all the cells.

  resp_ret = responses[iretina]

  if FLAGS.is_null:
    # Make null stimulus
    stimulus_test = get_null_stimulus(resp_ret, subunit_fit_path, stimulus_test)

  resp_su = get_su_spks(subunit_fit_path, stimulus_test, responses[iretina])

  save_dict = {'resp_su': resp_su.astype(np.int8),
               'cell_ids': responses[iretina]['cellID_list'].squeeze() }

  if FLAGS.is_null:
    save_dict.update({'stimulus_null': stimulus_test})
    save_suff = '_null'
  else:
    save_suff = ''

  pickle.dump(save_dict,
              gfile.Open(os.path.join(subunit_fit_path,
                                      'response_prediction%s.pkl' % save_suff), 'w' ))



def get_su_spks(subunit_fit_path, stimulus_test, resp_ret):
  '''Predict spikes for each cell and each number of subunits. '''

  # Predict responses to 'stimulus_test'
  n_valid_cells = resp_ret['valid_cells'].sum()
  resp_su = np.zeros((10, stimulus_test.shape[0], n_valid_cells))
  cell_ids = resp_ret['cellID_list'].squeeze()

  # time x cells x subunits
  for Nsub in range(1, 11):
    for icell in range(n_valid_cells):

      # Get subunits.
      fit_file = os.path.join(subunit_fit_path, 'Cell_%d_Nsub_%d.pkl' % (cell_ids[icell], Nsub))
      try:
        su_fit = pickle.load(gfile.Open(fit_file, 'r'))
      except:
        print('Cell %d not loaded ' %  cell_ids[icell])
        continue

      print(Nsub, '%d' % cell_ids[icell])

      # Get window to extract stimulus around RF.
      windx = su_fit['windx']
      windy = su_fit['windy']
      stim_cell = np.reshape(stimulus_test[:, windx[0]: windx[1], windy[0]: windy[1]], [stimulus_test.shape[0], -1])

      # Filter in time.
      ttf = su_fit['ttf']
      stim_filter = np.zeros_like(stim_cell)
      for idelay in range(30):
        length = stim_filter[idelay: , :].shape[0]
        stim_filter[idelay: , :] += stim_cell[:length, :] * ttf[idelay]

      stim_filter -= np.mean(stim_filter)
      stim_filter /= np.sqrt(np.var(stim_filter))

      # Compute firing rate.
      K = su_fit['K']
      b = su_fit['b']
      firing_rate = np.exp(stim_filter.dot(K) + b[:, 0]).sum(-1)

      # Sample spikes.
      resp_su[Nsub - 1, :, icell] = np.random.poisson(firing_rate)

  return resp_su


def get_null_stimulus(resp_ret, subunit_fit_path, stimulus_test):
  """Project the sitmulus into null space."""

  A = []

  # Collect RF for all cells
  n_valid_cells = resp_ret['valid_cells'].sum()
  cell_ids = resp_ret['cellID_list'].squeeze()

  # time x cells x subunits
  Nsub = 1
  for icell in range(n_valid_cells):

    # Get subunits.
    fit_file = os.path.join(subunit_fit_path, 'Cell_%d_Nsub_%d.pkl' % (cell_ids[icell], Nsub))
    try:
      su_fit = pickle.load(gfile.Open(fit_file, 'r'))
    except:
      print('Cell %d not loaded ' %  cell_ids[icell])
      continue

    print(Nsub, '%d' % cell_ids[icell])

    # Get window to extract stimulus around RF.
    windx = su_fit['windx']
    windy = su_fit['windy']

    sta = np.zeros((stimulus_test.shape[1], stimulus_test.shape[2]))
    sta[windx[0]: windx[1], windy[0]: windy[1]] = np.reshape(su_fit['K'].squeeze(), (windx[1] - windx[0], windy[1] - windy[0]))
    A += [sta]

  A = np.array(A)
  A_2d = np.reshape(A, (A.shape[0], -1))
  stim_test_2d = np.reshape(stimulus_test, (stimulus_test.shape[0], -1))

  stimulus_test_null = stim_test_2d.T - A_2d.T.dot(np.linalg.solve(A_2d.dot(A_2d.T), A_2d.dot(stim_test_2d.T)))
  stimulus_test_null = stimulus_test_null.T

  stimulus_test_null = np.reshape(stimulus_test_null,
                                  [-1, stimulus_test.shape[1], stimulus_test.shape[2]])
  return stimulus_test_null

if __name__ == '__main__':
  app.run(main)
