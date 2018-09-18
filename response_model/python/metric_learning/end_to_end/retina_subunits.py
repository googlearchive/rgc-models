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
r"""Fit subunits in multiple retina.

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

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=()):

  #np.random.seed(23)
  #tf.set_random_seed(1234)
  #random.seed(50)

  # Load stimulus-response data.
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

  taskid = FLAGS.taskid
  dat = responses[taskid]
  stimulus = stimuli[dat['stimulus_key']]

  # parameters
  window = 5

  # Compute time course and non-linearity as two parameters which might be should be explored in embedded space.
  n_cells = dat['responses'].shape[1]
  cell_ids = dat['cellID_list'].squeeze()
  T = np.minimum(stimulus.shape[0], dat['responses'].shape[0])

  stim_short = stimulus[FLAGS.train_min: np.minimum(FLAGS.train_max, T), :, :]
  resp_short = dat['responses'][FLAGS.train_min: np.minimum(FLAGS.train_max, T),
                                :].astype(np.float32)

  print('Stimulus', stim_short.shape)
  print('Response', resp_short.shape)

  # from IPython import embed; embed()

  ########################################################################
  # compute STAs
  ########################################################################
  stas = np.zeros((n_cells, 80, 40, 30))
  for icell in range(n_cells):
    if resp_short[:, icell].sum(0) < 10:
      print('Too few spikes, skip')
      continue
    print(icell)
    center = dat['centers'][icell, :].astype(np.int)
    windx = [np.maximum(center[0]-window, 0), np.minimum(center[0]+window, 80-1)]
    windy = [np.maximum(center[1]-window, 0), np.minimum(center[1]+window, 40-1)]
    stim_cell = np.reshape(stim_short[:, windx[0]: windx[1], windy[0]: windy[1]], [stim_short.shape[0], -1])
    for idelay in range(30):
      length = resp_short[idelay:, icell].shape[0]
      print(idelay, length)
      stas[icell, windx[0]: windx[1], windy[0]: windy[1], idelay] = np.reshape(resp_short[idelay:, icell].dot(stim_cell[0: length, :]),
                                                                           [windx[1] - windx[0], windy[1] - windy[0]]) / np.sum(resp_short[idelay:, icell])
  # save_dict.update({'stas_dict': stas_dict})


  ########################################################################
  # Find time courses for each cell
  ########################################################################
  ttf_log = []
  for icell in range(n_cells):
    if resp_short[:, icell].sum(0) < 10:
      print('Too few spikes, skip')
      ttf_log += [np.zeros(30)]
      continue
    center = dat['centers'][icell, :].astype(np.int)
    windx = [np.maximum(center[0]-window, 0), np.minimum(center[0]+window, 80-1)]
    windy = [np.maximum(center[1]-window, 0), np.minimum(center[1]+window, 40-1)]
    ll = stas[icell, windx[0]: windx[1], windy[0]: windy[1], :]
    ll_2d = np.reshape(ll, [-1, ll.shape[-1]])
    u, s, v = np.linalg.svd(ll_2d)
    ttf_log += [v[0, :]]

  ttf_log = np.array(ttf_log)


  '''
  plt.plot(ttf_corrected[np.squeeze(dat['cell_type'])==1, :].T, 'r', alpha=0.3)
  plt.plot(ttf_corrected[np.squeeze(dat['cell_type'])==2, :].T, 'k', alpha=0.3)

  plt.plot(ttf_mean_1, 'r--')
  plt.plot(ttf_mean_2, 'k--')
  '''

  ########################################################################
  ## Find subunits
  ########################################################################

  dir_scratch = '/home/bhaishahster/stim-resp_collection_big_wn_retina_subunit_properties_train_scratch/%s_taskid_%d' % (dat['piece'][:-4], FLAGS.taskid)
  if not gfile.Exists(dir_scratch):
    gfile.MkDir(dir_scratch)

  dir_save = os.path.join(FLAGS.save_folder, '%s_taskid_%d' % (dat['piece'][:-4], FLAGS.taskid))
  if not gfile.Exists(dir_save):
    gfile.MkDir(dir_save)

  for icell in np.random.permutation(np.arange(n_cells)):
    print(icell)
    if resp_short[:, icell].sum(0) < 10:
      print('Too few spikes, skip')
      continue
    center = dat['centers'][icell, :].astype(np.int)
    windx = [np.maximum(center[0]-window, 0), np.minimum(center[0]+window, 80-1)]
    windy = [np.maximum(center[1]-window, 0), np.minimum(center[1]+window, 40-1)]

    stim_cell = np.reshape(stim_short[:, windx[0]: windx[1], windy[0]: windy[1]], [stim_short.shape[0], -1])

    stim_filter = np.zeros_like(stim_cell)

    for idelay in range(30):
      length = stim_filter[idelay: , :].shape[0]
      stim_filter[idelay: , :] += stim_cell[:length, :] * ttf_log[icell, idelay]

    # Normalize stim_filter
    stim_filter -= np.mean(stim_filter)
    stim_filter /= np.sqrt(np.var(stim_filter))

    resp_cell = resp_short[:, icell]

    for Nsub in np.arange(1, 11):
      print(icell, Nsub)
      cell_su_fname = os.path.join(dir_save, 'Cell_%d_Nsub_%d.pkl' % (cell_ids[icell], Nsub))
      if gfile.Exists(cell_su_fname):
        continue
      op = su_model.Flat_clustering_jnt(stim_filter, np.expand_dims(resp_cell, 1), Nsub,
                                      np.arange(FLAGS.train_min, stim_filter.shape[0]),
                                      np.arange(FLAGS.test_min + 30, FLAGS.test_max),
                                      steps_max=200, eps=1e-9,
                                      projection_type=None,
                                      neighbor_mat=None,
                                      lam_proj=0, eps_proj=0.01,
                                      save_filename_partial=os.path.join(dir_scratch, 'Cell_%d_Nsub_%d.pkl' % (icell, Nsub)),
                                      fitting_phases=[1])

      K, b, alpha, lam_log, lam_log_test, fitting_phase, fit_params = op

      save_dict = {'K': K, 'b': b,
                   'lam_log': lam_log, 'lam_log_test': lam_log_test,
                   'fitting_phase': fitting_phase, 'fit_params': fit_params,
                   'ttf': ttf_log[icell, :], 'windx': windx, 'windy': windy, 'center': center}
      pickle.dump(save_dict,
                  gfile.Open(cell_su_fname, 'w' ))



def get_times(ttf):
  max_time = np.argmax(np.abs(ttf))
  max_sign = np.sign(ttf[max_time])

  second_max = np.argmax(-1 * max_sign * ttf)
  max_times = np.sort([max_time, second_max])
  zero_crossing = np.argmin((ttf[max_times[0]: max_times[1]])**2)


  delay = np.where(np.abs(ttf) > 0.1 * np.max(np.abs(ttf)))[0]
  delay = np.min(delay)

  print(max_time, second_max, zero_crossing, max_sign, delay)
  return max_time, second_max, zero_crossing, max_sign, delay

if __name__ == '__main__':
  app.run(main)


if __name__ == '__main__':
  app.run(main)
