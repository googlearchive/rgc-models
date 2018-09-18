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
r"""Find properties of multiple retina
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

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=()):

  np.random.seed(23)
  tf.set_random_seed(1234)
  random.seed(50)

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
  T = np.minimum(stimulus.shape[0], dat['responses'].shape[0])

  stim_short = stimulus[:T, :, :]
  resp_short = dat['responses'][:T, :].astype(np.float32)

  save_dict = {}

  # Find time course, non-linearity and RF parameters

  ########################################################################
  # Separation between cell types
  ########################################################################
  save_dict.update({'cell_type': dat['cell_type']})
  save_dict.update({'dist_nn_cell_type': dat['dist_nn_cell_type']})

  ########################################################################
  # Find mean firing rate
  ########################################################################
  mean_fr = dat['responses'].mean(0)
  mean_fr_1 = np.mean(mean_fr[np.squeeze(dat['cell_type'])==1])
  mean_fr_2 = np.mean(mean_fr[np.squeeze(dat['cell_type'])==2])

  mean_fr_dict = {'mean_fr': mean_fr,
                  'mean_fr_1': mean_fr_1, 'mean_fr_2': mean_fr_2}
  save_dict.update({'mean_fr_dict': mean_fr_dict})

  ########################################################################
  # compute STAs
  ########################################################################
  stas = np.zeros((n_cells, 80, 40, 30))
  for icell in range(n_cells):
    print(icell)
    center = dat['centers'][icell, :].astype(np.int)
    windx = [np.maximum(center[0]-window, 0), np.minimum(center[0]+window, 80-1)]
    windy = [np.maximum(center[1]-window, 0), np.minimum(center[1]+window, 40-1)]
    stim_cell = np.reshape(stim_short[:, windx[0]: windx[1], windy[0]: windy[1]], [stim_short.shape[0], -1])
    for idelay in range(30):
      stas[icell, windx[0]: windx[1], windy[0]: windy[1], idelay] = np.reshape(resp_short[idelay:, icell].dot(stim_cell[:T-idelay, :]),
                                                                           [windx[1] - windx[0], windy[1] - windy[0]]) / np.sum(resp_short[idelay:, icell])

  stas_dict = {'stas': stas}
  # save_dict.update({'stas_dict': stas_dict})


  ########################################################################
  # Find time courses for each cell
  ########################################################################
  ttf_log = []
  for icell in range(n_cells):
    center = dat['centers'][icell, :].astype(np.int)
    windx = [np.maximum(center[0]-window, 0), np.minimum(center[0]+window, 80-1)]
    windy = [np.maximum(center[1]-window, 0), np.minimum(center[1]+window, 40-1)]
    ll = stas[icell, windx[0]: windx[1], windy[0]: windy[1], :]

    ll_2d = np.reshape(ll, [-1, ll.shape[-1]])

    u, s, v = np.linalg.svd(ll_2d)

    ttf_log += [v[0, :]]

  ttf_log = np.array(ttf_log)
  signs = [np.sign(ttf_log[icell, np.argmax(np.abs(ttf_log[icell, :]))]) for icell in range(ttf_log.shape[0])]
  ttf_corrected = np.expand_dims(np.array(signs), 1) * ttf_log
  ttf_corrected[np.squeeze(dat['cell_type'])==1, :] = ttf_corrected[np.squeeze(dat['cell_type'])==1, :] * -1

  ttf_mean_1 = ttf_corrected[np.squeeze(dat['cell_type'])==1, :].mean(0)
  ttf_mean_2 = ttf_corrected[np.squeeze(dat['cell_type'])==2, :].mean(0)

  ttf_params_1 = get_times(ttf_mean_1)
  ttf_params_2 = get_times(ttf_mean_2)

  ttf_dict = {'ttf_log': ttf_log,
              'ttf_mean_1': ttf_mean_1, 'ttf_mean_2': ttf_mean_2,
              'ttf_params_1': ttf_params_1, 'ttf_params_2': ttf_params_2}

  save_dict.update({'ttf_dict': ttf_dict})
  '''
  plt.plot(ttf_corrected[np.squeeze(dat['cell_type'])==1, :].T, 'r', alpha=0.3)
  plt.plot(ttf_corrected[np.squeeze(dat['cell_type'])==2, :].T, 'k', alpha=0.3)

  plt.plot(ttf_mean_1, 'r--')
  plt.plot(ttf_mean_2, 'k--')
  '''

  ########################################################################
  ## Find non-linearity
  ########################################################################
  f_nl = lambda x, p0, p1, p2, p3: p0 + p1*x + p2* np.power(x, 2) + p3* np.power(x, 3)

  nl_params_log = []
  stim_resp_log = []
  for icell in range(n_cells):
    print(icell)
    center = dat['centers'][icell, :].astype(np.int)
    windx = [np.maximum(center[0]-window, 0), np.minimum(center[0]+window, 80-1)]
    windy = [np.maximum(center[1]-window, 0), np.minimum(center[1]+window, 40-1)]

    stim_cell = np.reshape(stim_short[:, windx[0]: windx[1], windy[0]: windy[1]], [stim_short.shape[0], -1])
    sta_cell = np.reshape(stas[icell, windx[0]: windx[1], windy[0]: windy[1], :], [-1, stas.shape[-1]])

    stim_filter = np.zeros(stim_short.shape[0])
    for idelay in range(30):
      stim_filter[idelay: ] += stim_cell[:T-idelay, :].dot(sta_cell[:, idelay])

    # Normalize stim_filter
    stim_filter -= np.mean(stim_filter)
    stim_filter /= np.sqrt(np.var(stim_filter))

    resp_cell = resp_short[:, icell]

    stim_nl = []
    resp_nl = []
    for ipercentile in range(3, 97, 1):
      lb = np.percentile(stim_filter, ipercentile-3)
      ub = np.percentile(stim_filter, ipercentile+3)
      tms = np.logical_and(stim_filter >= lb, stim_filter < ub)
      stim_nl += [np.mean(stim_filter[tms])]
      resp_nl += [np.mean(resp_cell[tms])]

    stim_nl = np.array(stim_nl)
    resp_nl = np.array(resp_nl)

    popt, pcov = scipy.optimize.curve_fit(f_nl, stim_nl, resp_nl, p0=[1, 0, 0, 0])
    nl_params_log += [popt]
    stim_resp_log += [[stim_nl, resp_nl]]

  nl_params_log = np.array(nl_params_log)

  np_params_mean_1 = np.mean(nl_params_log[np.squeeze(dat['cell_type'])==1, :], 0)
  np_params_mean_2 = np.mean(nl_params_log[np.squeeze(dat['cell_type'])==2, :], 0)

  nl_params_dict = {'nl_params_log': nl_params_log,
                    'np_params_mean_1': np_params_mean_1,
                    'np_params_mean_2': np_params_mean_2,
                    'stim_resp_log': stim_resp_log}

  save_dict.update({'nl_params_dict': nl_params_dict})

  '''
  # Visualize Non-linearities
  for icell in range(n_cells):

    stim_in = np.arange(-3, 3, 0.1)
    fr = f_nl(stim_in, *nl_params_log[icell, :])
    if np.squeeze(dat['cell_type'])[icell] == 1:
      c = 'r'
    else:
      c = 'k'
    plt.plot(stim_in, fr, c, alpha=0.2)

  fr = f_nl(stim_in, *np_params_mean_1)
  plt.plot(stim_in, fr, 'r--')

  fr = f_nl(stim_in, *np_params_mean_2)
  plt.plot(stim_in, fr, 'k--')
  '''

  pickle.dump(save_dict, gfile.Open(os.path.join(FLAGS.save_folder , dat['piece']), 'w'))
  pickle.dump(stas_dict, gfile.Open(os.path.join(FLAGS.save_folder , 'stas' + dat['piece']), 'w'))

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
