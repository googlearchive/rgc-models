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
"""Fit subunits for coarse resolution data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import pickle
from absl import app
from absl import flags
import h5py
import numpy as np
import scipy.io as sio
from tensorflow.python.platform import gfile
from retina.response_model.python.ASM.su_fit_nov import su_model


flags.DEFINE_string('src_dir',
                    '/home/bhaishahster/',
                    'temporary folder on machine for better I/O')

flags.DEFINE_integer('datarun', 3, 'datarun corresponding to null/WN')

flags.DEFINE_string('tmp_dir',
                    '/home/bhaishahster/',
                    'temporary folder on machine for better I/O')

flags.DEFINE_string('save_path', '/home/bhaishahster/'
                    'su_fits_jan_null/',
                    'where to store results')

flags.DEFINE_string('save_path_partial',
                    '/home/bhaishahster/su_fits_jan_null_partial/',
                    'where to store intermediate fits - incase fitting breaks')

flags.DEFINE_string('task_params_file',
                    '/home/bhaishahster/tasks_null_2015-11-09-3_2.txt',
                    'parameters of individual tasks')

flags.DEFINE_integer('taskid', 0, 'Task ID')

FLAGS = flags.FLAGS

rng = np.random


def filterMov_time(mov_filtered, ttf):

  T = mov_filtered.shape[0]
  d = mov_filtered.shape[1]
  stalen = len(ttf)
  for idim in np.arange(d):
      xx = np.zeros((stalen-1+T))
      xx[stalen-1:]=np.squeeze(mov_filtered[:, idim])
      mov_filtered[:, idim] = np.expand_dims(np.convolve(xx, ttf, mode='valid'), 0)
  return mov_filtered


def main(argv):

  # copy WN data
  dst = os.path.join(FLAGS.tmp_dir, 'Off_parasol.mat')

  if not gfile.Exists(dst):
    print('Started Copy')
    src = os.path.join(FLAGS.src_dir, 'Off_parasol.mat')
    if not gfile.IsDirectory(FLAGS.tmp_dir):
      gfile.MkDir(FLAGS.tmp_dir)

    gfile.Copy(src, dst)
    print('File copied to destination')

  else:
    print('File exists')

  # load stimulus
  file=h5py.File(dst, 'r')

  # Load Masked movie
  data = file.get('maskedMovdd')
  stimulus = np.array(data)

  # load cell response
  cells = file.get('cells')
  cells = np.array(cells)
  cells = np.squeeze(cells)

  ttf_log = file.get('ttf_log')
  ttf_avg = file.get('ttf_avg')

  # Load spike Response of cells
  data = file.get('Y')
  responses = np.array(data)

  # get mask
  total_mask_log = np.array(file.get('totalMaskAccept_log'))

  print('Got WN data')

  # Get NULL data
  dat_null = sio.loadmat(gfile.Open(os.path.join(FLAGS.src_dir,
                                                 'OFF_parasol_trial'
                                                 '_resp_data.mat'), 'r'))

  # Load Masked movie
  cids = np.squeeze(np.array(dat_null['cids']))
  condition_idx = FLAGS.datarun
  stimulus_null = dat_null['condMov'][condition_idx, 0]
  stimulus_null = np.transpose(stimulus_null, [2, 1, 0])
  stimulus_null = np.reshape(stimulus_null, [stimulus_null.shape[0], -1])

  resp_cell_log = dat_null['resp_cell_log']
  print('Got Null data')

  # read line corresponding to task
  with gfile.Open(FLAGS.task_params_file, 'r') as f:
    for itask in range(FLAGS.taskid + 1):
      line = f.readline()
  line = line[:-1]  # Remove \n from end.
  print(line)

  # get task parameters by parsing the lines
  line_split = line.split(';')
  cell_idx = line_split[0]
  cell_idx = cell_idx[1:-1].split(',')
  cell_idx = [int(i) for i in cell_idx]

  Nsub = int(line_split[1])
  projection_type = line_split[2]
  lam_proj = float(line_split[3])

  #ipartition = int(line_split[4])
  #cell_idx_mask = cell_idx

  partitions_fit = line_split[4]
  partitions_fit = partitions_fit[1:-1].split(',')
  partitions_fit = [int(i) for i in partitions_fit]

  if len(line_split) == 5:
    cell_idx_mask = cell_idx
  else:
    cell_idx_mask = line_split[5]
    cell_idx_mask = cell_idx_mask[1:-1].split(',')
    cell_idx_mask = [int(i) for i in cell_idx]
  ##
  ##

  print(cell_idx)
  print(Nsub)
  print(cell_idx_mask)

  mask = (total_mask_log[cell_idx_mask, :].sum(0) != 0)
  mask_matrix = np.reshape(mask != 0, [40, 80])

  # make mask bigger - add one row one left/right
  r, c = np.where(mask_matrix)
  mask_matrix[r.min()-1: r.max()+1, c.min()-1:c.max()+1] = True
  neighbor_mat = su_model.get_neighbormat(mask_matrix, nbd=1)
  mask = np.ndarray.flatten(mask_matrix)

  ## WN preprocess
  stim_use_wn = stimulus[:, mask]
  resp_use_wn = responses[:, cell_idx]

  # get last 10% as test data
  np.random.seed(23)

  frac_test = 0.1
  tms_test = np.arange(np.floor(stim_use_wn.shape[0]*(1 - frac_test)),
                       1*np.floor(stim_use_wn.shape[0])).astype(np.int)

  # Random partitions
  n_partitions = 10
  tms_train_validate = np.arange(0, np.floor(stim_use_wn.shape[0] *
                                             (1 - frac_test))).astype(np.int)

  frac_validate = 0.1

  partitions_wn = []
  for _ in range(n_partitions):
    perm = np.random.permutation(tms_train_validate)
    tms_train = perm[0: np.int(np.floor((1 - frac_validate) * perm.shape[0]))]
    tms_validate = perm[np.int(np.floor((1 - frac_validate) *
                                        perm.shape[0])): np.int(perm.shape[0])]

    partitions_wn += [{'tms_train': tms_train,
                       'tms_validate': tms_validate,
                       'tms_test': tms_test}]

  print('Made partitions')
  print('WN data preprocessed')

  ## NULL preprocess
  stimulus_use_null = stimulus_null[:, mask]
  ttf_use = np.array(ttf_log[cell_idx, :]).astype(np.float32).squeeze()
  stimulus_use_null = filterMov_time(stimulus_use_null, ttf_use)

  if len(cell_idx) > 1:
    print('More than 1 cell not supported!')

  try:
    resp_use_null = np.array(resp_cell_log[cell_idx[0],
                             0][condition_idx, 0]).T.astype(np.float32)
  except:
    resp_use_null = np.array(resp_cell_log[cell_idx[0],
                             0][0, condition_idx].T).astype(np.float32)

  # Remove first 30 frames due to convolution artifact.
  stimulus_use_null = stimulus_use_null[30:, :]
  resp_use_null = resp_use_null[30:, :]

  n_trials = resp_use_null.shape[1]
  t_null = resp_use_null.shape[0]
  tms_train_1tr_null = np.arange(np.floor(t_null/2)).astype(np.int)
  tms_test_1tr_null = np.arange(np.ceil(t_null/2), t_null).astype(np.int)

  # repeat in time dimension, divide into training and testing.
  stimulus_use_null = np.tile(stimulus_use_null.T, n_trials).T
  resp_use_null = np.ndarray.flatten(resp_use_null.T)
  resp_use_null = np.expand_dims(resp_use_null, 1)

  tms_train_null = np.array([])
  tms_test_null = np.array([])
  for itrial in range(n_trials):
    tms_train_null = np.append(tms_train_null,
                               tms_train_1tr_null + itrial * t_null)
    tms_test_null = np.append(tms_test_null,
                              tms_test_1tr_null + itrial * t_null)
  tms_train_null = tms_train_null.astype(np.int)
  tms_test_null = tms_test_null.astype(np.int)

  print('NULL data preprocessed')

  ss = '_'.join([str(cells[ic]) for ic in cell_idx])

  for ipartition in partitions_fit:
    save_filename = os.path.join(FLAGS.save_path,
                                 'Cell_%s_nsub_%d_%s_%.3f_part_%d_jnt.pkl' %
                                 (ss, Nsub, projection_type,
                                  lam_proj, ipartition))

    save_filename_partial = os.path.join(FLAGS.save_path_partial,
                                         'Cell_%s_nsub_%d_%s_%.3f_part_%d_'
                                         'jnt.pkl' %
                                         (ss, Nsub, projection_type,
                                          lam_proj, ipartition))

    ## Do fitting
    # Fit SU on WN
    if not gfile.Exists(save_filename):
      print('Fitting started on WN')
      op = su_model.Flat_clustering_jnt(stim_use_wn, resp_use_wn, Nsub,
                                        partitions_wn[ipartition]['tms_train'],
                                        partitions_wn[ipartition][
                                            'tms_validate'],
                                        steps_max=10000, eps=1e-9,
                                        projection_type=projection_type,
                                        neighbor_mat=neighbor_mat,
                                        lam_proj=lam_proj, eps_proj=0.01,
                                        save_filename_partial=
                                        save_filename_partial,
                                        fitting_phases=[1])

      _, _, alpha, lam_log_wn, lam_log_test_wn, fitting_phase, fit_params_wn = op
      print('Fitting done on WN')

      # Fit on NULL
      op = su_model.fit_scales(stimulus_use_null[tms_train_null, :],
                               resp_use_null[tms_train_null, :],
                               stimulus_use_null[tms_test_null, :],
                               resp_use_null[tms_test_null, :],
                               Ns=Nsub,
                               K=fit_params_wn[0][0], b=fit_params_wn[0][1],
                               params=fit_params_wn[0][2], lr=0.01, eps=1e-9)

      k_null, b_null, nl_params_null, lam_log_null, lam_log_test_null = op

      # Collect results and save
      fit_params = fit_params_wn + [[k_null, b_null, nl_params_null]]
      lam_log = [lam_log_wn, np.array(lam_log_null)]
      lam_log_test = [lam_log_test_wn, np.array(lam_log_test_null)]

      save_dict = {'lam_log': lam_log, 'lam_log_test': lam_log_test,
                   'fit_params': fit_params}
      pickle.dump(save_dict, gfile.Open(save_filename, 'w'))
      print('Saved results')


if __name__ == '__main__':
  app.run(main)


