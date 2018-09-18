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
r"""Fit subunits for coarse resolution data."""

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

flags.DEFINE_string('tmp_dir',
                    '/home/bhaishahster/',
                    'temporary folder on machine for better I/O')

flags.DEFINE_string('save_path', '/home/bhaishahster/'
                    'su_fits_jan_nsem/',
                    'where to store results')

flags.DEFINE_string('save_path_partial',
                    '/home/bhaishahster/su_fits_jan_nsem_partial/',
                    'where to store intermediate fits - incase fitting breaks')

flags.DEFINE_string('task_params_file',
                    '/home/bhaishahster/tasks_nsem_2015-11-09-3.txt',
                    'parameters of individual tasks')
flags.DEFINE_integer('taskid', 0, 'Task ID')

FLAGS = flags.FLAGS


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

  # Get NSEM data
  dat_nsem_mov = sio.loadmat(gfile.Open('/home/bhaishahster/nsem_data/'
                                        'pc2015_10_29_2/NSinterval_30_025.mat',
                                        'r'))
  stimulus_nsem = dat_nsem_mov['mov']

  stimulus_nsem = np.transpose(stimulus_nsem, [2, 1, 0])
  stimulus_nsem = np.reshape(stimulus_nsem, [stimulus_nsem.shape[0], -1])

  dat_nsem_resp = sio.loadmat(gfile.Open('/home/bhaishahster/nsem_data/'
                                         'pc2015_10_29_2/OFF_parasol_trial_resp'
                                         '_data_NSEM_data039.mat', 'r'))
  responses_nsem = dat_nsem_resp['resp_cell_log']
  print('Git NSEM data')

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
  ipartition = int(line_split[4])

  cell_idx_mask = cell_idx

  ##

  print(cell_idx)
  print(Nsub)
  print(cell_idx_mask)

  mask = (total_mask_log[cell_idx_mask, :].sum(0) != 0)
  mask_matrix = np.reshape(mask!=0, [40, 80])

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
  tms_train_validate = np.arange(0, np.floor(stim_use_wn.shape[0]*(1 - frac_test))).astype(np.int)

  frac_validate = 0.1

  partitions_wn = []
  for _ in range(n_partitions):
    perm = np.random.permutation(tms_train_validate)
    tms_train = perm[0: np.floor((1 - frac_validate) * perm.shape[0])]
    tms_validate = perm[np.floor((1 - frac_validate) * perm.shape[0]): perm.shape[0]]

    partitions_wn += [{'tms_train': tms_train,
                    'tms_validate': tms_validate,
                    'tms_test': tms_test}]

  print('Made partitions')
  print('WN data preprocessed')

  ## NSEM preprocess
  stim_use_nsem = stimulus_nsem[:, mask]
  ttf_use = np.array(ttf_log[cell_idx, :]).astype(np.float32).squeeze()
  stim_use_nsem = filterMov_time(stim_use_nsem, ttf_use)
  resp_use_nsem = np.array(responses_nsem[cell_idx][0, 0]).astype(np.float32).T

  # Remove first 30 frames due to convolution artifact.
  stim_use_nsem = stim_use_nsem[30:, :]
  resp_use_nsem = resp_use_nsem[30:, :]

  n_trials = resp_use_nsem.shape[1]
  t_nsem = resp_use_nsem.shape[0]
  tms_train_1tr_nsem = np.arange(np.floor(t_nsem/2))
  tms_test_1tr_nsem = np.arange(np.ceil(t_nsem/2), t_nsem)

  # repeat in time dimension, divide into training and testing.
  stim_use_nsem = np.tile(stim_use_nsem.T, n_trials).T
  resp_use_nsem = np.ndarray.flatten(resp_use_nsem.T)
  resp_use_nsem = np.expand_dims(resp_use_nsem, 1)

  tms_train_nsem = np.array([])
  tms_test_nsem = np.array([])
  for itrial in range(n_trials):
    tms_train_nsem = np.append(tms_train_nsem,
                               tms_train_1tr_nsem + itrial * t_nsem)
    tms_test_nsem = np.append(tms_test_nsem,
                              tms_test_1tr_nsem + itrial * t_nsem)
  tms_train_nsem = tms_train_nsem.astype(np.int)
  tms_test_nsem = tms_test_nsem.astype(np.int)

  print('NSEM data preprocessed')

  ss = '_'.join([str(cells[ic]) for ic in cell_idx])

  save_filename = os.path.join(FLAGS.save_path,
                               'Cell_%s_nsub_%d_%s_%.3f_part_%d_jnt.pkl' %
                               (ss, Nsub, projection_type,
                                lam_proj, ipartition))

  save_filename_partial = os.path.join(FLAGS.save_path_partial,
                               'Cell_%s_nsub_%d_%s_%.3f_part_%d_jnt.pkl' %
                               (ss, Nsub, projection_type,
                                lam_proj, ipartition))

  ## Do fitting
  # Fit SU on WN
  print('Fitting started on WN')
  op = su_model.Flat_clustering_jnt(stim_use_wn, resp_use_wn, Nsub,
                                    partitions_wn[ipartition]['tms_train'],
                                    partitions_wn[ipartition]['tms_validate'],
                                    steps_max=10000, eps=1e-9,
                                    projection_type=projection_type,
                                    neighbor_mat=neighbor_mat,
                                    lam_proj=lam_proj, eps_proj=0.01,
                                    save_filename_partial=save_filename_partial,
                                    fitting_phases=[1])

  _, _, alpha, lam_log_wn, lam_log_test_wn, fitting_phase, fit_params_wn = op
  print('Fitting done on WN')

  # Fit on NSEM
  op = su_model.fit_scales(stim_use_nsem[tms_train_nsem, :],
                           resp_use_nsem[tms_train_nsem, :],
                           stim_use_nsem[tms_test_nsem, :],
                           resp_use_nsem[tms_test_nsem, :],
                           Ns=Nsub,
                           K=fit_params_wn[0][0], b=fit_params_wn[0][1],
                           params=fit_params_wn[0][2], lr=0.1, eps=1e-9)

  K_nsem, b_nsem, nl_params_nsem, lam_log_nsem, lam_log_test_nsem = op

  # Collect results and save
  fit_params = fit_params_wn + [[K_nsem, b_nsem, nl_params_nsem]]
  lam_log = [lam_log_wn, np.array(lam_log_nsem)]
  lam_log_test = [lam_log_test_wn, np.array(lam_log_test_nsem)]

  save_dict = {'lam_log': lam_log, 'lam_log_test': lam_log_test,
               'fit_params': fit_params}
  pickle.dump(save_dict, gfile.Open(save_filename, 'w' ))
  print('Saved results')


if __name__ == '__main__':
  app.run(main)


