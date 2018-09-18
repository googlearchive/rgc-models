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
from tensorflow.python.platform import gfile
from retina.response_model.python.ASM.su_fit_nov import su_model

flags.DEFINE_string('src_dir',
                    '/home/bhaishahster/',
                    'temporary folder on machine for better I/O')

flags.DEFINE_string('tmp_dir',
                    '/home/bhaishahster/',
                    'temporary folder on machine for better I/O')

'''
flags.DEFINE_string('save_path', '/home/bhaishahster/su_fits_nov_pop/',
                    'where to store results')

flags.DEFINE_string('save_path_partial', '/home/bhaishahster/'
                                         'su_fits_nov_pop_partial/',
                    'where to store intermediate fits - incase fitting breaks')

flags.DEFINE_string('task_params_file',
                    '/home/bhaishahster/tasks_coarse_pop_2.txt',
                    'parameters of individual tasks')
'''

flags.DEFINE_string('save_path', '/home/bhaishahster/'
                    'su_fits_aug15_2018_pop/',
                    'where to store results')

flags.DEFINE_string('save_path_partial', '/home/bhaishahster/'
                    'su_fits_aug15_2018_pop_partial/',
                    'where to store intermediate fits - incase fitting breaks')

flags.DEFINE_string('task_params_file',
                    '/home/bhaishahster/tasks_coarse_aug.txt',
                    'parameters of individual tasks')

rng = np.random

flags.DEFINE_integer('taskid', 0, 'Task ID')

FLAGS = flags.FLAGS


def main(argv):

  # copy data
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
  file_data = h5py.File(dst, 'r')

  # Load Masked movie
  data = file_data.get('maskedMovdd')
  stimulus = np.array(data)

  # load cell response
  cells = file_data.get('cells')
  cells = np.array(cells)
  cells = np.squeeze(cells)

  ttf_log = file_data.get('ttf_log')
  ttf_avg = file_data.get('ttf_avg')

  # Load spike Response of cells
  data = file_data.get('Y')
  responses = np.array(data)

  # get mask
  total_mask_log = np.array(file_data.get('totalMaskAccept_log'))

  print('Got data')

  # read line corresponding to task
  with gfile.Open(FLAGS.task_params_file, 'r') as f:
    for _ in range(FLAGS.taskid + 1):
      line = f.readline()
  line = line[:-1]  # Remove \n from end.
  print(line)

  # get task parameters by parsing the lines
  line_split = line.split(';')
  cell_idx = line_split[0]
  cell_idx = cell_idx[1:-1].split(',')
  cell_idx = [int(i) for i in cell_idx]

  nsub = int(line_split[1])
  projection_type = line_split[2]
  lam_proj = float(line_split[3])

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

  print(cell_idx)
  print(nsub)
  print(cell_idx_mask)

  mask = (total_mask_log[cell_idx_mask, :].sum(0) != 0)
  mask_matrix = np.reshape(mask != 0, [40, 80])

  # make mask bigger - add one row one left/right
  r, c = np.where(mask_matrix)
  mask_matrix[r.min()-1: r.max()+1, c.min()-1:c.max()+1] = True
  neighbor_mat = su_model.get_neighbormat(mask_matrix, nbd=1)
  mask = np.ndarray.flatten(mask_matrix)

  stim_use = stimulus[:, mask]
  resp_use = responses[:, cell_idx]

  print('Prepared data')

  # get last 10% as test data
  np.random.seed(23)

  frac_test = 0.1
  tms_test = np.arange(np.floor(stim_use.shape[0]*(1 - frac_test)),
                       1*np.floor(stim_use.shape[0])).astype(np.int)

  # Random partitions
  n_partitions = 10
  tms_train_validate = np.arange(0, np.floor(
      stim_use.shape[0]*(1 - frac_test))).astype(np.int)

  frac_validate = 0.1

  partitions = []
  for _ in range(n_partitions):
    perm = np.random.permutation(tms_train_validate)
    tms_train = perm[0: np.int(np.floor((1 - frac_validate) * perm.shape[0]))]
    tms_validate = perm[np.int(np.floor(
        (1 - frac_validate) * perm.shape[0])): np.int(perm.shape[0])]
    partitions += [{'tms_train': tms_train,
                    'tms_validate': tms_validate,
                    'tms_test': tms_test}]

  print('Made partitions')

  # Do fitting
  # tms_train = np.arange(0, np.floor(stim_use.shape[0] * 0.8)).astype(np.int)
  # tms_test = np.arange(np.floor(stim_use.shape[0] * 0.8),
  #                       1 * np.floor(stim_use.shape[0] * 0.9)).astype(np.int)

  ss = '_'.join([str(cells[ic]) for ic in cell_idx])
  for ipartition in partitions_fit:
    save_filename = os.path.join(FLAGS.save_path,
                                 'Cell_%s_nsub_%d_%s_%.3f_part_%d_jnt.pkl' %
                                 (ss, nsub, projection_type,
                                  lam_proj, ipartition))

    save_filename_partial = os.path.join(FLAGS.save_path_partial,
                                         'Cell_%s_nsub_%d_%s_%.3f_part_%d_jnt.p'
                                         'kl' %
                                         (ss, nsub, projection_type,
                                          lam_proj, ipartition))

    if not gfile.Exists(save_filename):
      print('Fitting started')
      op = su_model.Flat_clustering_jnt(stim_use, resp_use, nsub,
                                        partitions[ipartition]['tms_train'],
                                        partitions[ipartition]['tms_validate'],
                                        steps_max=10000, eps=1e-9,
                                        projection_type=projection_type,
                                        neighbor_mat=neighbor_mat,
                                        lam_proj=lam_proj, eps_proj=0.01,
                                        save_filename_partial=
                                        save_filename_partial)

      k, b, _, lam_log, lam_log_test, fitting_phase, fit_params = op

      tms_test = partitions[ipartition]['tms_test']
      lam_test_on_test_data = su_model.compute_fr_loss(fit_params[2][0],
                                                       fit_params[2][1],
                                                       stim_use[tms_test, :],
                                                       resp_use[tms_test, :],
                                                       nl_params=
                                                       fit_params[2][2])

      print('Fitting done')
      save_dict = {'K': k, 'b': b,
                   'lam_log': lam_log, 'lam_log_validate': lam_log_test,
                   'lam_test': lam_test_on_test_data,
                   'fitting_phase': fitting_phase, 'fit_params': fit_params}
      pickle.dump(save_dict, gfile.Open(save_filename, 'w'))
      print('Saved results')


if __name__ == '__main__':
  app.run(main)
