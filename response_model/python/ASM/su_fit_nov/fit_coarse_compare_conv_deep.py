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
r"""Fit subunits for coarse resolution data.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import pickle
from absl import app
from absl import flags
import h5py
import numpy as np
from retina.response_model.python.ASM.su_fit_nov import conv_model
from retina.response_model.python.ASM.su_fit_nov import su_model
from tensorflow.python.platform import gfile

flags.DEFINE_string('src_dir',
                    '/home/bhaishahster/',
                    'temporary folder on machine for better I/O')

flags.DEFINE_string('tmp_dir',
                    '/home/bhaishahster/',
                    'temporary folder on machine for better I/O')

flags.DEFINE_string('save_path', '/home/bhaishahster/'
                    'su_fits_dec_compare_conv_deep/',
                    'where to store results')

flags.DEFINE_string('save_path_partial', '/home/bhaishahster/'
                    'su_fits_dec_compare_conv_deep_partial/',
                    'where to store intermediate fits - incase fitting breaks')

flags.DEFINE_string('task_params_file',
                    '/home/bhaishahster/tasks_compare_conv_deep.txt',
                    'parameters of individual tasks')

# Running on : tmp_dir = '~/'
# On bhaishahster0:
# /home/bhaishahster/Downloads/data_breakdown/
flags.DEFINE_integer('taskid', 0, 'Task ID')

FLAGS = flags.FLAGS

rng = np.random


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

  model = line_split[1]
  if model == 'su':
    nsub = int(line_split[2])
    projection_type = line_split[3]
    lam_proj = float(line_split[4])
    frac_train = float(line_split[5])

  if model == 'conv':
    layers = line_split[2]
    frac_train = float(line_split[3])

  mask = (total_mask_log[cell_idx, :].sum(0) != 0)
  mask_matrix = np.reshape(mask != 0, [40, 80])

  # make mask bigger - add one row one left/right
  r, c = np.where(mask_matrix)
  mask_matrix[r.min()-1: r.max()+1, c.min()-1:c.max()+1] = True
  neighbor_mat = su_model.get_neighbormat(mask_matrix, nbd=1)
  mask = np.ndarray.flatten(mask_matrix)

  stimulus_2d = np.reshape(stimulus, [-1, 40, 80])
  stim_use_2d = stimulus_2d[:, r.min()-1: r.max()+1, c.min()-1:c.max()+1]
  stim_use = stimulus[:, mask]
  resp_use = responses[:, cell_idx]

  print('Prepared data')

  # get last 10% as test data
  np.random.seed(23)

  frac_test = 0.1
  tms_test = np.arange(np.floor(stim_use.shape[0] * (1 - frac_test)),
                       1*np.floor(stim_use.shape[0])).astype(np.int)

  # Random partitions
  n_partitions = 10
  tms_train_validate = np.arange(0, np.floor(stim_use.shape[0] *
                                             (1 - frac_test))).astype(np.int)

  frac_validate = 0.1
  # 'frac_train' needs to be < 0.9

  partitions = []
  for _ in range(n_partitions):
    perm = np.random.permutation(tms_train_validate)
    tms_train = perm[0: np.floor(frac_train * perm.shape[0])]
    tms_validate = perm[np.floor((1 - frac_validate) *
                                 perm.shape[0]): perm.shape[0]]

    partitions += [{'tms_train': tms_train,
                    'tms_validate': tms_validate,
                    'tms_test': tms_test}]

  ipartition = 0
  print('Made partitions')

  # Do fitting
  # tms_train = np.arange(0, np.floor(stim_use.shape[0] * 0.8)).astype(np.int)
  # tms_test = np.arange(np.floor(stim_use.shape[0] * 0.8),
  #                       1 * np.floor(stim_use.shape[0] * 0.9)).astype(np.int)

  ss = '_'.join([str(cells[ic]) for ic in cell_idx])

  if model == 'su':
    save_filename = os.path.join(FLAGS.save_path,
                                 'Cell_%s_su_nsub_%d_%s_%.3f_frac_train'
                                 '_%.4f_jnt.pkl' %
                                 (ss, nsub, projection_type,
                                  lam_proj, frac_train))

    save_filename_partial = os.path.join(FLAGS.save_path_partial,
                                         'Cell_%s_su_nsub_%d_%s_%.3f_frac_train'
                                         '_%.4f_jnt.pkl' %
                                         (ss, nsub, projection_type,
                                          lam_proj, frac_train))
    if not gfile.Exists(save_filename):
      print('Fitting started for SU')
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

      print('Fitting done')
      save_dict = {'K': k, 'b': b,
                   'lam_log': lam_log, 'lam_log_test': lam_log_test,
                   'fitting_phase': fitting_phase, 'fit_params': fit_params}
      pickle.dump(save_dict, gfile.Open(save_filename, 'w'))
      print('Saved results')

  if model == 'conv':
    save_filename = os.path.join(FLAGS.save_path,
                                 'Cell_%s_conv_layers_%s_frac_train_%.4f.pkl' %
                                 (ss, layers, frac_train))

    if not gfile.Exists(save_filename):
      print('Fitting started for CONV')

      op = conv_model.convolutional(stim_use_2d, np.squeeze(resp_use),
                                    partitions[ipartition]['tms_train'],
                                    partitions[ipartition]['tms_validate'],
                                    layers,
                                    lr=0.1, num_steps_max=100000,
                                    eps=1e-9)

      loss_train_log, loss_test_log, model_params = op

      print('Convolutional model fitting done')
      save_dict = {'lam_log': loss_train_log,
                   'lam_log_test': loss_test_log,
                   'model_params': model_params}
      pickle.dump(save_dict, gfile.Open(save_filename, 'w'))
      print('Saved results')


if __name__ == '__main__':
  app.run(main)
