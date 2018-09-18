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
"""Fit subunits with localized sparsity prior."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Import module
import scipy as sp
import numpy as np , h5py,numpy
import matplotlib.pyplot as plt
import matplotlib
import time
rng = np.random
import pickle
import copy
from tensorflow.python.platform import gfile
import os.path
from retina.response_model.python.ASM.op_nl import jnt_model

flags.DEFINE_string('src_dir',
                    '/home/bhaishahster/',
                    'temporary folder on machine for better I/O')

flags.DEFINE_string('tmp_dir',
                    '/home/bhaishahster/',
                    'temporary folder on machine for better I/O')

flags.DEFINE_string('save_path', '/home/bhaishahster/su_fits_pop/',
                    'where to store results')

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
  file=h5py.File(dst, 'r')

  # Load Masked movie
  data = file.get('maskedMovdd')
  stimulus = np.array(data)
  # load cell response
  cells = file.get('cells')

  ttf_log = file.get('ttf_log')
  ttf_avg = file.get('ttf_avg')

  # Load spike Response of cells
  data = file.get('Y')
  responses = np.array(data)

  # get mask
  total_mask_log=file.get('totalMaskAccept_log')

  print('Got data')

  # get cell and mask
  nsub_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  if FLAGS.taskid < 107 * len(nsub_list):
    cell_idx = [np.int(np.floor(FLAGS.taskid / len(nsub_list)))]
    cellid = cells[np.int(np.floor(FLAGS.taskid / len(nsub_list)))]
    Nsub = nsub_list[FLAGS.taskid % len(nsub_list)]
    partition_list = np.arange(10)

  elif FLAGS.taskid < 107 * len(nsub_list) + 37 * 10:
    cell_idx = [39, 42, 44, 45] #[np.int(FLAGS.taskid)]
    cellid = cells[cell_idx]
    cellid = np.squeeze(cellid)
    task_id_effective = FLAGS.taskid - 107 * len(nsub_list)
    partition_list = [task_id_effective % 10]
    nsub_list_pop = np.arange(4, 41)
    Nsub = nsub_list_pop[np.int(np.floor(task_id_effective /10))]

  elif FLAGS.taskid < 107 * len(nsub_list) + 37 * 10 + 19 * 10:
    cell_idx = [39, 42] #[np.int(FLAGS.taskid)]
    cellid = cells[cell_idx]
    cellid = np.squeeze(cellid)
    task_id_effective = FLAGS.taskid - 107 * len(nsub_list) - 37 * 10
    partition_list = [task_id_effective % 10]
    nsub_list_pop = np.arange(2, 21)
    Nsub = nsub_list_pop[np.int(np.floor(task_id_effective /10))]

  elif FLAGS.taskid < 107 * len(nsub_list) + 37 * 10 + 19 * 10 + 19 * 10:
    cell_idx = [44, 45] #[np.int(FLAGS.taskid)]
    cellid = cells[cell_idx]
    cellid = np.squeeze(cellid)
    task_id_effective = FLAGS.taskid - 107 * len(nsub_list) - 37 * 10 - 19 * 10
    partition_list = [task_id_effective % 10]
    nsub_list_pop = np.arange(2, 21)
    Nsub = nsub_list_pop[np.int(np.floor(task_id_effective /10))]

  print(cell_idx)
  print(Nsub)

  mask = (total_mask_log[cell_idx,:].sum(0) != 0)
  mask_matrix = np.reshape(mask!=0, [40,80])

  # make mask bigger - add one row one left/right
  r, c = np.where(mask_matrix)
  mask_matrix[r.min()-1: r.max()+1, c.min()-1:c.max()+1] = True
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
  tms_train_validate = np.arange(0, np.floor(stim_use.shape[0]*(1 - frac_test))).astype(np.int)

  frac_validate = 0.1

  partitions = []
  for ipartition in range(n_partitions):
    perm = np.random.permutation(tms_train_validate)
    tms_train = perm[0: np.floor((1 - frac_validate) * perm.shape[0])]
    tms_validate = perm[np.floor((1 - frac_validate) * perm.shape[0]): perm.shape[0]]

    partitions += [{'tms_train': tms_train,
                    'tms_validate': tms_validate,
                    'tms_test': tms_test}]

  print('Made partitions')

  # Do fitting
  # tms_train = np.arange(0, np.floor(stim_use.shape[0] * 0.8)).astype(np.int)
  # tms_test = np.arange(np.floor(stim_use.shape[0] * 0.8),
  #                       1 * np.floor(stim_use.shape[0] * 0.9)).astype(np.int)

  for ipartition in partition_list:
      print(cell_idx, cellid, Nsub)
      
      ss = '_'.join([str(ic) for ic in cellid])

      save_filename = os.path.join(FLAGS.save_path,
                                   'Cell_%s_nsub_%d_part_%d_jnt.pkl' %
                                   (ss, Nsub, ipartition))
      if not gfile.Exists(save_filename):
        print('Fitting started')
        op = jnt_model.Flat_clustering_jnt(stim_use, resp_use, Nsub,
                                         partitions[ipartition]['tms_train'],
                                         partitions[ipartition]['tms_validate'],
                                         steps_max=10000, eps=1e-9)

        # op = jnt_model.Flat_clustering_jnt(stim_use, resp_use, Nsub,
        #                                   tms_train,
        #                                   tms_test,
        #                                   steps_max=10000, eps=1e-9)

        K, b, alpha, lam_log, lam_log_test, fitting_phase, fit_params  = op

        print('Fitting done')
        save_dict = {'K': K, 'b': b,
                      'lam_log': lam_log, 'lam_log_test': lam_log_test,
                       'fitting_phase': fitting_phase, 'fit_params': fit_params}
        pickle.dump(save_dict,  gfile.Open(save_filename, 'w' ))
        print('Saved results')


if __name__ == '__main__':
  app.run(main)
