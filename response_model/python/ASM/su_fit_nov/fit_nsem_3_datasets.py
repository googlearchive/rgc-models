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
r"""Fit subunits with localized sparsity prior."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os.path
import pickle
from absl import app
from absl import flags
import numpy as np
import scipy.io as sio
from tensorflow.python.platform import gfile
from retina.response_model.python.ASM.su_fit_nov import su_model


flags.DEFINE_string('src_dir',
                    '/home/bhaishahster/NSEM_process/',
                    'temporary folder on machine for better I/O')

flags.DEFINE_string('tmp_dir',
                    '/home/bhaishahster/Downloads/'
                    'NSEM_process/NSEM_preprocess',
                    'temporary folder on machine for better I/O')

flags.DEFINE_string('save_path', '/home/bhaishahster/'
                    'su_fits_nsem_3_datasets/',
                    'where to store results')

flags.DEFINE_string('save_path_partial', '/home/bhaishahster/'
                    'su_fits_nsem_3_datasets_partial/',
                    'where to store results')

flags.DEFINE_string('task_params_file',
                    '/home/bhaishahster/tasks_nsem_3_datasets.txt',
                    'parameters of individual tasks')

flags.DEFINE_integer('taskid', 0, 'Task ID')

FLAGS = flags.FLAGS

rng = np.random


def main(argv):

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
  cell_idx = int(cell_idx[0])
  file_list = gfile.ListDirectory(FLAGS.src_dir)
  cell_file = file_list[cell_idx]
  print('Cell file %s' % cell_file)

  nsub = int(line_split[1])
  projection_type = line_split[2]
  lam_proj = float(line_split[3])

  # copy data
  dst = os.path.join(FLAGS.tmp_dir, cell_file)

  if not gfile.Exists(dst):
    print('Started Copy')
    src = os.path.join(FLAGS.src_dir, cell_file)
    if not gfile.IsDirectory(FLAGS.tmp_dir):
      gfile.MkDir(FLAGS.tmp_dir)

    gfile.Copy(src, dst)
    print('File copied to destination')

  else:
    print('File exists')

  # load stimulus, response data
  try:
    data = sio.loadmat(dst)
    trainMov_filterNSEM = data['trainMov_filterNSEM']
    testMov_filterNSEM = data['testMov_filterNSEM']
    trainSpksNSEM = data['trainSpksNSEM']
    testSpksNSEM = data['testSpksNSEM']
    mask = data['mask']

    neighbor_mat = su_model.get_neighbormat(mask, nbd=1)

    trainMov_filterWN = data['trainMov_filterWN']
    testMov_filterWN = data['testMov_filterWN']
    trainSpksWN = data['trainSpksWN']
    testSpksWN = data['testSpksWN']

    # get NSEM stimulus and resposne
    stimulus_WN = np.array(trainMov_filterWN.transpose(), dtype='float32')
    response_WN = np.array(np.squeeze(trainSpksWN), dtype='float32')

    stimulus_NSEM = np.array(trainMov_filterNSEM.transpose(), dtype='float32')
    response_NSEM = np.array(np.squeeze(trainSpksNSEM), dtype='float32')
    print('Prepared data')
    # Do fitting

    # set random seed.
    np.random.seed(23)

    print('Made partitions')

    # Do fitting
    # WN data
    ifrac = 0.8
    tms_train_WN = np.arange(0, np.floor(stimulus_WN.shape[0] *
                                         ifrac)).astype(np.int)
    tms_test_WN = np.arange(np.floor(stimulus_WN.shape[0] * ifrac),
                            1 * np.floor(stimulus_WN.shape[0] *
                                         1)).astype(np.int)

    # NSEM data
    ifrac = 0.8
    tms_train_NSEM = np.arange(0, np.floor(stimulus_NSEM.shape[0] *
                                           ifrac)).astype(np.int)
    tms_test_NSEM = np.arange(np.floor(stimulus_NSEM.shape[0] * ifrac),
                              1 * np.floor(stimulus_NSEM.shape[0] *
                                           1)).astype(np.int)

    # Give filename
    ss = str(cell_idx)

    save_filename = os.path.join(FLAGS.save_path,
                                 'Cell_%s_nsub_%d_%s_%.3f_jnt.pkl' %
                                 (ss, nsub, projection_type,
                                  lam_proj))

    save_filename_partial = os.path.join(FLAGS.save_path_partial,
                                         'Cell_%s_nsub_%d_%s_%.3f_jnt.pkl' %
                                         (ss, nsub, projection_type,
                                          lam_proj))

    ## Do fitting
    if  not gfile.Exists(save_filename):

      # Fit SU on WN
      print('Fitting started on WN')
      op = su_model.Flat_clustering_jnt(stimulus_WN,
                                        np.expand_dims(response_WN, 1), nsub,
                                        tms_train_WN,
                                        tms_test_WN,
                                        steps_max=10000, eps=1e-9,
                                        projection_type=projection_type,
                                        neighbor_mat=neighbor_mat,
                                        lam_proj=lam_proj, eps_proj=0.01,
                                        save_filename_partial=
                                        save_filename_partial,
                                        fitting_phases=[1])

      _, _, alpha, lam_log_wn, lam_log_test_wn, fitting_phase, fit_params_wn = op

      print('WN fit done')

      # Fit on NSEM
      op = su_model.fit_scales(stimulus_NSEM[tms_train_NSEM, :],
                               np.expand_dims(response_NSEM[tms_train_NSEM], 1),
                               stimulus_NSEM[tms_test_NSEM, :],
                               np.expand_dims(response_NSEM[tms_test_NSEM], 1),
                               Ns=nsub,
                               K=fit_params_wn[0][0], b=fit_params_wn[0][1],
                               params=fit_params_wn[0][2], lr=0.01, eps=1e-9)

      k_nsem, b_nsem, nl_params_nsem, lam_log_nsem, lam_log_test_nsem = op

      # Collect results and save
      fit_params = fit_params_wn + [[k_nsem, b_nsem, nl_params_nsem]]
      lam_log = [lam_log_wn, np.array(lam_log_nsem)]
      lam_log_test = [lam_log_test_wn, np.array(lam_log_test_nsem)]

      save_dict = {'lam_log': lam_log, 'lam_log_test': lam_log_test,
                   'fit_params': fit_params, 'mask': mask}
      pickle.dump(save_dict, gfile.Open(save_filename, 'w'))
      print('Saved results')

  except:
    print('Error')


if __name__ == '__main__':
  app.run(main)
