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
import os
import scipy.io as sio
from retina.response_model.python.ASM.op_nl import jnt_model

flags.DEFINE_string('src_dir',
                    '/home/bhaishahster/nsem_data/pc2012_08_09_3',
                    'temporary folder on machine for better I/O')

flags.DEFINE_string('tmp_dir',
                    '/home/bhaishahster/Downloads/pc2012_08_09_3',
                    'temporary folder on machine for better I/O')

flags.DEFINE_string('save_path', '/home/bhaishahster/su_fits/nsem/',
                    'where to store results')

flags.DEFINE_integer('taskid', 0, 'Task ID')

FLAGS = flags.FLAGS

def main(argv):


  cell_idx = FLAGS.taskid
  file_list = gfile.ListDirectory(FLAGS.src_dir)
  cell_file = file_list[cell_idx]
  print('Cell file %s' % cell_file)

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
  data = sio.loadmat(dst)
  trainMov_filterNSEM = data['trainMov_filterNSEM']
  testMov_filterNSEM = data['testMov_filterNSEM']
  trainSpksNSEM = data['trainSpksNSEM']
  testSpksNSEM = data['testSpksNSEM']
  mask = data['mask']

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
  tms_train_WN = np.arange(0, np.floor(stimulus_WN.shape[0]*ifrac)).astype(np.int)
  tms_test_WN = np.arange(np.floor(stimulus_WN.shape[0]*ifrac), 1*np.floor(stimulus_WN.shape[0] * 1)).astype(np.int)

  # NSEM data
  ifrac = 0.8

  tms_train_NSEM = np.arange(0, np.floor(stimulus_NSEM.shape[0]*ifrac)).astype(np.int)
  tms_test_NSEM = np.arange(np.floor(stimulus_NSEM.shape[0]*ifrac), 1*np.floor(stimulus_NSEM.shape[0] * 1)).astype(np.int)

  '''
  eps = 1e-7
  for Nsub in [1, 2, 3, 4, 5, 7, 10]:
      print('Fitting started')

      # WN fit
      op = jnt_model.Flat_clustering(stimulus_WN, response_WN, Nsub, tms_train_WN, tms_test_WN,
                           steps_max=10000, eps=eps)
      K, b, alpha, lam_log, lam_log_test, fitting_phase, fit_params  = op
      WN_fit = {'K': K, 'b': b,
                'lam_log': lam_log, 'lam_log_test': lam_log_test}
      print('WN fit done')

      # NSEM fit
      # Just fit the scales
      # fit NL + b + Kscale
      K, b, params, loss_log, loss_log_test  = jnt_model.fit_scales(stimulus_NSEM[tms_train_NSEM, :], response_NSEM[tms_train_NSEM],
                                                          stimulus_NSEM[tms_test_NSEM, :], response_NSEM[tms_test_NSEM],
                                                          Ns=Nsub, K=WN_fit['K'], b=WN_fit['b'], params=[1.0, 0.0],
                                                          lr=0.001, eps=eps)
      NSEM_fit_scales = {'K': K, 'b': b, 'nl_params': params,
                         'lam_log': loss_log, 'lam_log_test': loss_log_test}
      print('NSEM scales fit')

      # Fit all params
      K, b, params, loss_log, loss_log_test  = jnt_model.fit_all(stimulus_NSEM[tms_train_NSEM, :], response_NSEM[tms_train_NSEM],
                                                       stimulus_NSEM[tms_test_NSEM, :], response_NSEM[tms_test_NSEM],
                                                       Ns=Nsub,
                                                       K=NSEM_fit_scales['K'], b=NSEM_fit_scales['b'],
                                                       train_phase=3,
                                                       params=NSEM_fit_scales['nl_params'],
                                                       lr=0.001, eps=eps)
      NSEM_fit_full = {'K': K, 'b': b, 'nl_params': params,
                       'lam_log': loss_log, 'lam_log_test': loss_log_test}
      print('NSEM all fit')

      save_dict = {'WN_fit': WN_fit,
                   'NSEM_fit_scales': NSEM_fit_scales,
                   'NSEM_fit_full': NSEM_fit_full}

      pickle.dump(save_dict,
                  gfile.Open(os.path.join(FLAGS.save_path,
                                    'Cell_%s_nsub_%d_suff_%d_jnt.pkl' %
                                    (cell_file, Nsub, 1)), 'w' ))
      print('Saved results')
  '''

  '''
  eps = 1e-7
  for Nsub in [1, 2, 3, 4, 5, 7, 10]:
      print('Fitting started')

      # Fit all params
      K = 2*rng.rand(stimulus_NSEM.shape[1], Nsub)-0.5
      b = 2*rng.rand(Nsub)-0.5

      K, b, params, loss_log, loss_log_test  = jnt_model.fit_all(stimulus_NSEM[tms_train_NSEM, :], response_NSEM[tms_train_NSEM],
                                                       stimulus_NSEM[tms_test_NSEM, :], response_NSEM[tms_test_NSEM],
                                                       Ns=Nsub,
                                                       K=K.astype(np.float32), b=b.astype(np.float32),
                                                       train_phase=3,
                                                       params=[1.0, 0.0],
                                                       lr=0.001, eps=eps)
      NSEM_fit_full = {'K': K, 'b': b, 'nl_params': params,
                       'lam_log': loss_log, 'lam_log_test': loss_log_test}
      print('NSEM all (random) fit')

      save_dict = {'NSEM_fit_full_random': NSEM_fit_full}

      pickle.dump(save_dict,
                  gfile.Open(os.path.join(FLAGS.save_path,
                                    'Cell_%s_nsub_%d_suff_%d_randomly_init.pkl' %
                                    (cell_file, Nsub, 1)), 'w' ))
      print('Saved results')
  '''

  eps = 1e-7
  for Nsub in [1, 2, 3, 4, 5, 7, 10]:
      print('Fitting started')

      # NSEM clustering fit
      op = jnt_model.Flat_clustering(stimulus_NSEM, response_NSEM, Nsub, tms_train_NSEM, tms_test_NSEM,
                                     steps_max=10000, eps=eps)
      K, b, alpha, lam_log, lam_log_test, fitting_phase, fit_params  = op
      NSEM_clustering = {'K': K, 'b': b,
                         'lam_log': lam_log, 'lam_log_test': lam_log_test}
      print('NSEM clustering fit')

      # NSEM fit
      # Just fit the scales
      # fit NL + b + Kscale
      K, b, params, loss_log, loss_log_test  = jnt_model.fit_scales(stimulus_NSEM[tms_train_NSEM, :], response_NSEM[tms_train_NSEM],
                                                          stimulus_NSEM[tms_test_NSEM, :], response_NSEM[tms_test_NSEM],
                                                          Ns=Nsub, K=NSEM_clustering['K'], b=NSEM_clustering['b'], params=[1.0, 0.0],
                                                          lr=0.001, eps=eps)
      NSEM_fit_scales = {'K': K, 'b': b, 'nl_params': params,
                         'lam_log': loss_log, 'lam_log_test': loss_log_test}
      print('NSEM scales fit')

      # Fit all params
      K, b, params, loss_log, loss_log_test  = jnt_model.fit_all(stimulus_NSEM[tms_train_NSEM, :], response_NSEM[tms_train_NSEM],
                                                       stimulus_NSEM[tms_test_NSEM, :], response_NSEM[tms_test_NSEM],
                                                       Ns=Nsub,
                                                       K=NSEM_fit_scales['K'], b=NSEM_fit_scales['b'],
                                                       train_phase=3,
                                                       params=NSEM_fit_scales['nl_params'],
                                                       lr=0.001, eps=eps)
      NSEM_fit_full = {'K': K, 'b': b, 'nl_params': params,
                       'lam_log': loss_log, 'lam_log_test': loss_log_test}
      print('NSEM all fit')

      save_dict = {'NSEM_clustering': NSEM_clustering,
                   'NSEM_fit_scales': NSEM_fit_scales,
                   'NSEM_fit_full': NSEM_fit_full}

      pickle.dump(save_dict,
                  gfile.Open(os.path.join(FLAGS.save_path,
                                    'Cell_%s_nsub_%d_suff_%d_NSEM_3_steps.pkl' %
                                    (cell_file, Nsub, 1)), 'w' ))
      print('Saved results')

if __name__ == '__main__':
  app.run(main)
