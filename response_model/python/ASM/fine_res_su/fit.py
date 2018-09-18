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
import retina.response_model.python.ASM.fine_res_su.sparse_model as sp

flags.DEFINE_string('src_dir', '/home/bhaishahster/pc2005_08_08_1',
                    'Where is the cell')  # data copied to il

flags.DEFINE_integer('taskid', 0,
                     'Task Id')

flags.DEFINE_string('tmp_dir',
                    '/home/bhaishahster/'
                    'Downloads/pc2005_08_08_1',
                    'temporary folder on machine for better I/O')

flags.DEFINE_string('save_path', '/home/bhaishahster/su_fine/fits/',
                    'where to store results')


FLAGS = flags.FLAGS

def get_su_nsub(stimulus, response, mask_matrix, n_sub, cell_string):
  """Get 'n_sub' subunits."""
  np.random.seed(46) # 23 for _jnt.pkl, 46 for _jnt_2.pkl

  # Get a few (5) training, testing, validation partitions

  # continuous partitions
  # ifrac = 0.8
  # tms_train = np.arange(0, np.floor(stimulus.shape[0]*ifrac)).astype(np.int)

  # Random partitions
  # get last 10% as test data
  frac_test = 0.1
  tms_test = np.arange(np.floor(stimulus.shape[0]*(1 - frac_test)),
                       1*np.floor(stimulus.shape[0])).astype(np.int)

  # Random partitions
  n_partitions = 10
  tms_train_validate = np.arange(0, np.floor(stimulus.shape[0]*(1 - frac_test))).astype(np.int)

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

  # do fitting for different lambdas
  # from IPython import embed; embed()
  neighbor_mat = sp.get_neighbormat(mask_matrix, nbd=1)
  for ipartition in range(n_partitions):

    # for lam_l1 in [10, 20, 50, 25, 0.01, 0.001, 0.0001,
    #               0.05, 0, 0.1, 0.25, 0.5, 0.75, 1, 3]:
    for lam_l1 in np.random.permutation(np.arange(0, 2, 0.05)):
      save_name = os.path.join(FLAGS.save_path,
                                      'Cell_%s_nsub_%d_lam_%.6f_part_%d_jnt_2.pkl' %
                                      (cell_string, n_sub, lam_l1, ipartition)); # _faster_2 before, wherer max_iter was 1000.
      if not gfile.Exists(save_name):
        print('Starting su: %d, lam l1: %.6f' %(n_sub, lam_l1))
        op = sp.Flat_clustering_sparse2(stimulus, response, n_sub,
                                        partitions[ipartition]['tms_train'],
                                        partitions[ipartition]['tms_validate'],
                                        lam_l1=lam_l1, eps=0.01,
                                        neighbor_mat=neighbor_mat,
                                        stop_th=1e-7, max_iter=100000)
        K_f, b_f, alpha_f, loss_log_f, loss_log_test_f, grad_K_log_f = op
        print('Fitting done')
        save_dict = {'K': K_f, 'b': b_f,
                     'loss_log': loss_log_f,
                     'loss_log_test': loss_log_test_f, 'lam_l1': lam_l1}
        pickle.dump(save_dict, gfile.Open(save_name, 'w' ))
        print('Saved results')


def main(argv):

  cells = gfile.ListDirectory(FLAGS.src_dir)
  nsub_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
               14, 15, 16, 17, 18, 19, 20]
  cell_id = np.floor(FLAGS.taskid / len(nsub_list)).astype(np.int)
  cell_string = cells[cell_id]

  # copy data
  dst = os.path.join(FLAGS.tmp_dir, cell_string)

  if not gfile.Exists(dst):
    print('Started Copy')
    src = os.path.join(FLAGS.src_dir, cell_string)
    if not gfile.IsDirectory(FLAGS.tmp_dir):
      gfile.MkDir(FLAGS.tmp_dir)

    gfile.Copy(src, dst)
    print('File copied to destination')

  else:
    print('File exists')

  # Load data
  data = h5py.File(dst)
  stimulus = np.array(data.get('stimulus'))
  stimulus = stimulus[:-1,:] # drop the last frame so that it's the same size as the binned spike train

  response = np.squeeze(np.array(data.get('response')))
  mask_matrix = np.array(data.get('mask'))

  # Fit with a given number of subunits

  nsub = nsub_list[FLAGS.taskid % len(nsub_list)]
  print('Cell: %s, Nsub: %d' % (cell_string, nsub))
  get_su_nsub(stimulus, response, mask_matrix, nsub, cell_string)


if __name__ == '__main__':
  app.run(main)
