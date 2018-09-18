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
import os.path
import pickle
from absl import app
from absl import flags
import h5py
import numpy as np
from tensorflow.python.platform import gfile
from retina.response_model.python.ASM.su_fit_nov import su_model


flags.DEFINE_string('src_dir', '/home/bhaishahster/pc2005_08_08_1',
                    'Where is the cell')

flags.DEFINE_integer('taskid', 0,
                     'Task Id')

flags.DEFINE_string('tmp_dir',
                    '/home/bhaishahster/'
                    'Downloads/pc2005_08_08_1',
                    'temporary folder on machine for better I/O')

flags.DEFINE_string('save_path', '/home/bhaishahster/su_fine_pop_jan/',
                    'where to store results')

flags.DEFINE_string('save_path_partial',
                    '/home/bhaishahster/su_fine_pop_jan_partial/',
                    'where to store intermediate fits - incase fitting breaks')

flags.DEFINE_string('save_suffix', '_jan',
                    'save suffix')

flags.DEFINE_string('task_params_file',
                    '/home/bhaishahster/pc2005_08_08_1/'
                    'tasks_fine_population.txt',
                    'parameters of individual tasks')


FLAGS = flags.FLAGS

rng = np.random


def get_su_nsub(stimulus, response, mask_matrix, cell_string, nsub,
                projection_type, lam_proj, ipartition):
  """Get 'nsub' subunits."""

  np.random.seed(95)  # 23 for _jnt.pkl, 46 for _jnt_2.pkl, 93 for _nov, _jan

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
  tms_train_validate = np.arange(0, np.floor(stimulus.shape[0] *
                                             (1 - frac_test))).astype(np.int)

  frac_validate = 0.1

  partitions = []
  for _ in range(n_partitions):
    perm = np.random.permutation(tms_train_validate)
    tms_train = perm[0: np.floor((1 - frac_validate) * perm.shape[0])]
    tms_validate = perm[np.floor((1 - frac_validate) *
                                 perm.shape[0]): perm.shape[0]]

    partitions += [{'tms_train': tms_train,
                    'tms_validate': tms_validate,
                    'tms_test': tms_test}]

  print('Made partitions')

  # do fitting for different lambdas
  # from IPython import embed; embed()
  neighbor_mat = su_model.get_neighbormat(mask_matrix, nbd=1)
  save_name = os.path.join(FLAGS.save_path,
                           'Cell_%s_nsub_%d_%s_%.6f_part_%d_%s.pkl' %
                           (cell_string, nsub, projection_type, lam_proj,
                            ipartition, FLAGS.save_suffix))

  save_filename_partial = os.path.join(FLAGS.save_path_partial,
                                       'Cell_%s_nsub_%d_%s_%.6f_part'
                                       '_%d_%s.pkl' %
                                       (cell_string, nsub, projection_type,
                                        lam_proj, ipartition,
                                        FLAGS.save_suffix))

  if not gfile.Exists(save_name):
    print(cell_string, nsub, projection_type, lam_proj, ipartition)
    op = su_model.Flat_clustering_jnt(stimulus, response, nsub,
                                      partitions[ipartition]['tms_train'],
                                      partitions[ipartition]['tms_validate'],
                                      steps_max=10000, eps=1e-9,
                                      projection_type=projection_type,
                                      neighbor_mat=neighbor_mat,
                                      lam_proj=lam_proj, eps_proj=0.01,
                                      save_filename_partial=
                                      save_filename_partial)

    k_f, b_f, _, loss_log_f, loss_log_test_f, fitting_phase_f, fit_params_f = op

    print('Fitting done')
    save_dict = {'K': k_f, 'b': b_f,
                 'loss_log': loss_log_f,
                 'loss_log_test': loss_log_test_f,
                 'fitting_phase': fitting_phase_f,
                 'fit_params': fit_params_f}

    pickle.dump(save_dict, gfile.Open(save_name, 'w'))
    print('Saved results')


def main(argv):

  # parse task params
  # read line corresponding to task
  with gfile.Open(FLAGS.task_params_file, 'r') as f:
    for _ in range(FLAGS.taskid + 1):
      line = f.readline()

  print(line)

  # get task parameters by parsing the line.
  line_split = line.split(';')
  cells = gfile.ListDirectory(FLAGS.src_dir)

  cell_idx = line_split[0]
  cell_idx = cell_idx[1:-1].split(',')

  nsub = int(line_split[1])
  projection_type = line_split[2]
  lam_proj = float(line_split[3])
  ipartition = int(line_split[4][:-1])

  # Copy data for all the data
  cell_str_final = ''
  dst_log = []
  for icell in cell_idx:
    icell = int(icell)
    cell_string = cells[icell]
    cell_str_final += cell_string

    # copy data for the corresponding task
    dst = os.path.join(FLAGS.tmp_dir, cell_string)

    dst_log += [dst]
    if not gfile.Exists(dst):
      print('Started Copy')
      src = os.path.join(FLAGS.src_dir, cell_string)
      if not gfile.IsDirectory(FLAGS.tmp_dir):
        gfile.MkDir(FLAGS.tmp_dir)

      gfile.Copy(src, dst)
      print('File %s copied to destination' % cell_string)

    else:
      print('File %s exists' % cell_string)

  # Load data for different cells
  stim_log = []
  resp_log = []
  mask_matrix_log = []
  for dst in dst_log:
    print('Loading %s' % dst)
    data = h5py.File(dst)
    stimulus = np.array(data.get('stimulus'))
    stimulus = stimulus[:-1, :]  # drop the last frame so that it's
                                # the same size as the binned spike train

    response = np.squeeze(np.array(data.get('response')))
    response = np.expand_dims(response, 1)
    mask_matrix = np.array(data.get('mask'))

    stim_log += [stimulus]
    resp_log += [response]
    mask_matrix_log += [mask_matrix]

  # Prepare for fitting across multiple cells
  # Get total mask
  mask_matrix_pop = np.array(mask_matrix_log).sum(0) > 0

  # Get total response.
  resp_len = np.min([resp_log[icell].shape[0] for icell in range(4)])
  response_pop = np.zeros((resp_len, len(resp_log)))
  for icell in range(len(resp_log)):
    response_pop[:, icell] = resp_log[icell][:resp_len, 0]

  # Get total stimulus.
  stimulus_pop = np.zeros((resp_len, mask_matrix_pop.sum()))
  # Find non-zero locations for each mask element
  nnz_log = [np.where(imask > 0) for imask in mask_matrix_log]
  nnz_pop = np.where(mask_matrix_pop > 0)

  for ipix in range(mask_matrix_pop.sum()):
    print(ipix)
    r = nnz_pop[0][ipix]
    c = nnz_pop[1][ipix]

    stim_pix = np.zeros(resp_len)
    nc = 0
    for icell in range(len(nnz_log)):
      pix_cell_bool = np.logical_and(nnz_log[icell][0] == r,
                                     nnz_log[icell][1] == c)
      if pix_cell_bool.sum() > 0:
        pix_cell = np.where(pix_cell_bool > 0)[0][0]
        stim_pix += stim_log[icell][:resp_len, pix_cell]
        nc += 1

    if nc == 0:
      print('Error')

    stim_pix = stim_pix / nc
    stimulus_pop[:, ipix] = stim_pix

  # Fit with a given number of subunits
  print('Starting fitting')
  get_su_nsub(stimulus_pop, response_pop, mask_matrix_pop, cell_str_final,
              nsub, projection_type, lam_proj, ipartition)


if __name__ == '__main__':
  app.run(main)
