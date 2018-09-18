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
"""Collect responses across multiple retina for same stimulus."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import h5py
import numpy as np
import scipy.io as sio
import tensorflow as tf
import skimage.transform
from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS


def copy_locally(src, dst):
  """Copy data locally if not already there."""

  # Make the directory if it does not already exist.
  if not gfile.IsDirectory(dst):
    gfile.MkDir(dst)

  files = gfile.ListDirectory(src)
  for ifile in files:
    dst_file = os.path.join(dst, ifile)
    src_file = os.path.join(src, ifile)

    if not gfile.Exists(dst_file):
      gfile.Copy(src_file, dst_file)
      tf.logging.info('Copied %s' % src_file)
    else:
      tf.logging.info('File exists %s' % dst_file)

  tf.logging.info('File copied to/exists at local destination')


def get_stimulus_response(src_dir, src_dataset, stim_id, boundary=0,
                          if_get_stim=True):
  """Get stimulus-response data for all datasets.

  Args :
    src_dir : Location of all joint embedding datasets.
    src_dataset : Dataset corresponding of a specific stimulus.
    stim_id : string ID of the stimulus.
    boundary : Remove cells within a boundary to the edges.
    if_get_stim : If False, do not load stimulus

  Returns :
    stimulus : Stimulus matrix (Time x dimx x dimy).
    responses : Discretized cell responses (Time x n_cells).
    dimx : X dimension of stimulus.
    dimy : Y dimension of stimulus.
    num_cell_types : number of cell types.
  """

  # Copy data locally.
  # Since gfile does not support reading of large files directly from CNS,
  # we need to copy the data locally first.
  src = os.path.join(src_dir, src_dataset)
  if not gfile.IsDirectory(FLAGS.tmp_dir):
    gfile.MkDir(FLAGS.tmp_dir)
  dst = os.path.join(FLAGS.tmp_dir, src_dataset)
  print('Source %s' % src)
  print('Destination %s' % dst)
  copy_locally(src, dst)

  # Load stimulus-response data.
  if if_get_stim:
    data = h5py.File(os.path.join(dst, 'stimulus.mat'))
    stimulus = np.array(data.get('stimulus'))

    # Make dynamic range of stimuli from -0.5 to 0.5
    stim_min = np.min(stimulus)
    stim_max = np.max(stimulus)
    stimulus -= stim_min
    stimulus /= (stim_max - stim_min)
    stimulus -= 0.5

    # Make the stimuli mean 0
    stimulus -= np.mean(stimulus)

  else:
    stimulus = None

  # Load responses from multiple retinas.
  datasets_list = os.path.join(dst, 'datasets.txt')
  datasets = open(datasets_list, 'r').read()
  training_datasets = [line for line in datasets.splitlines()]

  num_cell_types = 2
  dimx_desired = 80
  dimy_desired = 40
  if stimulus is not None:
    dimx_actual = stimulus.shape[1]
    dimy_actual = stimulus.shape[2]
  else:
    stix_sz = np.int(src_dataset.split('-')[1])
    dimx_actual = np.int(640 / stix_sz)
    dimy_actual = np.int(320 / stix_sz)

  responses = []
  for idata in training_datasets:
    print(idata)
    data_file = os.path.join(dst, idata)
    data = sio.loadmat(data_file)
    data.update({'stimulus_key': stim_id})
    process_dataset(data, dimx_desired, dimy_desired, dimx_actual, dimy_actual,
                    num_cell_types, boundary=boundary)
    data.update({'piece': idata})
    responses += [data]

  if FLAGS.minimize_disk_usage:
    gfile.DeleteRecursively(dst)

  return stimulus, responses, dimx_desired, dimy_desired, num_cell_types


def process_dataset(iresp, dimx_desired, dimy_desired, dimx_actual, dimy_actual,
                    num_cell_types, boundary=0):
  """Clean data and compute auxillary properties of the responses.

  Args :
    iresp : Discretized cell response of one population (Time x n_cells).
    dimx_desired : Desired X dimension of stimulus.
    dimy_desired : Desired Y dimension of stimulus.
    dimx_actual : Actual X dimension of stimulus.
    dimy_actual : Actual Y dimension of stimulus.
    num_cell_types : number of cell types.
    boundary : Remove cells within a boundary to edges.

  Returns :
    iresp : Responses with added auxillary properties.
  """

  # Scale centers from 'actual' dimensions to 'desired'.
  iresp['centers'][:, 0] = (dimx_desired *
                            np.double(iresp['centers'][:, 0]) / dimx_actual)
  iresp['centers'][:, 1] = (dimy_desired *
                            np.double(iresp['centers'][:, 1]) / dimy_actual)

  iresp['dimx_initial'] = dimx_actual
  iresp['dimy_initial'] = dimy_actual
  iresp['dimx_final'] = dimx_desired
  iresp['dimy_final'] = dimy_desired

  # Remove cells with RFs outide the visual space.
  valid_cells0 = np.logical_and(iresp['centers'][:, 0] <= dimx_desired - boundary,
                                iresp['centers'][:, 1] <= dimy_desired - boundary)
  valid_cells1 = np.logical_and(iresp['centers'][:, 0] > boundary,
                                iresp['centers'][:, 1] > boundary)
  valid_cells = np.logical_and(valid_cells0, valid_cells1)

  iresp.update({'valid_cells': valid_cells})

  # Remove invalid cells.
  iresp['centers'] = iresp['centers'][valid_cells, :]

  try:
    iresp['sta_params'] = iresp['sta_params'][valid_cells, :]
  except KeyError:
    print('No STA params')

  try:
    iresp['responses'] = iresp['responses'][:, valid_cells]
    mean_resp = np.mean(iresp['responses'], 0)
    iresp.update({'mean_firing_rate': mean_resp})
  except KeyError:
    print('No responses')

  try:
    iresp['repeats'] = iresp['repeats'][:, :, valid_cells]
    mean_resp = np.mean(np.mean(iresp['repeats'], 0), 0)
    iresp.update({'mean_firing_rate': mean_resp})
  except KeyError:
    print('No repeats')

  try:
    iresp['cellID_list'] = iresp['cellID_list'][:, valid_cells]
  except KeyError:
    print('No cell ID list')

  iresp['cell_type'] = iresp['cell_type'][:, valid_cells]

  # find mosaic separation for different cell types
  iresp['dist_nn_cell_type'] = get_mosaic_distances(iresp['centers'],
                                                    iresp['cell_type'])

  print('Valid cells: %d/%d' % (np.sum(valid_cells), valid_cells.shape[0]))

  # Compute mean firing rate for cells.
  n_cells = np.squeeze(iresp['centers']).shape[0]

  # Do embedding of centers on a grid.
  _, _, map_cell_grid, mask_cells = give_cell_grid(iresp['centers'],
                                                   dimx=dimx_desired,
                                                   dimy=dimy_desired,
                                                   resolution=1)
  iresp.update({'map_cell_grid': map_cell_grid})
  iresp.update({'mask_cells': mask_cells})

  # Encode cell type as 1-hot vector.
  ctype_1hot = np.zeros((n_cells, num_cell_types))
  for icell_type in np.arange(1, num_cell_types+1):
    ctype_1hot[:, icell_type-1] = np.double(iresp['cell_type'] == icell_type)

  iresp.update({'ctype_1hot': ctype_1hot})

  # get EIs
  iresp['ei_image'] = iresp['ei_image'][valid_cells, :, :]

def give_cell_grid(centers, resolution, dimx=80, dimy=40, mask_distance=6):
  """Embeds each RF center on a discrete grid.

  Args:
    centers: center location of cells (n_cells x 2).
    resolution: Float specifying the resolution of grid.
    dimx : X dimension of grid.
    dimy : Y dimension of grid.
    mask_distance : Distance of pixel from center to be included in a cell's
                      receptive field mask.

  Returns:
    centers_grid : Discretized centers (n_cells x 2).
    grid_size : dimensions of the grid (2D integer tuple).
    map_cell_grid : mapping between cells to grid (grid_x x grid_y x n_cells)
    mask_cells : Mask of receptive field for each cell
                  (gird_x x grid_y x n_cells)
  """

  n_cells = centers.shape[0]
  centers_grid = np.floor(centers - 1 / resolution).astype(np.int)
  # subtract 1 because matlab indexing starts from 1.

  grid_size = [dimx, dimy]

  # map_cell_grid is location of each cell to grid point
  map_cell_grid = np.zeros((grid_size[0], grid_size[1], n_cells))
  for icell in range(n_cells):
    map_cell_grid[centers_grid[icell, 0], centers_grid[icell, 1], icell] = 1

  # get mask
  mask_cells = np.zeros((grid_size[0], grid_size[1], n_cells))
  yy, xx = np.meshgrid(np.arange(dimy), np.arange(dimx))
  for icell in range(n_cells):
    mask_cell = (np.sqrt((xx - centers_grid[icell, 0]) ** 2 +
                         (yy - centers_grid[icell, 1]) ** 2) <= mask_distance)
    mask_cells[:, :, icell] = mask_cell


  return centers_grid, grid_size, map_cell_grid, mask_cells


def get_mosaic_distances(center_locations_log, cell_type):
  """Use cell locations to get nearest neighbor distances for each cell type.

  Args:
    center_locations_log : Cell locations (numpy array = # cells x 2)
    cell_type : Cell types (1: OFF parasol, 2: ON parasol)
                  (numpy array of size # cells)

  Returns:
    dist_nn_cell_type : Dictionary {cell_type: nearest neighbor separation}.

  """
  cell_type = np.squeeze(cell_type)

  # Find NN distance for each cell
  dist_nn_cell_type = {}
  for icell_type in np.unique(cell_type):
    cells_selected = np.where(cell_type == icell_type)[0]
    dist_nn = []
    for icell in cells_selected:
      d_cell = []
      for jcell in cells_selected:
        if icell == jcell:
          continue
        d_cell += [np.sqrt(np.sum((center_locations_log[icell, :] -
                                   center_locations_log[jcell, :]) ** 2))]
      dist_nn += [np.min(d_cell)]
    dist_nn_cell_type.update({icell_type: np.mean(dist_nn)})

  return dist_nn_cell_type
