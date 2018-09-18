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
r"""Baseline stimulus-response embedding models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import pickle
import numpy as np
import tensorflow as tf
from absl import gfile


FLAGS = tf.app.flags.FLAGS


def get_neighbormat(mask_matrix, nbd=1):
  """Get neighbor mapping.

  For a rectangular nxm mask (1/0), give a neighborhood map M (nm x nm).
  where M(i, j) = 1 if pixel at i (in flattened mask) is neighbor of pixel at j.

  Args:
    mask_matrix: Boolean mask of active pixels (numpy array, n x m).
    nbd: Neighborhood to consider.

  Returns:
    neighbor_mat : neighborhood map as numpy array(nm x nm).
  """

  mask = np.ndarray.flatten(mask_matrix) > 0

  x = np.repeat(np.expand_dims(np.arange(mask_matrix.shape[0]), 1),
                mask_matrix.shape[1], 1)
  y = np.repeat(np.expand_dims(np.arange(mask_matrix.shape[1]), 0),
                mask_matrix.shape[0], 0)
  x = np.ndarray.flatten(x)
  y = np.ndarray.flatten(y)

  xx = np.expand_dims(x[mask], 1)
  yy = np.expand_dims(y[mask], 1)

  distance = (xx - xx.T)**2 + (yy - yy.T)**2
  neighbor_mat = np.double(distance <= nbd)

  return neighbor_mat


def project_spatially_locaized_l1(k_tf, dimx, dimy, n_cells,
                                  lnl1_reg=0.001, eps_neigh=0.01):
  """Project into constrain set of spatially localized L1 regularization.

  The locally normalized L1 regularize is :
    L(w) = sum_i (|w_i| / sum_j |w_j| + eps ), where j in neighborhood of i.
  The projection operation is : argmin( ||w - w0||^2 + lambda*(L(w))).
  This is just like soft-thresholding for L1 norm, but the threshold for each
    pixel determined by current values of neighbors of each weight.

  Currently supports only weights k_tf of form (dimx x dimy x n_cells).

  Args:
    k_tf : Tensor for weight. (dimx * dimy, n_cells).
    dimx : X dimension of mask.
    dimy : Y dimension of mask.
    n_cells : Number of cells.
    lnl1_reg : regularization value.
    eps_neigh : eps in loss.

  Returns :
    proj_k : Projected weights.
  """

  neighbor_mat = get_neighbormat(np.ones((dimx, dimy)))
  n_mat = tf.constant(neighbor_mat.astype(np.float32))
  wts_tf = 1 / (tf.matmul(n_mat,
                          tf.reshape(tf.abs(k_tf),
                                     [dimx * dimy, n_cells])) + eps_neigh)

  wts_tf_3d = tf.reshape(wts_tf, [dimx, dimy, n_cells])
  proj_k = tf.assign(k_tf, tf.nn.relu(k_tf - wts_tf_3d * lnl1_reg) -
                     tf.nn.relu(-k_tf - wts_tf_3d * lnl1_reg))

  return proj_k


def approximate_rfs_from_centers(center_locations, cell_type, firing_rates,
                                 dimx, dimy, n_cells):
  """Use cell locations to get approximate RF centers.

  Put gaussians at centers of cells with sigma = (dist_nn / 2) / 1.7.
  The signs of gaussians are negative for OFF cells and positive for ON cells.

  Args:
    center_locations : Cell locations (numpy array = X x Y x # cells)
    cell_type : Cell types (1: OFF parasol, 2: ON parasol)
                  (numpy array of size # cells)
    firing_rates : mean firing rate of cells (unused right now.)
                    (numpy array - # cells)
    dimx : X dimension of visual stimulus (float)
    dimy : Y dimension of visual stimulus (float)
    n_cells : number of cells (integer)

  Returns:
    k_gaussian : Gaussian placed at corresponding cell locations
                  (numpy array: dimx X dimy X n_cells).

  """

  # Get spatial filter for blind retina.
  # Find nearest neighbor distances from center_locations
  # Convert centers to (r, c) indices
  center_locations_log = []
  for icell in range(center_locations.shape[2]):
    r, c = np.where(center_locations[:, :, icell] > 0)
    center_locations_log += [[r.squeeze(), c.squeeze()]]
  center_locations_log = np.array(center_locations_log).squeeze()

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

  # place gaussians at corresponding locations
  center_locations_log = center_locations_log.astype(np.float32)
  k_gaussian = np.zeros((dimx, dimy, n_cells))
  for icell in range(n_cells):
    dist_nn = dist_nn_cell_type[cell_type[icell]]
    sigma = (dist_nn / 2) / 1.7
    print(center_locations_log[icell, :])
    for ix in np.arange(dimx):
      for iy in np.arange(dimy):
        g = np.exp(- ((ix - center_locations_log[icell, 0]) ** 2 +
                      (iy - center_locations_log[icell, 1]) ** 2) /
                   (2 * sigma ** 2))
        k_gaussian[ix, iy, icell] = g
  k_gaussian /= np.sqrt(np.sum(np.sum(k_gaussian**2, 0), 0))

  # Assign sign of RF based on cell type.
  sign_dict = {1: -1, 2: 1}
  sign_vector = [sign_dict[ct] for ct in cell_type]
  sign_vector = np.array(sign_vector)
  k_gaussian *= sign_vector

  # TODO(bhaishahster) : Scale spatial filters based on cell's mean firing rate.

  return k_gaussian


def linear_rank1_models(sr_model, sess, dimx, dimy, n_cells, center_locations, cell_masks,
                        firing_rates, cell_type, time_window=30):
  """Learn a linear rank 1 (space time separable) stimulus response metric.

  The metric d(s, r) = -r' F s g. with r=response, s = space x time,
    g is time filter and F is (# cells x space) is spatial filter.

  g is initialized with precomputed average time filter
      across multiple retinas.
  F is initialized with signed gaussians at the center locations
      such that it forms a mosaic.

  g, F are learned by minimzing distance between positive examples and
    all the negatives in a batch.

  Since we think F must be spatially localized, we project to
    minimize locally normalized L1 regularization at each step.


  Args:
    sr_model : The variant of linear model ('lin_rank1' or 'lin_rank1_blind')
    sess : Tensorflow session.
    dimx: X dimension of the stimulus.
    dimy: Y dimension of the stimulus.
    n_cells : Number of cells
    center_locations : Location of centers for cell in the
                        2D grid (Dimx X Dimy X n_cells).
    cell_masks : Masks for spatial filter for each cell (Dimx x Dimy x n_cells).
    firing_rates : Mean firing rate of different cells ( # cells)
    cell_type : Cell type of each cell. (# cells)
    time_window : Length of stimulus (in time).

  Returns:
    sr_graph : Container of the embedding parameters and losses.

  Raises:
    ValueError : If the model type is not supported.
  """

  # Embed stimulus.
  stim_tf = tf.placeholder(tf.float32,
                           shape=[None, dimx,
                                  dimy,
                                  time_window])  # batch x X x Y x time_window

  pos_resp = tf.placeholder(dtype=tf.float32,
                            shape=[None, n_cells, 1],
                            name='pos')

  neg_resp = tf.placeholder(dtype=tf.float32,
                            shape=[None, n_cells, 1],
                            name='neg')

  # n_cells x number of selected cells
  select_cells_mat = tf.placeholder(dtype=tf.float32, shape=[n_cells, None],
                                    name='selected_cells')

  # Declare spatial and temporal filter variables.
  k_init = approximate_rfs_from_centers(center_locations, cell_type,
                                        firing_rates, dimx, dimy, n_cells)
  ttf_data = pickle.load(gfile.Open(FLAGS.ttf_file, 'r'))
  ttf_init = ttf_data['ttf'].astype(np.float32)

  k_tf = tf.Variable(k_init.astype(np.float32))
  ttf_tf = tf.Variable(ttf_init.astype(np.float32))

  # Do filtering in space and time
  tfd = tf.expand_dims
  ttf_4d = tfd(tfd(tfd(ttf_tf, 0), 0), 3)
  stim_time_filtered = tf.nn.conv2d(stim_tf, ttf_4d,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID')  # Batch x X x Y x 1
  stim_time_filtered_3d = stim_time_filtered[:, :, :, 0]
  stim_time_filtered_2d = tf.reshape(stim_time_filtered_3d, [-1, dimx * dimy])
  k_tf_flat = tf.reshape(k_tf, [dimx * dimy, n_cells])
  lam_raw = tf.matmul(stim_time_filtered_2d, k_tf_flat)  # Batch x n_cells

  # select_cells - all outputs of size Batch x n_selected_cells
  pos_resp_sel = tf.matmul(pos_resp[:, :, 0], select_cells_mat)
  neg_resp_sel = tf.matmul(neg_resp[:, :, 0], select_cells_mat)
  lam_raw_sel = tf.matmul(lam_raw, select_cells_mat)

  # Loss
  beta = FLAGS.beta
  d_pos = - tf.reduce_mean(pos_resp_sel * lam_raw_sel, 1)  # Batch
  d_pairwise_neg = - tf.reduce_mean(tfd(neg_resp_sel, 0) *
                                    tfd(lam_raw_sel, 1), 2)  # Batch

  difference = (tf.expand_dims(d_pos/beta, 1) -
                d_pairwise_neg/beta)  # postives x negatives

  # Option 2
  # log(1 + \sum_j(exp(d+ - dj-)))
  difference_padded = tf.pad(difference, [[0, 0], [0, 1]])
  loss = tf.reduce_sum(beta * tf.reduce_logsumexp(difference_padded, 1), 0)

  accuracy_tf = tf.reduce_mean(tf.sign(-tf.expand_dims(d_pos, 1) +
                                       d_pairwise_neg))

  ## Train model
  if sr_model == 'lin_rank1':
    train_op_part = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(loss)
    # Project K
    with tf.control_dependencies([train_op_part]):
      # Locally reweighted L1 for sptial locality.
      # proj_k = project_spatially_locaized_l1(k_tf, dimx, dimy, n_cells,
      #                                       lnl1_reg=0.00001, eps_neigh=0.01)

      # Project K to the mask
      cell_masks_tf = tf.constant(cell_masks.astype(np.float32))
      proj_k = tf.assign(k_tf, k_tf * cell_masks_tf)
    train_op = tf.group(train_op_part, proj_k)

  elif sr_model == 'lin_rank1_blind':
    # Only time filter is trained in blind model.
    # train_op = tf.train.AdagradOptimizer(FLAGS.learning_rate
    #                                    ).minimize(loss, var_list=[ttf_tf])
    train_op = []  # Blind model not trained.

  else:
    raise ValueError('Only lin_rank1 and lin_rank1_blind supported')

  # Store everything in a graph.
  sr_graph = collections.namedtuple('SR_Graph', 'sess train_op '
                                    'select_cells_mat '
                                    'd_s_r_pos d_pairwise_s_rneg '
                                    'loss accuracy_tf stim_tf '
                                    'pos_resp neg_resp ttf_tf k_tf ')

  sr_graph = sr_graph(sess=sess,
                      train_op=train_op,
                      select_cells_mat=select_cells_mat,
                      d_s_r_pos=d_pos,
                      d_pairwise_s_rneg=d_pairwise_neg,
                      loss=loss, accuracy_tf=accuracy_tf, stim_tf=stim_tf,
                      pos_resp=pos_resp, neg_resp=neg_resp,
                      ttf_tf=ttf_tf, k_tf=k_tf)

  return sr_graph

