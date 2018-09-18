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
r"""Convolutional metric with responses weighed by a function of firing rate.

For use in prosthesis setting, we get rid of cell-specific parameters,
so we can train the model on few cells and
test on completely new set of cells using only the
firing rate, center location and cell type.

Distance between two responses (r1, r2) computed as ||phi(r1) - phi(r2)||^2.

Embedding of responses phi(r) is same as in convolutional.py,
modify line 2 with a learned polynomial function of firing rate (fr):

''' scale = (a[0]*fr**3 + a[1]*fr**2 + a[2]*fr + a[3]) '''


For training : train and test on differnet stimuli for same group of cells.

--data_path="/cns/oi-d/home/bhaishahster/metric_learning/examples_pc2017_04_25_1/" \
--data_test="data012_lr_remove_dups_with_stimulus_test_grp1.mat" \
--data_train="data012_lr_remove_dups_with_stimulus_train_grp1.mat" \
--grid_resolution=0.5  \
--model="conv_prosthesis" --save_suffix="_2017_04_25_1_all_cells_np_dups_grp1" \
--triplet_type="a" --batch_size_train=100 --batch_norm="False"  \
--convolutional_layers="10, 5, 5" --lam=0.0


For testing on different cells :
--data_path="/cns/oi-d/home/bhaishahster/metric_learning/examples_pc2017_04_25_1/" \
--data_test="data012_lr_remove_dups_with_stimulus_test_grp2.mat" \
--data_train="data012_lr_remove_dups_with_stimulus_train_grp2.mat" \
--grid_resolution=0.5  \
--model="conv_prosthesis" --save_suffix="_2017_04_25_1_all_cells_np_dups_grp1" \
--triplet_type="a" --batch_size_train=100 --batch_norm="False"  \
--convolutional_layers="10, 5, 5" --lam=0.0
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from retina.response_model.python.metric_learning.score_fcns import convolutional


class ConvolutionalProsthesisScore(convolutional.ConvolutionalScore):
  """Convolutional distance over local groups."""

  def _get_cell_weights(self, cell_type, cell_statistics):
    """Get weights for each cell using mean firing rate and cell type."""

    firing_rate = cell_statistics[0].astype(np.float32)
    n_poly = 4  # polynomial with factors from (0) to (n_poly-1)

    # cell type 1
    outputs = []
    for cell_type_index in np.unique(cell_type):
      cells = (cell_type == cell_type_index)
      output = tf.zeros(np.sum(cells))

      a = tf.Variable(np.ones(n_poly).astype(np.float32),
                      name='polynomial_coeffs_%d' % cell_type_index)
      for icoeff in range(n_poly):
        output += a[icoeff]*(firing_rate[cells]**icoeff)
      outputs += [output]

    concat_output = tf.concat(outputs, 0)

    return concat_output

  def _embed_responses(self, response, map_cell_grid_tf, n_cells,
                       grid_size, cell_type, reuse_variables=False):
    """Embed a response vector into a two dimensional grid.

    Args :
      response : Tensor of shape (batch_size x n_cells x time_steps).
      map_cell_grid_tf : mapping of cells to grid
                         (grid_x x grid_y x n_cells) tensor.
      n_cells : number of cells
      grid_size : size of 2D grid in which lie (int array).
      cell_type : gives cell types (+1/-1) of each cell.
      reuse_variables : same as reuse_variables in tensorflow

    Returns :
      embed : Sparse tensor of size (grid_sz x grid_sz).
             embed(i, j) = response(k) where centers_grid(k, :) = (i, j).
    """

    # separate responses of different cell types into two channels
    response_flat = tf.reshape(response, [-1, self.dim])
    response_flat_weights = response_flat * self.cell_weights

    # assign cells at different locations in 2D grid (cells x dimx x dimy x 1)
    cell_rfs_transpose = slim.conv2d(tf.expand_dims(
        tf.transpose(map_cell_grid_tf, [2, 0, 1]), 3),
                                     1, [5, 5], stride=1,
                                     activation_fn=None, padding='SAME',
                                     scope='cell_embedding_rf',
                                     reuse=reuse_variables)
    cell_rfs = tf.transpose(cell_rfs_transpose, [3, 1, 2, 0])

    # cell_rfs = 1 x dimx x dimy x # cells
    # embed resposnes (None x dimx x dimy x n_cells)
    embed_cells = cell_rfs * tf.expand_dims(
        tf.expand_dims(response_flat_weights, 1), 1)

    # do 1x1 convolution to compress in space.
    cell_embed_idx = []
    for cell_type_index in np.unique(cell_type):
      cell_embed_idx += [np.array(cell_type_index ==
                                  cell_type).astype(np.float32)]
    cell_embed_idx_tf = tf.constant(np.array(
        cell_embed_idx).T)  # cells x num_cell_types
    conv_filter = tf.expand_dims(tf.expand_dims(cell_embed_idx_tf, 0), 0)

    embed_responses = tf.nn.conv2d(embed_cells, conv_filter,
                                   strides=[1, 1, 1, 1], padding='SAME')

    print(embed_responses)
    return embed_responses
