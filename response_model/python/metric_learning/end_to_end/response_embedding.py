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
r"""Different response embedddings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import retina.response_model.python.metric_learning.end_to_end.stimulus_embedding as embed

slim = tf.contrib.slim


class Convolutional(object):
  """Convolutional distance over local groups."""

  def __init__(self, **kwargs):
    """Initialize the tensorflow model to learn the metric."""

    tf.logging.info('Building graph for convolutional metric')
    self._build_graph(**kwargs)

  def _build_graph(self, time_window, layers, batch_norm,
                   is_training, reuse_variables, num_cell_types, dimx, dimy, responses_tf=None):
    """Build tensorflow graph for learning the metric.

    Currently support only time_window=1.

    Args:
      time_window : number of time bins fo responses used.
                    Currently support only time_window=1.
      layers : Architecture of the network.
               Format - (a1, b1, c1, a2, b2, c2 .. ),
               where (a, b, c) = (filter width, num of filters, stride)
      batch_norm : If batch-norm is applied between layers.
      is_training : If its training or testing (used for normalization layers)
      reuse_variables : Whether to reuse an already created graph.
      num_cell_types : Number of cell types.
      dimx : X dimension of initial cell center embedding.
      dimy : Y dimension of initial cell center embedding.
    """

    # placeholders for responses, centers, cell_types and mean firing rates
    if responses_tf is None:
      responses_tf = tf.placeholder(dtype=tf.float32,
                                    shape=[None, None, time_window],
                                    name='anchor')  # batch x cells x time_window
    map_cell_grid_tf = tf.placeholder(dtype=tf.float32, shape=[dimx, dimy,
                                                               None],
                                      name='centers')  # dimx x dimy x n_cells
    cell_types_tf = tf.placeholder(dtype=tf.float32,
                                   shape=[None,
                                          num_cell_types])  # n_cellxcell_types
    mean_fr_tf = tf.placeholder(dtype=tf.float32,
                                shape=[None], name='mean_fr')  # n_cells

    # embed anchor, pos, neg responses onto a 2D grid
    with tf.variable_scope('mean_fr_fcn', reuse=reuse_variables):
      a = tf.get_variable('poly_mean_fr', shape=[4])
    cell_weights_tf = (a[0]*mean_fr_tf**3 + a[1]*mean_fr_tf**2 +
                       a[2]*mean_fr_tf + a[3])
    self.responses_embed_1 = self._embed_responses(responses_tf,
                                                   map_cell_grid_tf,
                                                   cell_types_tf,
                                                   cell_weights_tf,
                                                   reuse_variables=
                                                   reuse_variables)

    # embed anchor, pos, neg through a neural network
    self.layer_sizes = layers.split(',')
    op_ = self._embed_network(self.responses_embed_1,
                              self.layer_sizes,
                              reuse_variables=reuse_variables,
                              batch_norm=batch_norm,
                              is_training=is_training)
    self.responses_embed, self.layer_collection = op_
    
    self.params = slim.get_model_variables() + [a]

    self.responses_tf = responses_tf
    self.map_cell_grid_tf = map_cell_grid_tf
    self.cell_types_tf = cell_types_tf
    self.mean_fr_tf = mean_fr_tf

  def _embed_responses(self, responses_tf, map_cell_grid_tf,
                       cell_types_tf, cell_weights_tf, reuse_variables=False):

    """Embed a response vector into a two dimensional grid.

    Args :
      responses_tf : Tensor of shape (batch_size x n_cells x time_steps).
      map_cell_grid_tf : mapping of cells to grid
                         (grid_x x grid_y x n_cells) tensor.
      cell_types_tf : gives cell types (+1/-1) of each cell.
      cell_weights_tf : weights (float) for each cell.
      reuse_variables : whether to use previously defined graph.

    Returns :
      embed_responses : Embed responses on a 2D grid,
                          separate for cells of each type.
    """

    # Assign cells at different locations in 2D grid.
    with tf.variable_scope('cell_center_rf', reuse=reuse_variables):
      init = tf.constant(np.ones((5, 5, 1, 1)).astype(np.float32))
      conv_filter = tf.get_variable(name='rf_mask',
                                    initializer=init)
    cell_rfs_transpose = tf.nn.conv2d(tf.expand_dims(tf.transpose(
        map_cell_grid_tf, [2, 0, 1]), 3),
                                      conv_filter, strides=[1, 1, 1, 1],
                                      padding='SAME')

    cell_rfs = tf.transpose(cell_rfs_transpose, [3, 1, 2, 0])

    # Weigh cell responses by mean firing rate.

    response_flat = tf.gather(tf.transpose(responses_tf, [2, 0, 1]), 0) # batch x # cells
    response_flat_weights = 2 * (response_flat - 0.5) * cell_weights_tf

    # cell_rfs = 1 x dimx x dimy x # cells
    # embed resposnes
    embed_cells = (cell_rfs *
                   tf.expand_dims(tf.expand_dims(response_flat_weights, 1), 1))

    # do 1x1 convolution to compress cells (4th dimension) of same type.
    conv_filter = tf.expand_dims(tf.expand_dims(cell_types_tf, 0), 0)
    embed_responses = tf.nn.conv2d(embed_cells, conv_filter,
                                   strides=[1, 1, 1, 1], padding='SAME')
    print(embed_responses)
    return embed_responses

  def _embed_network(self, response, layers, reuse_variables=False,
                     batch_norm=False, is_training=True):
    """Convolutional network for a hierarchical response embedding.

    Args:
      response : Cell responses embedded on a 2D grid,
                  separate for each cell type.
      layers : Network architecture.
                Format - (a1, b1, c1, a2, b2, c2 .. ),
                where (a, b, c) = (filter width, num of filters, stride)
      reuse_variables : Whether to reuse previously defined network.
      batch_norm : Whether to apply batch norm after each convolution.
      is_training : Mode for batch norm.

    Returns:
      net : Network embedding of responses.
    """

    n_layers = int(len(layers)/3)
    tf.logging.info('Number of layers: %d' % n_layers)
    # set normalization
    if batch_norm:
      normalizer_fn = slim.batch_norm
      tf.logging.info('Batch normalization')
    else:
      normalizer_fn = None

    activation_fn = tf.nn.softplus
    tf.logging.info('Logistic activation')

    # net = tf.expand_dims(response, 3)  # if two layers, not needed.
    net = response
    layer_collection = [net]
    for ilayer in range(n_layers):
      tf.logging.info('Building layer: %d, %d, %d'
                      % (int(layers[ilayer*3 + 1]), int(layers[ilayer*3]),
                         int(layers[ilayer*3 + 2])))
      net = slim.conv2d(net, int(layers[ilayer*3 + 1]),
                        int(layers[ilayer*3]),
                        stride=int(layers[ilayer*3 + 2]),
                        scope='resp_layer_wt_%d' % ilayer,
                        reuse=reuse_variables,
                        normalizer_fn=normalizer_fn,
                        activation_fn=activation_fn,
                        normalizer_params={'is_training': is_training})
      layer_collection += [net]

    return net, layer_collection



class Convolutional2(Convolutional):
  """Convolutional distance over local groups.

  Just like class above, except we replace first level with convolution over
    gaussians with with distance determined by nearest neighbor distances.
  """

  def _build_graph(self, time_window, layers, batch_norm,
                   is_training, reuse_variables, num_cell_types, dimx, dimy, responses_tf=None):
    """Build tensorflow graph for learning the metric.

    Currently support only time_window=1.

    Args:
      time_window : number of time bins fo responses used.
                    Currently support only time_window=1.
      layers : Architecture of the network.
               Format - (a1, b1, c1, a2, b2, c2 .. ),
               where (a, b, c) = (filter width, num of filters, stride)
      batch_norm : If batch-norm is applied between layers.
      is_training : If its training or testing (used for normalization layers)
      reuse_variables : Whether to reuse an already created graph.
      num_cell_types : Number of cell types.
      dimx : X dimension of initial cell center embedding.
      dimy : Y dimension of initial cell center embedding.
    """

    # placeholders for responses, centers, cell_types and mean firing rates
    if responses_tf is None:
      responses_tf = tf.placeholder(dtype=tf.float32,
                                    shape=[None, None, time_window],
                                    name='anchor')  # batch x cells x time_window
    map_cell_grid_tf = tf.placeholder(dtype=tf.float32, shape=[dimx, dimy,
                                                               None],
                                      name='centers')  # dimx x dimy x n_cells
    cell_types_tf = tf.placeholder(dtype=tf.float32,
                                   shape=[None,
                                          num_cell_types])  # n_cell x cell_types
    mean_fr_tf = tf.placeholder(dtype=tf.float32,
                                shape=[None], name='mean_fr')  # n_cells

    dist_nn = tf.placeholder(dtype=tf.float32,
                                shape=[2], name='dist_nn')  # cells_types

    # embed anchor, pos, neg responses onto a 2D grid
    with tf.variable_scope('mean_fr_fcn', reuse=reuse_variables):
      a = tf.get_variable('poly_mean_fr', shape=[4])
    cell_weights_tf = (a[0]*mean_fr_tf**3 + a[1]*mean_fr_tf**2 +
                       a[2]*mean_fr_tf + a[3])
    op = self._embed_responses(responses_tf, map_cell_grid_tf,
                               cell_types_tf, cell_weights_tf, dist_nn,
                               reuse_variables=reuse_variables)
    self.responses_embed_1, self.response_flat_weights, self.gaussians_layer1 = op

    # embed anchor, pos, neg through a neural network
    self.layer_sizes = layers.split(',')
    op_ = self._embed_network(self.responses_embed_1,
                              self.layer_sizes,
                              reuse_variables=reuse_variables,
                              batch_norm=batch_norm,
                              is_training=is_training)
    self.responses_embed, self.layer_collection = op_

    self.params = slim.get_model_variables() + [a]

    self.responses_tf = responses_tf
    self.map_cell_grid_tf = map_cell_grid_tf
    self.cell_types_tf = cell_types_tf
    self.mean_fr_tf = mean_fr_tf
    self.dist_nn = dist_nn

    self.embed_responses_original = self._embed_responses_original()
    self.embed_locations_original = tf.reshape(tf.matmul(tf.reshape(self.map_cell_grid_tf, [dimx * dimy, -1]),
                                                         self.cell_types_tf), [dimx, dimy, -1])  # dimx x dimy x num_cell_types(2)

  def _embed_responses(self, responses_tf, map_cell_grid_tf,
                       cell_types_tf, cell_weights_tf, dist_nn, reuse_variables=False):

    """Embed a response vector into a two dimensional grid.

    Args :
      responses_tf : Tensor of shape (batch_size x n_cells x time_steps).
      map_cell_grid_tf : mapping of cells to grid
                         (grid_x x grid_y x n_cells) tensor.
      cell_types_tf : gives cell types (+1/-1) of each cell.
      cell_weights_tf : weights (float) for each cell.
      dist_nn : Separation between nearest neighbors of each cell type
                  (# cell_types)
      reuse_variables : whether to use previously defined graph.

    Returns :
      embed_responses : Embed responses on a 2D grid,
                          separate for cells of each type.
    """

    # Weigh cell responses by mean firing rate.
    response_flat = tf.gather(tf.transpose(responses_tf, [2, 0, 1]), 0)
    response_flat_weights = 2 * (response_flat - 0.5) * cell_weights_tf

    # Get gaussian filter for each cell type
    embed_responses_list = []
    tfd = tf.expand_dims
    xx, yy = np.meshgrid(np.arange(5), np.arange(5))
    g = []
    for icell_type in range(2):
      sigma = (dist_nn[icell_type] / 2) / 1.7
      k  = tf.exp(-((xx - 2.0) ** 2 + (yy - 2.0) ** 2) / (2 * (sigma ** 2)))
      k = k / tf.sqrt(tf.reduce_sum(k ** 2))
      k_4d = tfd(tfd(k, 2), 3)
      g += [k_4d]

      cell_rfs_transpose = tf.nn.conv2d(tfd(tf.transpose(map_cell_grid_tf,
                                                         [2, 0, 1]), 3), k_4d,
                                        strides=[1, 1, 1, 1], padding='SAME')  # n_cells x dimx x dimy x 1
      cell_rfs = tf.transpose(cell_rfs_transpose, [3, 1, 2, 0])  # 1 x dimx x dimy x n_cells
      # cell_rfs = 1 x dimx x dimy x # cells
      # embed resposnes
      embed_cells = (cell_rfs * tf.expand_dims(tf.expand_dims(response_flat_weights, 1), 1))
      # do 1x1 convolution to compress cells (4th dimension) of same type.
      conv_filter = tfd(tfd(tfd(cell_types_tf[:, icell_type], 1), 0), 0)
      embed_responses_list += [tf.nn.conv2d(embed_cells, conv_filter,
                                     strides=[1, 1, 1, 1], padding='SAME')]

    embed_responses = tf.concat(embed_responses_list, 3)

    return embed_responses, response_flat_weights, g

  def _embed_responses_original(self):
    tfd = tf.expand_dims
    resp_4d = tfd(tfd(tf.gather(tf.transpose(self.responses_tf, [2, 0, 1]), 0), 1), 1)
    embed_cells = resp_4d * tfd(self.map_cell_grid_tf, 0)  # batch x dimx x dimy x cells
    embed_responses_list = []
    for icell_type in range(2):
      conv_filter = tfd(tfd(tfd(self.cell_types_tf[:, icell_type], 1), 0), 0)
      embed_responses_list += [tf.nn.conv2d(embed_cells, conv_filter,
                                          strides=[1, 1, 1, 1], padding='SAME')]
    embed_responses = tf.concat(embed_responses_list, 3)  # batch x dimx dimy x 2

    return embed_responses


  def _embed_network(self, response, layers, reuse_variables=False,
                     batch_norm=False, is_training=True):
    """Convolutional network for a hierarchical response embedding.

    Args:
      response : Cell responses embedded on a 2D grid,
                  separate for each cell type.
      layers : Network architecture.
                Format - (a1, b1, c1, a2, b2, c2 .. ),
                where (a, b, c) = (filter width, num of filters, stride)
      reuse_variables : Whether to reuse previously defined network.
      batch_norm : Whether to apply batch norm after each convolution.
      is_training : Mode for batch norm.

    Returns:
      net : Network embedding of responses.
    """

    n_layers = int(len(layers)/3)
    tf.logging.info('Number of layers: %d' % n_layers)
    # set normalization
    if batch_norm:
      normalizer_fn = slim.batch_norm
      tf.logging.info('Batch normalization')
    else:
      normalizer_fn = None

    activation_fn = tf.nn.softplus
    tf.logging.info('Logistic activation')

    # net = tf.expand_dims(response, 3)  # if two layers, not needed.
    net = response
    layer_collection = [net]
    for ilayer in range(n_layers):
      if ilayer == n_layers - 1:
        activation_fn = None
      else:
        activation_fn = tf.nn.softplus
      tf.logging.info('Building layer: %d, %d, %d'
                      % (int(layers[ilayer*3 + 1]), int(layers[ilayer*3]),
                         int(layers[ilayer*3 + 2])))
      net = slim.conv2d(net, int(layers[ilayer*3 + 1]),
                        int(layers[ilayer*3]),
                        stride=int(layers[ilayer*3 + 2]),
                        scope='resp_layer_wt_%d' % ilayer,
                        reuse=reuse_variables,
                        normalizer_fn=normalizer_fn,
                        activation_fn=activation_fn,
                        normalizer_params={'is_training': is_training})
      layer_collection += [net]

    return net, layer_collection


class Residual(Convolutional2):
  def _embed_network(self, response, layers, reuse_variables=False,
                     batch_norm=False, is_training=True):

    net, layer_collection = embed.residual_encode(layers, batch_norm, response,
                                                  is_training,
                                                  reuse_variables=reuse_variables,
                                                  scope_prefix='resp',
                                                  input_channels=2)

    return net, layer_collection

