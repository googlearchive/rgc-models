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
r""""Population response metric with convolutional structure.

Distance between two responses (r1, r2) computed as ||phi(r1) - phi(r2)||^2.

Embedding of responses phi(r) is given as follows:
1) Responses to 1/-1:  $\tilde{r} = 2*(r-0.5) $
2) Learn a weight for each cell. (scale(i))
3) Map each cell to its center location on grid (grid of size 80 x 40).
Let $M_{i}$ be grid embedding on cell $i$.
So,  $M_{i}$ has zero for all positions except center of cell.
4) 5x5 convolution of each cell's $M_{i}$ to get
    RF estimate of cell ($\tilde{M}_i$).
5) Activation map for each cell type
   $A_{i} = \sum_{i}\tilde{r}_{i}*scale(i)*\tilde{M}_i$.
6) Hence, responses embedded using 2 layered activation map
    of ON and OFF parasols each.
7) Responses embedded further by layers of convolution, batch norm and relu.


# pylint: disable-line-too-long
Run using:
--model='conv' --logtostderr \
--data_path='/cns/oi-d/home/bhaishahster/metric_learning/examples_pc2017_04_25_1/' \
--data_train='data012_lr_remove_dups_with_stimulus_train_grp1.mat' \
--data_test='data012_lr_remove_dups_with_stimulus_test_grp1.mat' \
--save_suffix='_2017_04_25_1_all_cells_np_dups' --gfs_user='foam-brain-gpu' \
--triplet_type='a' \
--grid_resolution=0.5 \
--convolutional_layers='10, 7, 6' \
--batch_norm=True

# pylint: enable-line-too-long
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from retina.response_model.python.metric_learning.score_fcns import metric


class ConvolutionalScore(metric.Metric):
  """Convolutional distance over local groups."""

  def __init__(self, sess, save_folder, file_name, **kwargs):
    """Initialize the tensorflow model to learn the metric."""

    tf.logging.info('Building graph for convolutional metric')
    self._build_graph(**kwargs)

    self.build_summaries()
    tf.logging.info('Summary operator made')

    self.sess = sess
    self.initialize_model(save_folder, file_name, sess)
    tf.logging.info('Model initialized')

  def _build_graph(self, n_cells, time_window, lr, centers, resolution,
                   layers, lam, batch_norm, is_training, cell_type,
                   cell_statistics=None, triplet_type='a', beta=1):
    """Build tensorflow graph for learning the metric.


    Args:
      n_cells : number of cells
      time_window : number of time bins fo responses used.
                    Currently support only time_window=1.
      lr : learning rate
      centers: location of centers for each cells (n_cells x 2)
      resolution : resolution of the grid in visual space,
                   where cell responses are embedded.
      layers : Architecture of the network.
               Format - (a1, b1, c1, a2, b2, c2 .. ),
               where (a, b, c) = (filter width, num of filters, stride)
      lam : L2 regularization on parameters of the network.'
      batch_norm : If batch-norm is applied between layers.
      is_training : if its training or testing (used for normalization layers)
      cell_type : Cell type information for each cell ( length: n_cells)
      cell_statistics : Mean firing rate of each cell
      triplet_type : Type of triplet used for training.
                     Loss modified accordingly.
      beta : Temperature parameter for soft-max loss.
    """

    # account for triplet_type
    self.triplet_type = triplet_type
    self.beta = beta

    # make centers into grid points
    output_grid = self._give_cell_grid(centers, resolution)
    self.centers_grid, self.grid_size, self.map_cell_grid = output_grid
    self.map_cell_grid_tf = tf.constant(self.map_cell_grid.astype(np.float32))

    self.dim = n_cells * time_window

    self.cell_weights = self._get_cell_weights(cell_type, cell_statistics)

    self.layer_sizes = layers.split(',')

    # placeholders for anchor, pos and neg
    self.anchor = tf.placeholder(dtype=tf.float32,
                                 shape=[None, None, time_window],
                                 name='anchor')
    self.pos = tf.placeholder(dtype=tf.float32,
                              shape=[None, None, time_window], name='pos')
    self.neg = tf.placeholder(dtype=tf.float32,
                              shape=[None, None, time_window], name='neg')

    # embed anchor, pos, neg responses onto a 2D grid
    self.embed_anchor = self._embed_responses(self.anchor,
                                              self.map_cell_grid_tf,
                                              n_cells, self.grid_size,
                                              cell_type, reuse_variables=False)

    self.embed_pos = self._embed_responses(self.pos,
                                           self.map_cell_grid_tf,
                                           n_cells, self.grid_size,
                                           cell_type, reuse_variables=True)

    self.embed_neg = self._embed_responses(self.neg,
                                           self.map_cell_grid_tf,
                                           n_cells, self.grid_size,
                                           cell_type, reuse_variables=True)

    # embed anchor, pos, neg through a neural network
    self.nn_anchor = self._embed_network(self.embed_anchor,
                                         self.layer_sizes,
                                         reuse_variables=False,
                                         batch_norm=batch_norm,
                                         is_training=is_training)

    self.nn_pos = self._embed_network(self.embed_pos,
                                      self.layer_sizes,
                                      reuse_variables=True,
                                      batch_norm=batch_norm,
                                      is_training=is_training)

    self.nn_neg = self._embed_network(self.embed_neg,
                                      self.layer_sizes,
                                      reuse_variables=True,
                                      batch_norm=batch_norm,
                                      is_training=is_training)

    self.params = slim.get_model_variables() + [self.cell_weights]

    self._get_loss()

    l2_norm = [tf.nn.l2_loss(x) for x in self.params]
    loss_regularization = lam * tf.reduce_sum(l2_norm)
    self.loss = self.loss_triplet + loss_regularization
    self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)

  def _get_loss(self):
    """Get loss based on triplet type."""

    if self.triplet_type == 'a':
      self.score_anchor_pos = self._scores_pairwise(self.nn_anchor, self.nn_pos)
      self.score_anchor_neg = self._scores_pairwise(self.nn_anchor, self.nn_neg)
      self.score_pos_neg = self._scores_pairwise(self.nn_pos, self.nn_neg)

      self.score_neg = tf.minimum(self.score_anchor_neg, self.score_pos_neg)
      self.loss_triplet = (tf.reduce_sum(tf.nn.relu(self.score_anchor_pos -
                                                    self.score_neg + 1)))
      tf.logging.info('Set symmetric triplet loss')

    else:

      self.score_anchor_pos = self._scores_pairwise(self.nn_anchor,
                                                    self.nn_pos)
      self.score_anchor_all_neg = self._scores_all_negs(self.nn_anchor,
                                                        self.nn_neg)

      # Loss (changed on Oct 21, 2017 from Alternative 1 to Alternative 2)
      beta = self.beta
      difference = (tf.expand_dims(self.score_anchor_pos/beta, 1) -
                    self.score_anchor_all_neg/beta)  # postives x negatives

      # Alternative 1 - log-sum-exp loss
      # log(\sum_j(exp(d+ - dj-)))
      # loss = tf.reduce_sum(beta * tf.reduce_logsumexp(difference, 1), 0)

      # Alternative 2 - log(1 + \sum_j(exp(d+ - dj-))), makes it more robust
      difference_padded = tf.pad(difference, [[0, 0], [0, 1]])
      loss = tf.reduce_sum(beta * tf.reduce_logsumexp(difference_padded, 1), 0)
      self.loss_triplet = loss

      tf.logging.info('Set batch triplet loss')

  def _get_cell_weights(self, cell_type, cell_statistics):
    return tf.Variable(np.ones(self.dim).astype(np.float32),
                       name='cell_weights')

  def _give_cell_grid(self, centers, resolution):
    """Embeds each center on a discrete grid.

    Args:
      centers: center location of cells (n_cells x 2).
      resolution: Float specifying the resolution of grid.

    Returns:
      centers_grid : Discretized centers (n_cells x 2).
      grid_size : dimensions of the grid (2D integer tuple).
      map_cell_grid : mapping between cells to grid (grid_x x grid_y x n_cells)
    """

    n_cells = centers.shape[0]
    centers_grid = np.floor(centers / resolution)
    centers_grid -= np.min(centers_grid, 0)
    grid_size = np.max(centers_grid, 0).astype(np.int32) + 1

    # map_cell_grid is location of each cell to grid point
    map_cell_grid = np.zeros((grid_size[0], grid_size[1], n_cells))
    for icell in range(n_cells):
      map_cell_grid[centers_grid[icell, 0], centers_grid[icell, 1], icell] = 1
    tf.logging.info('Centers made into grid points')

    return centers_grid, grid_size, map_cell_grid

  def _embed_responses(self, response, map_cell_grid_tf, n_cells,
                       grid_size, cell_type, reuse_variables=False):
    """Takes response vector and embeds into a sparse two dimensional grid.

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

    response_flat = tf.reshape(response, [-1, self.dim])
    response_flat_weights = response_flat * self.cell_weights
    map_cell_grid_2d = tf.transpose(tf.reshape(map_cell_grid_tf, [-1, n_cells]))

    embed_resps = []
    for cell_type_index in np.unique(cell_type):
      cells = tf.squeeze(tf.where(cell_type == cell_type_index))
      resp_cell_type = tf.transpose(tf.gather(
          tf.transpose(response_flat_weights), cells))
      map_cell_cell_type = tf.gather(map_cell_grid_2d, cells)
      embed_response_2d = tf.matmul(resp_cell_type,
                                    map_cell_cell_type)
      embed_resps += [tf.expand_dims(embed_response_2d, 2)]

    embed_resps_concat = tf.concat(embed_resps, 2)
    n_cell_types = np.unique(cell_type).shape[0]
    embed = tf.reshape(embed_resps_concat, [-1, grid_size[0],
                                            grid_size[1], n_cell_types])

    return embed

  def _embed_network(self, response, layers, reuse_variables=False,
                     batch_norm=False, is_training=True):

    n_layers = int(len(layers)/3)
    tf.logging.info('Number of layers: %d' % n_layers)
    # set normalization
    if batch_norm:
      normalizer_fn = slim.batch_norm
      tf.logging.info('Batch normalization')
    else:
      normalizer_fn = None

    activation_fn = tf.nn.relu
    tf.logging.info('RELU activation')

    net = response
    for ilayer in range(n_layers):
      tf.logging.info('Building layer: %d, %d, %d'
                      % (int(layers[ilayer*3 + 1]), int(layers[ilayer*3]),
                         int(layers[ilayer*3 + 2])))
      net = slim.conv2d(net, int(layers[ilayer*3 + 1]),
                        int(layers[ilayer*3]),
                        stride=int(layers[ilayer*3 + 2]),
                        scope='layer_wt_%d' % ilayer,
                        reuse=reuse_variables,
                        normalizer_fn=normalizer_fn,
                        activation_fn=activation_fn,
                        normalizer_params={'is_training': is_training})
    return net

  def _scores_pairwise(self, nn_anchor, nn_pos):
    return tf.reduce_sum((nn_anchor - nn_pos)**2, [1, 2, 3])

  def _scores_all_negs(self, nn_anchor, nn_neg):
    x = tf.reduce_sum((tf.expand_dims(nn_anchor, 1) -
                       tf.expand_dims(nn_neg, 0))**2,
                      [2, 3, 4])
    return x

  def get_parameters(self):
    """Return insightful parameters of the score function."""
    return self.sess.run(self.params)

  def update(self, triplet_batch):
    """Given a batch of training data, update metric parameters.

    Args :
        triplet_batch : List [anchor, positive, negative], each with shape:
                        (batch x cells x time_window)
    Returns :
        loss : Training loss for the batch of data.
    """
    feed_dict = {self.anchor: triplet_batch[0],
                 self.pos: triplet_batch[1],
                 self.neg: triplet_batch[2]}
    _, loss = self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)
    tf.logging.info(loss)

    return loss

  def get_distance(self, anchor_in, pos_in):
    """Compute distance between pairs of responses.

    The two response arrays are (batch x cells x time_window) return distances
    between corresponding pairs of responses in anchor and pos.
    The distances are evaluated using tensorflow model setup.

    Args:
        anchor_in : Binned responses (each : batch x cells x time_window).
        pos_in : same as anchor.

    Returns:
        distances : evaluated distances of size (batch).
    """
    feed_dict = {self.anchor: anchor_in, self.pos: pos_in}
    distances = self.sess.run(self.score_anchor_pos, feed_dict=feed_dict)
    return distances

  def build_summaries(self):
    """Add loss summary operator."""

    # Loss summary.
    tf.summary.scalar('loss', self.loss)

    merged = tf.summary.merge_all()
    self.summary_op = merged
    tf.logging.info('summary op set')

  def get_embedding(self, resp, batch_sz=100):
    """Get embedding for responses.

    Args :
      resp : responses (Batch size x n_cells x time_window).

    Returns:
      embed_all_batches : embedding of each response.
    """

    n_resp = resp.shape[0]
    resp_idx_permute = np.random.permutation(np.arange(n_resp))

    embed_all = self.sess.run(self.nn_anchor, feed_dict={self.anchor: resp})
    '''
    embed_all_batches = np.zeros((n_resp, embed_all.shape[1],
                                  embed_all.shape[2], embed_all.shape[3]))
    for iresp in np.arange(0, n_resp, batch_sz):
      resps_selected = resp_idx_permute[iresp: iresp + batch_sz]
      embed_batch = self.sess.run(self.nn_anchor,
                                  feed_dict={self.anchor:
                                             resp[resps_selected, :, :]})

      embed_all_batches[resps_selected, :, :, :] = embed_batch
    '''
    return embed_all
