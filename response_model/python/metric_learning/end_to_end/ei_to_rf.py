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
r"""Learn RF from EIs

Run using:
ei_to_rf --logtostderr
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import tensorflow as tf
import tensorflow.contrib.slim as slim
from absl import app

import numpy as np, h5py,numpy

# for plotting stuff
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# for embedding stuff
import retina.prosthesis.end_to_end.embedding0 as em
import tensorflow.models.research.transformer.spatial_transformer as spatial_transformer

# utils
import retina.prosthesis.end_to_end.utils as utils

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=()):

  # load default data
  data = utils.get_data_retina()

  # verify utils
  # utils.verify_data(data)

  #########################################################################
  # Compute RF from EI.
  with tf.Session() as sess:

    # transform EI into RF for 1 retina.
    from IPython import embed; embed()
    # plot RF and EI for multiple retinas
    piece_list = ['2005-01-21-3_ei_sta.mat', '2005-04-06-4_ei_sta.mat',
                  '2005-04-26-0_ei_sta.mat', '2007-08-24-1_ei_sta.mat',
                 '2012-09-24-2_ei_sta.mat', '2015-03-09-0_ei_sta.mat']
    for ip, ipiece in enumerate(piece_list):
      dat = sio.loadmat('/home/bhaishahster'
                        '/Downloads/ei_sta/%s' % ipiece)
      plt.subplot(3, 2, ip + 1)
      plt.plot(dat['sta_params'][:, 0], dat['sta_params'][:, 1], 'r.')
    plt.show()


    # get target RFs and source
    rfs_pos = data['rfs'] * (data['cell_type'].T - 1.5) * 2
    rfs_target = np.expand_dims(np.transpose(rfs_pos, [2, 0, 1]), 3).astype(np.float32)
    source_ei = data['ei_magnitude']
    n_cells = source_ei.shape[1]
    batch_cells = 180

    # weigh nonzeros higher
    weights_loc = np.where(rfs_target != 0)
    total_elems = np.prod(rfs_target.shape)
    nz = weights_loc[0].shape[0]
    beta = nz / total_elems
    alpha = (total_elems - nz) / total_elems
    wts_mask = beta * np.ones_like(rfs_target)
    wts_mask[weights_loc[0], weights_loc[1], weights_loc[2], weights_loc[3]] = alpha


    # make placeholders
    rfs_target_tf = tf.placeholder(tf.float32)
    ei_tf = tf.placeholder(tf.float32, shape = [data['n_elec'], None])  # n_elec x # cells

    # make network
    ei_embed_tf = tf.constant(data['ei_embedding_matrix'].astype(np.float32),
                              name='ei_embedding')  # eix x eiy x n_elec
    ei_embed_2d_tf = tf.reshape(ei_embed_tf, [data['eix'] * data['eiy'],
                                              data['n_elec']])
    ei_embed = tf.matmul(ei_embed_2d_tf, ei_tf)
    ei_embed_3d = tf.reshape(ei_embed, [data['eix'], data['eiy'], -1])

    net = tf.expand_dims(tf.transpose(ei_embed_3d, [2, 0, 1]), 3) # cells x X x Y x 1

    # Pass through CNNs
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(0.005),
                        #normalizer_fn=slim.batch_norm,
                        #normalizer_params={'is_training': True}
                        ):
      net =  slim.repeat(net, 2, slim.conv2d, 1,
                         [4, 4], scope='conv_ei_0')

    # Rotate
    identity = np.array([[1., 0., 0.],
                         [0., 1., 0.]])
    identity = identity.flatten()
    theta = tf.Variable(initial_value=identity)
    out_size = [16, 32]
    rf_estimate = spatial_transformer.transformer(net, tf.tile(tf.expand_dims(theta, 0),
                                                       [batch_cells, 1]),
                                          out_size, size='st1')

    wts_mask_tf = tf.placeholder(tf.float32)
    loss = tf.norm(wts_mask_tf * (rfs_target_tf - rf_estimate), ord=1)

    train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

    sess.run(tf.global_variables_initializer())

    for iiter in range(1000000):
      random_cells = np.arange(n_cells)  # np.random.randint(0, n_cells, batch_cells)
      feed_dict = {rfs_target_tf: rfs_target[random_cells, :, :, :],
                   ei_tf: source_ei[:, random_cells],
                   wts_mask_tf: wts_mask[random_cells, :, :, :].astype(np.float32)}
      _, l_np = sess.run([train_op, loss], feed_dict=feed_dict)
      if iiter % 100 == 1:
        print(l_np)


    ## plot mapping results
    rf_mapped, rf_tar = sess.run([rf_estimate, rfs_target_tf],
                                 feed_dict=feed_dict)

    n_cells = 6
    random_cells = np.random.randint(0, data['ei_magnitude'].shape[1], n_cells)
    plt.figure()
    for icell in range(n_cells):
      # plot EI
      plt.subplot(n_cells, 2, 2 * icell + 1)
      # plt.scatter(elec_loc[:, 0], elec_loc[:, 1],
      #             data['ei_magnitude'][:, random_cells[icell]])
      plt.imshow(rf_mapped[random_cells[icell], :, :, 0], cmap='gray',
                 interpolation='nearest')
      plt.axis('Image')

      # plot RF
      plt.subplot(n_cells, 2, 2 * icell + 2)
      plt.imshow(rf_tar[random_cells[icell], :, :, 0], cmap='gray',
                 interpolation='nearest')
      plt.axis('Image')
    plt.show()
    #########################################################################
    ei_embed_np = sess.run(ei_embed_3d, feed_dict={ei_tf: data['ei_magnitude']})


    # plot RF and EI-embedded for few cells to see the correspondce.
    rfs_pos = data['rfs'] * (data['cell_type'].T - 1.5) * 2
    n_cells = 6
    random_cells = np.random.randint(0, data['ei_magnitude'].shape[1], n_cells)
    plt.figure()
    for icell in range(n_cells):
      # plot EI
      plt.subplot(n_cells, 2, 2 * icell + 1)
      # plt.scatter(elec_loc[:, 0], elec_loc[:, 1],
      #             data['ei_magnitude'][:, random_cells[icell]])
      plt.imshow(ei_embed_np[:, :, random_cells[icell]].T, cmap='gray',
                 interpolation='nearest')
      plt.axis('Image')

      # plot RF
      plt.subplot(n_cells, 2, 2 * icell + 2)
      plt.imshow(rfs_pos[:, ::-1, random_cells[icell]], cmap='gray',
                 interpolation='nearest')
      plt.axis('Image')
    plt.show()


if __name__ == '__main__':
   app.run()
