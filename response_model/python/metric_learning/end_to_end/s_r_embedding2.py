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
r"""Jointly embed stimulus and responses, learn RF and ttf."""

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
import tensorflow.models.research.spatial_transformer as spatial_transformer

# utils
import retina.prosthesis.end_to_end.utils as utils

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=()):

  # load default data
  data = utils.get_data_retina()

  # verify utils
  utils.verify_data(data)
  n_cells = data['responses'].shape[1]
  data.update({'n_cells': n_cells})

 #########################################################################
 # Joint stim-resp embedding
 #########################################################################

 #########################################################################
 # set up model.
  with tf.Session() as sess:

    # Embed stimulus
    stim_tf = tf.placeholder(tf.float32,
                             shape=[None, data['stimx'],
                                    data['stimy'], 30]) # batch x X x Y x 30
    ttf_tf = tf.Variable(np.ones(30).astype(np.float32)/10)
    filt = tf.expand_dims(tf.expand_dims(tf.expand_dims(ttf_tf, 0), 0), 3)
    stim_time_filt = tf.nn.conv2d(stim_tf, filt,
                                  strides=[1, 1, 1, 1], padding='SAME') # batch x X x Y x 1

    # Embed response using RF
    resp_tf = tf.placeholder(tf.float32, shape=[None, n_cells]) # batch x n_cells
    resp_neg_tf = tf.placeholder(tf.float32, shape=[None, n_cells]) # batch_neg x n_cells
    rf_tf = tf.Variable(data['rfs'].astype(np.float32)/100)  # X x Y x n_cells

    def embed_responses(resp_tf, rf_tf, data):
      rf_tf_flat = tf.reshape(rf_tf,
                              [data['stimx'] * data['stimy'], data['n_cells']])
      resp_decode_flat = tf.matmul(resp_tf, tf.transpose(rf_tf_flat))
      resp_decode = tf.expand_dims(tf.reshape(resp_decode_flat,
                                              [-1, data['stimx'],
                                               data['stimy']]), 3)
      return resp_decode

    resp_decode = embed_responses(resp_tf, rf_tf, data) # batch x X x Y x 1
    resp_neg_decode = embed_responses(resp_neg_tf, rf_tf, data) # batch_neg x X x Y x 1

    d_s_r_pos = tf.reduce_sum((stim_time_filt - resp_decode)**2, [1, 2, 3]) # batch

    d_pairwise_s_rneg = tf.reduce_sum((tf.expand_dims(stim_time_filt, 1) -
                                 tf.expand_dims(resp_neg_decode, 0))**2, [2, 3, 4]) # batch x batch_neg

    beta = 1000
    loss = tf.reduce_sum(beta * tf.reduce_logsumexp(tf.expand_dims(d_s_r_pos / beta, 1) -
                                                  d_pairwise_s_rneg / beta, 1), 0)

    train_op_ttf = tf.train.AdagradOptimizer(0.1).minimize(loss, var_list=[ttf_tf])

    # L1 norm regularization of RF
    '''
    train_op_rf = tf.train.ProximalAdagradOptimizer(0.1,
                                                    l1_regularization_strength=20.0).minimize(loss,
                                                    var_list=[rf_tf])
    '''

    # Spatially localized regularization of RF
    neighbor_mat = utils.get_neighbormat(np.ones((data['stimx'], data['stimy'])))
    n_mat = tf.constant(neighbor_mat.astype(np.float32))
    eps = 0.001
    wts_tf = 1 / (tf.matmul(n_mat, tf.reshape(tf.abs(rf_tf),
                                              [data['stimx'] * data['stimy'],
                                               data['n_cells']])) + eps)
    wts_tf_3d = tf.reshape(wts_tf, [data['stimx'], data['stimy'], data['n_cells']])
    train_op_rf_simple = tf.train.AdagradOptimizer(0.1).minimize(loss,
                                                                 var_list=[rf_tf])
    lam_l1 = 0.005
    with tf.control_dependencies([train_op_rf_simple]):
      proj_rf_tf = tf.assign(rf_tf, tf.nn.relu(rf_tf - wts_tf_3d * lam_l1) -
                                    tf.nn.relu(-rf_tf - wts_tf_3d * lam_l1))
    train_op_rf = tf.group(train_op_rf_simple, proj_rf_tf)

    # final training op
    train_op = tf.group(train_op_ttf, train_op_rf)
    sess.run(tf.global_variables_initializer())

    #########################################################################
    # get training and testing data
    frac_train = 0.8
    tms_train = np.arange(np.floor(frac_train * data['stimulus'].shape[0])).astype(np.int)
    tms_test = np.arange(np.floor(frac_train * data['stimulus'].shape[0]),
                                   data['stimulus'].shape[0]).astype(np.int)
    data_train = {'stimulus': data['stimulus'][tms_train, :, :],
                  'responses': data['responses'][tms_train, :],
                  'ei_magnitude': data['ei_magnitude']}

    data_test = {'stimulus': data['stimulus'][tms_test, :, :],
                  'responses': data['responses'][tms_test, :],
                  'ei_magnitude': data['ei_magnitude']}
    #########################################################################
    # Setup regular testing.
    # get test data
    stim_test, resp_test, _, resp_test_neg = utils.get_test_batch(data_test,
                                                                  batch_size=10000,
                                                                  stim_history=30)



    #########################################################################
    # Do learning
    plt.ion()
    prev_loss = []
    loss_plot = []
    random_cells = np.random.randint(0, n_cells, 13)

    for iiter in range(1000000):

      # update parameters
      stim_batch, resp_batch, _, resp_batch_neg = utils.get_train_batch(data_train, batch_size=1000,
                                                     stim_history=30,
                                                     batch_neg_resp=100)
      feed_dict = {stim_tf: stim_batch,
                   resp_tf: resp_batch,
                   resp_neg_tf:resp_batch_neg}
      _, l_np = sess.run([train_op, loss], feed_dict=feed_dict)

      # plot some results
      if iiter % 1 == 0 :
        ttf_np, rf_np = sess.run([ttf_tf, rf_tf])

        # plot learned ttf on each iteration
        plt.clf()
        plt.subplot(4, 4, 1)
        plt.plot(ttf_np)

        for iicell, icell in enumerate(random_cells):
          plt.subplot(4, 4, iicell + 2)
          plt.imshow(rf_np[:, :, icell], cmap='gray', interpolation='nearest')
          plt.colorbar()


        # plot loss curve
        prev_loss += [l_np]
        if len(prev_loss) > 10:
          prev_loss = prev_loss[-10:]
        loss_plot += [np.mean(prev_loss)]
        plt.subplot(4, 4, 15)
        plt.plot(loss_plot)
        plt.show()
        plt.draw()

        # plot ROC curve
        plt.subplot(4, 4, 16)
        # get distance between positive and negative pairs.
        feed_dict = {stim_tf: stim_test,
                     resp_tf: resp_test}
        distances_pos = sess.run(d_s_r_pos, feed_dict=feed_dict)

        feed_dict = {stim_tf: stim_test,
                    resp_tf: resp_test_neg}
        distances_neg = sess.run(d_s_r_pos, feed_dict=feed_dict)
        TPR, FPR = utils.get_ROC(distances_pos, distances_neg)

        plt.plot(FPR, TPR)
        plt.plot([0, 1], [0, 1], 'g')
        plt.title('ROC - Stim - resp')


        plt.suptitle('Iters: %d' % iiter)
        plt.pause(0.0001)






if __name__ == '__main__':
   app.run()
