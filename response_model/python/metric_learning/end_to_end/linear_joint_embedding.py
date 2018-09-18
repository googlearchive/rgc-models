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
r""""Jointly embed stim-resp using a linear model.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import tensorflow as tf
import tensorflow.contrib.slim as slim
from absl import app
from absl import gfile
import numpy as np, h5py,numpy
import scipy.io as sio
from numpy.random import RandomState

# for plotting stuff

import pickle
import retina.response_model.python.metric_learning.end_to_end.utils as utils
import retina.response_model.python.metric_learning.end_to_end.data_util as data_util
import retina.response_model.python.metric_learning.end_to_end.config as config
import retina.response_model.python.metric_learning.end_to_end.bookkeeping as bookkeeping
import retina.response_model.python.metric_learning.end_to_end.partitions as partitions
import retina.response_model.python.metric_learning.end_to_end.testing as testing
import retina.response_model.python.metric_learning.end_to_end.prosthesis as prosthesis
import retina.response_model.python.metric_learning.end_to_end.training as training
import retina.response_model.python.metric_learning.end_to_end.sample_datasets as sample_datasets

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=()):

  # Load stimulus-response data
  datasets = gfile.ListDirectory(FLAGS.src_dir)
  stimuli = {}
  responses = []
  print(datasets)
  for icnt, idataset in enumerate(datasets):
    #for icnt, idataset in enumerate([datasets[2]]):
    #  print('HACK - only one dataset used!!')

    fullpath = os.path.join(FLAGS.src_dir, idataset)
    if gfile.IsDirectory(fullpath):
      key = 'stim_%d' % icnt
      op = data_util.get_stimulus_response(FLAGS.src_dir, idataset, key)
      stimulus, resp, dimx, dimy, num_cell_types = op

      stimuli.update({key: stimulus})
      responses += resp

  print('# Responses %d' % len(responses))
  stimulus = stimuli[responses[FLAGS.taskid]['stimulus_key']]
  save_filename = ('linear_taskid_%d_piece_%s.pkl' % (FLAGS.taskid, responses[FLAGS.taskid]['piece']))
  print(save_filename)
  learn_lin_embedding(stimulus, np.double(responses[FLAGS.taskid]['responses']),
                      filename=save_filename,
                      lam_l1=0.00001, beta=10, time_window=30,
                      lr=0.01)


  print('DONE!')

def learn_lin_embedding(stimulus, responses, filename,
                        lam_l1=0.01, beta=10, time_window=30, lr=0.01):

  num_cell_types = 2
  dimx = 80
  dimy = 40

  leng = np.minimum(stimulus.shape[0], responses.shape[0])
  resp_short = responses[:leng, :]
  stim_short = np.reshape(stimulus[:leng, :, :], [leng, -1])
  init_A = stim_short[:-4, :].T.dot(resp_short[4:, :])
  init_A_3d = np.reshape(init_A.T, [-1, stimulus.shape[1], stimulus.shape[2]])

  n_cells = responses.shape[1]
  with tf.Session() as sess:

    stim_tf = tf.placeholder(tf.float32,
                             shape=[None, dimx,
                             dimy, time_window]) # batch x X x Y x time_window

    # Linear filter in time
    ttf = tf.Variable(np.ones((time_window, 1)).astype(np.float32))
    stim_tf_2d = tf.reshape(stim_tf, [-1, time_window])
    stim_filtered_2d = tf.matmul(stim_tf_2d, ttf)
    stim_filtered = tf.reshape(stim_filtered_2d, [-1, dimx, dimy, 1])
    stim_filtered = tf.gather(tf.transpose(stim_filtered, [3, 0, 1, 2]), 0)

    # filter in space
    # A = tf.Variable(np.ones((n_cells, dimx, dimy)).astype(np.float32))
    A = tf.Variable(init_A_3d.astype(np.float32))
    A_2d = tf.reshape(A, [n_cells, dimx*dimy])
    
    stim_filtered_perm_2d = tf.reshape(stim_filtered, [-1, dimx*dimy])
    stim_space_filtered_2d = tf.matmul(stim_filtered_perm_2d, tf.transpose(A_2d))
    stim_out = tf.expand_dims(stim_space_filtered_2d, 2)


    responses_anchor_tf = tf.placeholder(dtype=tf.float32,
                                 shape=[None, n_cells, 1],
                                 name='anchor') # batch x n_cells, 1

    responses_neg_tf = tf.placeholder(dtype=tf.float32,
                           shape=[None, n_cells, 1],
                           name='anchor') # batch x n_cells, 1

    from IPython import embed; embed()
    d_s_r_pos = - tf.reduce_sum((stim_out*responses_anchor_tf)**2, [1, 2]) # batch
    d_pairwise_s_rneg = - tf.reduce_sum((tf.expand_dims(stim_out, 1) *
                               tf.expand_dims(responses_neg_tf, 0))**2, [2, 3]) # batch x batch_neg


    difference = (tf.expand_dims(d_s_r_pos/beta, 1) -  d_pairwise_s_rneg/beta) # postives x negatives

    # # log(1 + \sum_j(exp(d+ - dj-)))
    difference_padded = tf.pad(difference, [[0, 0], [0, 1]])
    loss = tf.reduce_sum(beta * tf.reduce_logsumexp(difference_padded, 1), 0)

    accuracy_tf =  tf.reduce_mean(tf.sign(-tf.expand_dims(d_s_r_pos, 1) + d_pairwise_s_rneg))

    lr = 0.01
    train_op = tf.train.AdagradOptimizer(lr).minimize(loss)

    with tf.control_dependencies([train_op]):
      prox_op = tf.assign(A, tf.nn.relu(A - lam_l1) - tf.nn.relu(- A - lam_l1))

    update = tf.group(train_op, prox_op)


    # Now train
    sess.run(tf.global_variables_initializer())
    a_log = []
    for iiter in range(200000):
      stim_batch, resp_batch, resp_batch_neg = sample_datasets.get_batch(stimulus, responses,
                                                         batch_size=100, batch_neg_resp=100,
                                                         stim_history=time_window,
                                                         min_window=10,
                                                         batch_type='train')


      feed_dict = {stim_tf: stim_batch.astype(np.float32),
                   responses_anchor_tf: np.expand_dims(resp_batch, 2).astype(np.float32),
                   responses_neg_tf: np.expand_dims(resp_batch_neg, 2).astype(np.float32)}


      _, l, a = sess.run([update, loss, accuracy_tf], feed_dict = feed_dict)
      a_log += [a]

      if iiter % 10 == 0:
        print(a)
      if iiter % 1000 == 0:
        save_dict = {'A': sess.run(A), 'ttf': sess.run(ttf)}
        pickle.dump(save_dict, gfile.Open(os.path.join(FLAGS.save_folder, filename), 'w'))

    return [sess.run(A), sess.run(ttf)]

if __name__ == '__main__':
  app.run(main)
