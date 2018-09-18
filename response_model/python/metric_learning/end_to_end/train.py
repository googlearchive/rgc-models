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
r"""Learn the objective for end to end prosthesis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import app
import numpy as np, h5py,numpy

# for plotting stuff
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# for embedding stuff
import retina.prosthesis.end_to_end.embedding0 as em

# utils
import retina.prosthesis.end_to_end.utils as utils
FLAGS = tf.app.flags.FLAGS


def main(unused_argv=()):

  # load default data
  data = utils.get_data_retina()

  # verify utils
  utils.verify_data(data)

  #########################################################################

  ## Try some architecture.

  embedx = 20
  embedy = 10
  stim_history = 30
  batch_size = 1000
  batch_neg_resp = 100
  beta = 10
  is_training = True
  with tf.Session() as sess:
    ei_embedding, ei_tf = em.embed_ei(embedx, embedy, data['eix'],
                                      data['eiy'],
                                      data['n_elec'],
                                      data['ei_embedding_matrix'],
                                      is_training=is_training)

    responses_embedding, responses_tf = em.embed_responses(embedx, embedy,
                                                           ei_embedding,
                                                           is_training=
                                                           is_training)

    stimulus_embedding, stim_tf = em.embed_stimulus(embedx, embedy,
                                                    data['stimx'],
                                                    data['stimy'],
                                                    stim_history=stim_history,
                                                    is_training=is_training)

    responses_embedding_pos = tf.gather(responses_embedding,
                                        np.arange(batch_size).astype(np.int))

    responses_embedding_neg = tf.gather(responses_embedding,
                                        np.arange(batch_size,
                                                  batch_neg_resp +
                                                  batch_size).astype(np.int))
    d_pos = tf.reduce_sum((stimulus_embedding - responses_embedding_pos)**2,
                          [1, 2])
    d_neg_pairs = tf.reduce_sum((tf.expand_dims(responses_embedding_pos, 1) -
                                 tf.expand_dims(responses_embedding_neg, 0))**2,
                                [2, 3])
    d_neg = -tf.reduce_logsumexp(- d_neg_pairs / beta, 1)
    loss = tf.reduce_sum(tf.nn.relu(d_pos - d_neg + 1))

    train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
    sess.run(tf.global_variables_initializer())

    from IPython import embed; embed()

    for _ in range(10000):
      stim_batch, resp_batch, ei_batch, resp_batch_neg = get_train_batch(data,
                                                         batch_size=batch_size,
                                                         batch_neg_resp=batch_neg_resp,
                                                         stim_history=stim_history)
      feed_dict = {ei_tf: ei_batch,
                   responses_tf: np.append(resp_batch, resp_batch_neg, 0),
                   stim_tf: stim_batch}
      loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
      print(loss_np)

    ## NOTE : currently, everything goes to zero.


def get_train_batch(data, batch_size=100, batch_neg_resp=100,
                    stim_history=30, min_window=10):
  """Get a batch of training data."""

  stim = data['stimulus']
  resp = data['responses']
  ei_mag = data['ei_magnitude']

  stim_batch = np.zeros((batch_size, stim.shape[1],
                         stim.shape[2], stim_history))
  resp_batch = np.zeros((batch_size, resp.shape[1]))

  random_times = np.random.randint(stim_history, stim.shape[0]-1, batch_size)
  for isample in range(batch_size):
    itime = random_times[isample]
    stim_batch[isample, :, :, :] = np.transpose(stim[itime:
                                                     itime-stim_history:-1,
                                                     :, :],
                                                [1, 2, 0])
    resp_batch[isample, :] = resp[itime, :]

  # get negative responses.
  resp_batch_neg = np.zeros((batch_size, resp.shape[1]))
  for isample in range(batch_neg_resp):
    itime = np.random.randint(stim_history, stim.shape[0]-1, 1)
    while np.min(np.abs(itime - random_times)) < min_window:
      itime = np.random.randint(stim_history, stim.shape[0]-1, 1)
    resp_batch_neg[isample, :] = resp[itime, :]


  return stim_batch, resp_batch, ei_mag, resp_batch_neg

  # sample EIs - return ei_mag




if __name__ == '__main__':
   app.run()
