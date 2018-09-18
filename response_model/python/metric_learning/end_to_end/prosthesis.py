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
r"""Analyse joint embedding for a prosthesis.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import gfile
import os.path
import numpy as np, h5py,numpy
import retina.response_model.python.metric_learning.end_to_end.data_util as data_util
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.io as sio
import retina.response_model.python.metric_learning.end_to_end.response_embedding as resp

FLAGS = tf.app.flags.FLAGS


def stimulate(sr_graph, sess, file_name, dimx, dimy):

  piece = '2015-11-09-3.mat'

  # saving filename.
  save_analysis_filename = os.path.join(FLAGS.save_folder,
                                        file_name + '_prosthesis.pkl')
  save_dict = {}

  # load dictionary.
  dict_dir = '/home/bhaishahster/Downloads/dictionary'
  dict_src = os.path.join(dict_dir, piece)
  # _, dictionary, cellID_list, EA, elec_loc = load_single_elec_stim_data(gfile.Open(dict_src, 'r'))
  _, dictionary, cellID_list, EA, elec_loc = load_single_elec_stim_data(dict_src)
  dictionary = dictionary.T

  # Load cell properties
  cell_data_dir = '/home/bhaishahster/Downloads/rgb-8-1-0.48-11111'
  cell_file = os.path.join(cell_data_dir, piece)
  data_cell = sio.loadmat(gfile.Open(cell_file, 'r'))
  data_util.process_dataset(data_cell, dimx=80, dimy=40, num_cell_types=2)

  # Load stimulus
  data = h5py.File(os.path.join(cell_data_dir, 'stimulus.mat'))
  stimulus = np.array(data.get('stimulus')) - 0.5

  # Generate targets

  # random 100 samples
  t_len = 100
  stim_history = 30
  stim_batch = np.zeros((t_len, stimulus.shape[1],
                         stimulus.shape[2], stim_history))
  for isample, itime in enumerate(np.random.randint(0, stimulus.shape[0], t_len)):
    stim_batch[isample, :, :, :] = np.transpose(stimulus[itime: itime-stim_history:-1, :, :], [1, 2, 0])

  from IPython import embed; embed()

  # Use regression to decide dictionary elements
  regress_dictionary(sr_graph, stim_batch, dictionary, 10, dimx, dimy, data_cell)


  # Select stimulation pattern
  dict_sel_np_logit, r_s, dictionary, d_log = get_optimal_stimulation(stim_batch,
                                                               sr_graph,
                                                               dictionary,
                                                               data_cell, sess)

  save_dict.update({'dict_sel': dict_sel_np_logit,
                    'resp_sample': r_s,
                    'dictionary': dictionary,
                    'd_log': d_log})

  pickle.dump(save_dict, gfile.Open(save_analysis_filename, 'w'))


def regress_dictionary(sr_graph, stim_batch, dictionary, T, dimx, dimy, data_cell):

  # Constrained Regression method
  T = np.float32(T)
  t_len = stim_batch.shape[0]

  dict_choice_var = tf.Variable(np.random.randn(dictionary.shape[1] + 1, t_len).astype(np.float32), name='dict_choice')
  dict_choice = T * tf.nn.softmax(dict_choice_var, 0)
  dict_choice = dict_choice[:-1, :]

  dictionary_tf = tf.constant(dictionary.astype(np.float32))
  responses_tf = tf.matmul(dictionary_tf, dict_choice)

  # stim_embed = sr_graph.stim_embed
  is_training = True
  resp_embed = resp.Convolutional2(time_window=1,
                                   layers=FLAGS.resp_layers,
                                   batch_norm=FLAGS.batch_norm,
                                   is_training=is_training,
                                   reuse_variables=True,
                                   num_cell_types=2,
                                   dimx=dimx, dimy=dimy,
                                   responses_tf=tf.expand_dims(tf.transpose(responses_tf, [1, 0]), 2))

  loss = tf.reduce_sum((sr_graph.stim_embed - resp_embed.responses_embed) ** 2)
  train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss, var_list=[dict_choice_var])

  '''
  with tf.control_dependencies([train_step]):
    positive_dict = tf.assign(dict_choice, tf.nn.relu(dict_choice))
    with tf.control_dependencies([positive_dict]):
      scale_dict = tf.assign(dict_choice, T * dict_choice / tf.maximum(T, tf.reduce_sum(dict_choice)))

  train_op = tf.group(train_step, positive_dict, scale_dict)
  '''
  init_new_vars_op = tf.variables_initializer([dict_choice_var])
  sr_graph.sess.run(init_new_vars_op)

  feed_dict = {sr_graph.stim_tf: stim_batch,
               resp_embed.map_cell_grid_tf:data_cell['map_cell_grid'] ,
               resp_embed.cell_types_tf: data_cell['ctype_1hot'],
               resp_embed.mean_fr_tf: data_cell['mean_firing_rate']}

  if hasattr(resp_embed, 'dist_nn'):
      dist_nn = np.array([data_cell['dist_nn_cell_type'][1],
                          data_cell['dist_nn_cell_type'][2]]).astype(np.float32)
      feed_dict.update({resp_embed.dist_nn: dist_nn})

  for iiter in range(10000):
    _, l_np = sr_graph.sess.run([train_op, loss], feed_dict=feed_dict)
    print(l_np)



def get_optimal_stimulation(stim_target, sr_graph, dictionary, data, sess, n_stims=50):
  """Select optimal stimulation pattern.

  Args:
    n_stims: number of stimulations per target.
  """

  # from IPython import embed; embed()

  n_samples = stim_target.shape[0]
  n_dict = dictionary.shape[0]

  # dict_sel_np_logit is un-normalized log-probability
  dict_sel_np_logit = np.log(np.random.rand(n_samples, n_stims, n_dict))

  step_sz = 1
  eps = 1e-1
  dist_prev = np.inf
  d_log = []

  # gumble softmax for sampling relaxed responses
  temp_tf = tf.placeholder(tf.float32, shape=())
  dict_sel_tf = tf.placeholder(tf.float32, shape=(n_samples, n_stims, n_dict))

  # select dictionary elements
  dict_sample_logit = dict_sel_tf + sample_gumbel(tf.shape(dict_sel_tf))
  dict_sel = tf.nn.softmax(dict_sample_logit / temp_tf, dim=2) # n_samples, n_stims, n_dict

  # dictionary
  dictionary_tf = tf.constant(dictionary.astype(np.float32), name='dictionary')
  dict_sel_2d = tf.reshape(dict_sel, [n_samples * n_stims, -1])
  #cell_response_prob_2d = tf.matmul(dict_sel_2d, dictionary_tf) # n_samples, n_stims, n_cells
  #cell_response_prob = tf.reshape(cell_response_prob_2d, [n_samples, n_stims, -1])

  cell_response_prob_2d = tf.matmul(dict_sel_2d, dictionary_tf)
  cell_response_prob = tf.reshape(cell_response_prob_2d, [n_samples, n_stims, -1])
  response = cell_response_prob

  #c_r_logit = tf.log(cell_response_prob) + sample_gumbel(tf.shape(cell_response_prob))
  #response = tf.nn.softmax(c_r_logit / temp_tf, dim=2)
  resp_batch = tf.expand_dims(tf.reduce_sum(response, 1), 2)

  with tf.control_dependencies([resp_batch]):
    grad_dict_sel = tf.gradients(resp_batch, dict_sel_tf)[0]

  grad_resp = tf.gradients(sr_graph.d_s_r_pos, sr_graph.anchor_model.responses_tf)[0]
  grad_resp_remove_last = tf.gather(tf.transpose(grad_resp, [2, 0, 1]), 0)

  temperature = 10

  for iiter in range(1000):
    if iiter % 30 == 0:
      temperature *= 0.9
    # sample response, dr/dtheta
    resp_batch_np, grad_dict_sel_np = sess.run([resp_batch, grad_dict_sel],
                                            feed_dict ={dict_sel_tf : dict_sel_np_logit,
                                                        temp_tf : temperature})

    # d(distance)/dr
    feed_dict = {sr_graph.stim_tf: stim_target,
                 sr_graph.anchor_model.map_cell_grid_tf: data['map_cell_grid'],
                 sr_graph.anchor_model.cell_types_tf: data['ctype_1hot'],
                 sr_graph.anchor_model.mean_fr_tf: data['mean_firing_rate'],
                 sr_graph.anchor_model.responses_tf: resp_batch_np,
                 }
    dist_np, grad_resp_np = sr_graph.sess.run([sr_graph.d_s_r_pos, grad_resp_remove_last], feed_dict=feed_dict)

    # update theta
    grad_resp_np_back = np.expand_dims(grad_resp_np.dot(dictionary.T), 1)
    dict_sel_np_logit = dict_sel_np_logit - step_sz * grad_resp_np_back * grad_dict_sel_np

    '''
    # get scale firing rates!
    print('scaling mean firing rate')
    prob_np = np.exp(dict_sel_np_logit)
    prob_np = prob_np / (1 + prob_np)

    prob_np = np.minimum(prob_np, 1 - 1e-6)
    prob_np = np.maximum(prob_np, 1e-6)

    theta_np = np.log(prob_np / (1 - prob_np))
    '''
    if np.sum(np.abs(dist_prev - dist_np)) < eps:
      break
    print(iiter, np.sum(dist_np), np.sum(np.abs(dist_prev - dist_np)))
    dist_prev = dist_np
    d_log += [np.sum(dist_np)]

    '''
    plt.ion()
    plt.cla()
    plt.plot(d_log)
    plt.pause(0.05)
    '''
  # sample responses
  r_s = []
  for  sample_iters in range(100):
    r_ss = sess.run(resp_batch, feed_dict ={dict_sel_tf : dict_sel_np_logit,
                                           temp_tf : 1e-6})
    r_s += [r_ss]
  r_s = np.double(np.array(r_s) > 0.5).squeeze()

  return dict_sel_np_logit, r_s, dictionary, d_log


def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def load_single_elec_stim_data(filename):
    '''Load spike sorting results for single electrode stimulation'''
    file=h5py.File(filename, 'r')
    stas = np.array(file.get('stas'))
    cellID_list = np.array(file.get('cellID_list'))
    sigmoid_params1 = np.array(file.get('sigmoid_params1')).T
    sigmoid_params2 = np.array(file.get('sigmoid_params2')).T
    cell_elec = np.double(sigmoid_params1!=0)
    ncells  = cellID_list[0]
    elec_loc = np.array(file.get('elec_loc'))


    xx = np.expand_dims(np.expand_dims(np.arange(1,39+1),1),2)
    cell_elec = np.double(sigmoid_params2!=0)
    cell_act = (1/(1+np.exp(-(xx*sigmoid_params2.T + sigmoid_params1.T)))) * cell_elec.T
    print(cell_act.shape)
    dictionary = np.reshape(cell_act,[-1,cell_act.shape[-1]])

    electrodes = np.repeat(np.expand_dims(np.arange(1, 512+1), 0), 39, 0)
    electrodes = np.expand_dims(np.ndarray.flatten(electrodes),1)

    amplitudes = np.repeat(np.expand_dims(np.arange(1, 39+1), 1), 512, 1)
    amplitudes = np.expand_dims(np.ndarray.flatten(amplitudes),1)

    EA = np.append(electrodes, amplitudes, axis = 1)
    print(cell_act.shape, cell_elec.shape, dictionary.shape, xx.shape)
    #from IPython import embed
    #embed()
    return stas, dictionary, cellID_list, EA, elec_loc

