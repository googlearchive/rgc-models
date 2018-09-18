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
"""Analysis utils"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf
import numpy as np
import sklearn.manifold as manifold
import retina.response_model.python.metric_learning.end_to_end.sample_datasets as sample_datasets
from sklearn.decomposition import PCA

FLAGS = flags.FLAGS


def sample_stimuli(stimuli, n_stims_per_type, t_min, t_max,
                   dimx=80, dimy=40, tlen = 30):

  # Jointly embed NSEM and WN stimuli
  # 100 random test examples for NSEM and WN each
  stimulus_test = np.zeros((n_stims_per_type * len(stimuli.keys()), dimx,
                            dimy, tlen))
  stim_type = np.zeros(n_stims_per_type * len(stimuli.keys()))

  icnt = 0
  itype = 0
  for stim_key in stimuli.keys():
    stim_test = stimuli[stim_key][t_min: t_max, :, :]
    for itarget in range(n_stims_per_type):
      rand_sample = np.random.randint(30, stim_test.shape[0])
      stim_sample = stim_test[rand_sample: rand_sample-30:-1, :, :]
      stim_sample = np.expand_dims(np.transpose(stim_sample, [1, 2, 0]), 0)
      stim_sample = sample_datasets.verify_stimulus_dimensions(stim_sample, dimx=80, dimy=40)
      stimulus_test[itarget, :, :, :] = stim_sample
      stim_type[icnt] = itype
      icnt += 1
    itype += 1

  return stimulus_test, stim_type


def embed_stimuli(stimulus_test, sr_graph, embed_batch=100):

  embed_x = sr_graph.stim_embed.shape[-3].value
  embed_y = sr_graph.stim_embed.shape[-2].value
  embed_t = sr_graph.stim_embed.shape[-1].value

  rand_perm = np.random.permutation(np.arange(stimulus_test.shape[0]))
  stim_test_embed = np.zeros((stimulus_test.shape[0], embed_x, embed_y, embed_t))
  for istart in np.arange(0, stimulus_test.shape[0], embed_batch):
    iend = np.minimum(istart + embed_batch, stimulus_test.shape[0])
    t_idx = rand_perm[np.arange(istart, iend)]
    stim_test_embed[t_idx, :, :, :] = sr_graph.sess.run(sr_graph.stim_embed,
                                        feed_dict = {sr_graph.stim_tf:
                                        stimulus_test[t_idx, :, :, :].astype(np.float32)})
  stim_test_embed = stim_test_embed.squeeze()
  return stim_test_embed


def get_tsne(input_, n_components=2):
  """Get tSNE embedding of embeddings.

  Args:
    input_ : Input - numpy array (Batch, X, Y)
    n_components : Number of components - scalar
  Returns :
    Array : t-SNE embeddings - numpy array  (Batch, n_components)
  """

  model = manifold.TSNE(n_components=n_components)
  return model.fit_transform(input_)


def get_pca(input_, n_components=2):
  """Get PCA embedding.

  Args:
    input_ : Embedded samples (samples x embedded dimension).
    n_components : number of principal dimensions.

  Returns :
    array : PCA embedding (samples x n_components).
  """

  pca = PCA(n_components=2)
  return pca.fit_transform(input_)

def get_linear_discriminator(resp_embed, labels, n_components=2):

  # linear discriminant analysis based on retina

  resp_embed_flat = np.reshape(resp_embed, [resp_embed.shape[0], -1])
  embed_dim  = resp_embed_flat.shape[1]
  s_w = np.zeros((embed_dim, embed_dim))  # within class variance
  class_means = np.zeros((np.unique(labels).shape[0], embed_dim))
  for iilabel, ilabel in enumerate(np.unique(labels)):
    print(ilabel)
    embed_class = resp_embed_flat[labels == ilabel, :]
    s_w += np.cov(embed_class.T)
    class_means[iilabel, :] = np.mean(embed_class, 0)
  s_b = np.cov(class_means.T)  # between class variance

  SS = np.linalg.pinv(s_w).dot(s_b)
  w, v = np.linalg.eig(SS)
  P = np.abs(v[:, :n_components])

  resp_embed_2d = resp_embed_flat.dot(P)
  # plt.scatter(resp_embed_2d[:, 0], resp_embed_2d[:, 1], c=labels)
  return resp_embed_2d

def transform_stimulus(stimulus, levels, transform_type):
  """Transform stimulus of a certain type  with different levels
  
  Args: 
    stimulus : stimulus (Batch, X, Y, t_len)
    levels : different levels of transformation (list)
    transform_type : string indicating the type of transformation.
                      'luminance', 'contrast' or 'translate'.

  Returns:
    stimulus_transformed : collection of stimuli with changed luminance (Batch x # levels, X, Y, t_len)
    original_idx : mapping each transformed stimulus to input stimulus array (Batch x # levels)
  """
  
  if transform_type not in ['luminance', 'contrast', 'translate']:
    raise ValueError('Transform type not supported')

  stimulus_transformed = []
  original_idx = []
  
  # Subtract the mean
  stimulus_mean = np.mean(stimulus)
  stimulus_normalized = stimulus - stimulus_mean

  for ilevel in levels:
    print(transform_type, ilevel)
    if transform_type == 'luminance':
      stimulus_transformed += [stimulus_normalized + ilevel]
   
    if transform_type == 'contrast':
      stimulus_transformed += [stimulus_normalized * ilevel]
    
    if  transform_type == 'translate':
      stimulus_transformed += [np.roll(np.roll(stimulus_normalized, np.int(ilevel), axis=1), np.int(ilevel), axis=2)]

    original_idx += [np.arange(stimulus.shape[0])]

  stimulus_transformed = np.concatenate(stimulus_transformed, axis=0)
  original_idx = np.concatenate(original_idx, axis=0)
  
  # Add back the mean.
  stimulus_transformed = stimulus_transformed + stimulus_mean
  
  return stimulus_transformed, original_idx


def ROC(distances_pos, distances_neg):
  """Compute ROC curve."""

  all_distances = np.append(distances_pos, distances_neg)
  precision_log = []
  recall_log = []
  F1_log = []
  TPR_log = []
  FPR_log = []

  for iprc in np.arange(0,100,1):
    ithr = np.percentile(all_distances, iprc)
    TP = np.sum(distances_pos <= ithr)
    FP = np.sum(distances_neg <= ithr)
    FN = np.sum(distances_pos > ithr)
    TN = np.sum(distances_neg > ithr)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    precision_log += [precision]
    recall_log += [recall]
    F1_log += [F1]
    TPR_log += [TPR]
    FPR_log += [FPR]

  return precision_log, recall_log, F1_log, FPR_log, TPR_log

def predict_responses_ln(stimulus, k, b, ttf, n_trials=1):
  '''Predict responses using a linear(tf + spatial filter),exp, poiss model.'''

  stimx, stimy, n_cells = k.shape

  with tf.Graph().as_default():
    with tf.Session() as sess:

      stim_tf = tf.placeholder(tf.float32) # T x stimx x stimy
      k_tf = tf.constant(k.astype(np.float32))
      b_tf = tf.constant(np.float32(b))

      stim_tf_flat = tf.reshape(stim_tf, [-1, stimx * stimy])

      # convolve each pixel in time.
      ttf_tf = tf.constant(ttf)

      tfd = tf.expand_dims
      ttf_4d = tfd(tfd(tfd(ttf_tf, 1), 2), 3)
      stim_pad = tf.pad(stim_tf_flat, np.array([[29, 0], [0, 0]]).astype(np.int))
      stim_4d = tfd(tfd(tf.transpose(stim_pad, [1, 0]), 2), 3)
      stim_smooth = tf.nn.conv2d(stim_4d, ttf_4d, strides=[1, 1, 1, 1], padding="VALID")

      stim_smooth_2d = tf.squeeze(tf.transpose(stim_smooth, [2, 1, 0, 3]))

      k_tf_flat = tf.reshape(k_tf, [stimx*stimy, n_cells])

      lam_raw = tf.matmul(stim_smooth_2d, k_tf_flat) + b_tf
      lam = tf.exp(lam_raw) #tf.nn.softplus(lam_raw)

      sess.run(tf.global_variables_initializer())
      [lam_np, lam_raw_np] = sess.run([lam, lam_raw], feed_dict={stim_tf: stimulus.astype(np.float32)})

      # repeat lam_np for number of trials
      lam_np = np.repeat(np.expand_dims(lam_np, 0), n_trials, axis=0)
      lam_raw_np = np.repeat(np.expand_dims(lam_raw_np, 0), n_trials, axis=0)
      spikes = np.random.poisson(lam_np)

  return spikes, lam_np

def plot_raster(spikes, color='b'):
  '''Spikes: trials x T'''
  for itrial in range(spikes.shape[0]):
    r = np.where(spikes[itrial, :] > 0)[0]
    plt.eventplot(r, lineoffsets=itrial, colors=color)
