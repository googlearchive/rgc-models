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
r"""Utils for end-to-end training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

# for data IO
from absl import gfile  # tf.gfile does NOT work with big mat files.
import scipy.io as sio
import numpy as np, h5py,numpy

# for plotting stuff
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# for embedding
import scipy.stats

FLAGS = flags.FLAGS


def embed_ei_grid(elec_loc, smoothness_sigma=15):
  """Matrix to embed an EI into a 2D image with a given smoothness."""

  # create embedding metric
  n_elec = elec_loc.shape[0]
  spacingx = np.abs(np.diff(np.unique(elec_loc[:, 0]))[0])
  spacingy = np.abs(np.diff(np.unique(elec_loc[:, 1]))[0])

  embed_image = np.zeros((np.int((np.max(elec_loc[:, 0]) -
                                  np.min(elec_loc[:, 0])) / spacingx + 1),
                        np.int((np.max(elec_loc[:, 1]) -
                                np.min(elec_loc[:, 1])) / spacingy) + 1))

  embed_locx = np.unique(elec_loc[:, 0])
  embed_locy = np.unique(elec_loc[:, 1])

  embedding_matrix = np.zeros((embed_image.shape[0],
                               embed_image.shape[1],
                               n_elec))

  elec_loc = elec_loc.astype(np.float32)
  for iembed in range(embed_image.shape[0]):
    for jembed in range(embed_image.shape[1]):
      embed_location = np.array([embed_locx[iembed],
                                 embed_locy[jembed]]).astype(np.float32)
      distances = np.sqrt(np.sum((np.expand_dims(embed_location, 0) - elec_loc) ** 2, 1))
      weights = scipy.stats.norm.pdf(distances / smoothness_sigma)
      weights = weights / np.max(weights)
      embedding_matrix[iembed, jembed, :] = weights

  return embedding_matrix


def clean_rfs(rfs, nbd=1):
  """"Cleans RF by keeping a neighborhood around the maximum value.

  Args:
    rfs: stimx x stimy x # cells.
    nbs: scalar(int), around the maximum pixel.
  """
  n_cells = rfs.shape[2]
  rfs_new = np.zeros_like(rfs)

  for icell in range(n_cells):
    cell_rf = rfs[:, :, icell]

    # find the max position
    i, j = np.where(np.abs(cell_rf) == np.max(np.abs(cell_rf)))
    box_x = [np.maximum(i - nbd, 0), np.minimum(i + nbd + 1, rfs.shape[0])]
    box_y = [np.maximum(j - nbd, 0), np.minimum(j + nbd + 1, rfs.shape[1])]
    rfs_new[box_x[0]: box_x[1],
            box_y[0]: box_y[1], icell] = rfs[box_x[0]: box_x[1],
                                             box_y[0]: box_y[1],
                                             icell]

  return rfs_new


def get_data_retina(piece='2005-04-06-4'):
  """Load data for 1 piece."""

  # Load data
  file_d = h5py.File('/home/bhaishahster/'
                   'Downloads/%s.mat' % piece, 'r')

  stimulus = np.array(file_d.get('stimulus'))
  responses = np.array(file_d.get('response'))
  ei = np.array(file_d.get('ei'))
  cell_type = np.array(file_d.get('cell_type'))

  elec_loc_file = gfile.Open('/home/bhaishahster/'
                             'Downloads/Elec_loc512.mat', 'r')
  data = sio.loadmat(elec_loc_file)
  elec_loc = np.squeeze(np.array([data['elec_locx'], data['elec_locy']])).T

  #########################################################################
  # Process
  stimulus = np.mean(stimulus, 1) - 0.5
  stimulus = stimulus[:-1, :, :]
  stimx = stimulus.shape[1]
  stimy = stimulus.shape[2]
  ei_magnitude = np.sqrt(np.sum(ei**2, 0))
  rfs = np.reshape(stimulus[:-4, :],
                   [-1, stimulus.shape[1]*stimulus.shape[2]]).T.dot(responses[4:, :])
  rfs = np.reshape(rfs, [stimulus.shape[1], stimulus.shape[2], -1])
  rfs = clean_rfs(rfs, nbd=1)

  ei_embedding_matrix = embed_ei_grid(elec_loc, smoothness_sigma=15)
  eix, eiy, _ = ei_embedding_matrix.shape
  n_elec = 512

  # compile data
  data = {'stimulus': stimulus,
          'responses': responses,
          'ei_magnitude': ei_magnitude,
          'rfs': rfs,
          'elec_loc': elec_loc,
          'stimx': stimx,
          'stimy': stimy,
          'eix': eix,
          'eiy': eiy,
          'ei_embedding_matrix': ei_embedding_matrix, 'n_elec': n_elec,
          'cell_type': cell_type}

  return data


def verify_data(data):
  """Verify that data is fine."""

  # extract relevant variables
  ei_magnitude = data['ei_magnitude']
  elec_loc = data['elec_loc']
  rfs = data['rfs']

  # plot EI for few cells chosen randomly
  n_cells = 4 * 4
  random_cells = np.random.randint(0, ei_magnitude.shape[1], n_cells)
  plt.figure()
  for icell in range(n_cells):
    plt.subplot(4, 4, icell+1)
    plt.scatter(elec_loc[:, 0], elec_loc[:, 1],
                ei_magnitude[:, random_cells[icell]])
    plt.axis('Image')
  plt.show()

  # plot receptive fields for a few cells chosen randomly
  n_cells = 4 * 4
  random_cells = np.random.randint(0, rfs.shape[2], n_cells)
  plt.figure()
  for icell in range(n_cells):
    plt.subplot(4, 4, icell+1)
    plt.imshow(rfs[:, :, random_cells[icell]], cmap='gray',
               interpolation='nearest')
    plt.axis('Image')
  plt.show()


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
  resp_batch_neg = np.zeros((batch_neg_resp, resp.shape[1]))
  for isample in range(batch_neg_resp):
    itime = np.random.randint(stim_history, stim.shape[0]-1, 1)
    while np.min(np.abs(itime - random_times)) < min_window:
      itime = np.random.randint(stim_history, stim.shape[0]-1, 1)
    resp_batch_neg[isample, :] = resp[itime, :]


  return stim_batch, resp_batch, ei_mag, resp_batch_neg

def get_test_batch(data, batch_size=100, stim_history=30, min_window=10):
  """Get a batch of training data."""

  stim = data['stimulus']
  resp = data['responses']
  ei_mag = data['ei_magnitude']

  stim_batch = np.zeros((batch_size, stim.shape[1],
                         stim.shape[2], stim_history))
  resp_batch = np.zeros((batch_size, resp.shape[1]))
  resp_batch_neg = np.zeros((batch_size, resp.shape[1]))

  random_times = np.random.randint(stim_history, stim.shape[0]-1, batch_size)
  for isample in range(batch_size):
    itime = random_times[isample]
    stim_batch[isample, :, :, :] = np.transpose(stim[itime:
                                                     itime-stim_history:-1,
                                                     :, :],
                                                [1, 2, 0])
    resp_batch[isample, :] = resp[itime, :]

    # get negatives
    jtime = np.random.randint(stim_history, stim.shape[0]-1, 1)
    while np.min(np.abs(jtime - random_times[isample])) < min_window:
      jtime = np.random.randint(stim_history, stim.shape[0]-1, 1)
    resp_batch_neg[isample, :] = resp[jtime, :]

  return stim_batch, resp_batch, ei_mag, resp_batch_neg


def get_ROC(distances_pos, distances_neg):
  all_distances = np.append(distances_pos, distances_neg)

  # compute TPR, FPR for ROC
  TPR_log = []
  FPR_log = []

  for iprc in np.arange(0,100,1):
    ithr = np.percentile(all_distances, iprc)
    TP = np.sum(distances_pos <= ithr)
    FP = np.sum(distances_neg <= ithr)
    FN = np.sum(distances_pos > ithr)
    TN = np.sum(distances_neg > ithr)

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    TPR_log += [TPR]
    FPR_log += [FPR]

  return TPR_log, FPR_log


def get_neighbormat(mask_matrix, nbd=1):
  """Get neighbor mapping."""

  mask = np.ndarray.flatten(mask_matrix)>0

  x = np.repeat(np.expand_dims(np.arange(mask_matrix.shape[0]), 1), mask_matrix.shape[1], 1)
  y = np.repeat(np.expand_dims(np.arange(mask_matrix.shape[1]), 0), mask_matrix.shape[0], 0)
  x = np.ndarray.flatten(x)
  y = np.ndarray.flatten(y)

  idx = np.arange(len(mask))
  iidx = idx[mask]
  xx = np.expand_dims(x[mask], 1)
  yy = np.expand_dims(y[mask], 1)

  distance = (xx - xx.T)**2 + (yy - yy.T)**2
  neighbor_mat = np.double(distance <= nbd)

  return neighbor_mat

  # wts = 1/(neighbor_mat.dot(np.abs(K)) + eps)
  # K = np.maximum(K - (wts*lam_l1), 0) - np.maximum(- K - (wts*lam_l1), 0)
