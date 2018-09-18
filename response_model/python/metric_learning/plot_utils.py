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
"""Utils for plotting results. Mostly used in colab notebook."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from tensorflow.python.platform import gfile

import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pickle

FLAGS = flags.FLAGS
import numpy as np


# Local utility functions
def plot_responses(response, centers, ax, alphas=None, xlims=[18, 27], ylims=[3, 12]):
  """Plots circles which are filled for cells that fire."""
  n_cells = centers.shape[0]
  for icell in range(n_cells):

    cc = [(centers[icell, 0] - 2)/2, (centers[icell, 1] - 2)/2]

    if response[icell]>0:
      if alphas is None:
        alpha = 0.5
      else:
        alpha = alphas[icell]

      circle = plt.Circle(cc, 0.7, color='r', alpha=alpha, linewidth=2, ec='k')
      ax.add_artist(circle)



    circle = plt.Circle(cc, 0.7, color='k', fill=False, ec='k', linewidth=2)
    ax.add_artist(circle)


  plt.xlim(xlims)
  plt.ylim(ylims)

  plt.xticks([])
  plt.yticks([])
  ax.set_axis_bgcolor('white')

  ax.set_aspect('auto', 'datalim')


def time_filter_chunk(stim_chunk, ttf):
  """Filters in time.

  Args :
      stim_chunk : stimulus chunk of shape (Time x X x Y).
      ttf : time course (Time).
  Returns :
      stim_filtered : time filtered stimulus

  """

  stim_filtered  = np.sum(np.transpose(stim_chunk, [1, 2, 0]) * ttf, 2)
  return stim_filtered

def plot_stimulus(stimulus, ttf, probe_time):
  """Plot stimulus @ probe_time, filtered in time

  Args :
      stimulus : all of stimulus (Time x dimx x dimy)
      ttf : temporal filtering (30 x 1)
      probe_time : make chunk before probe time
  """

  stim_chunk = stimulus[probe_time-30: probe_time, :, :]
  # time filter
  stim_filtered = time_filter_chunk(stim_chunk, ttf)

  wk = np.array([[1, 1],
                 [1, 1]])
  from scipy import signal
  stim_filtered = signal.convolve2d(stim_filtered, wk, 'same')


  plt.imshow(stim_filtered, interpolation='nearest', cmap='gray')
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])


def compute_STAs(repeats, stimulus, tlen=30):
  # Compute STAs

  rr = np.mean(np.expand_dims(np.expand_dims(repeats.astype(np.float32), 2), 3), 0)
  ss = stimulus.astype(np.float32)
  stas = np.zeros((ss.shape[1], ss.shape[2], tlen, rr.shape[-1]))
  rf = np.zeros((ss.shape[1], ss.shape[2], rr.shape[-1]))

  for icell in range(rr.shape[-1]):
    print(icell)
    for itlen in range(tlen):
      stas[:, :, itlen, icell] = np.mean(ss[:-itlen-1,:,:]*rr[itlen+1:,:,:,icell], 0)

    rf[:, :, icell] = stas[:, :, 4, icell]

  return stas

def get_time_courses(stas, peak_time=-4):
  # Find time courses for cells
  ttf = []
  for icell in range(stas.shape[-1]):
    a = np.abs(stas[:, :, 4, icell])
    i, j = np.unravel_index(np.argmax(a), a.shape)
    ttf += [np.mean(np.mean(stas[i-1:i+2, j-1:j+2, :, icell], 1), 0)[::-1]]

  ttf_array = np.array(ttf)
  signs = np.sign(ttf_array[:, peak_time])
  ttf_array = np.array(ttf).T*signs
  ttf_use = np.mean(ttf_array, 1)
  plt.plot(ttf_use)
  plt.show()

  return ttf_use

# Learn linear reconstruction filter
def filter_stimulus(stimulus, ttf_use):
  stimulus_filtered = 0*stimulus
  T = stimulus.shape[0]

  # filter stimulus
  stalen = len(ttf_use)
  for idim in range(stimulus.shape[1]):
    for jdim in range(stimulus.shape[2]):
      xx = np.zeros((stalen-1+T))
      xx[stalen-1:]=np.squeeze(stimulus[:, idim, jdim])

      stimulus_filtered[:, idim, jdim] = np.convolve(xx,ttf_use,mode='valid')

  return stimulus_filtered

def learn_linear_decoder_time_filtered(stimulus, response):
  T, dimx, dimy = stimulus.shape
  n_cells = response.shape[1]
  decoder = np.zeros((dimx, dimy, n_cells + 1))

  response = np.append(response, np.ones((stimulus.shape[0], 1)), 1)
  X = np.linalg.inv(response.T.dot(response)).dot(response.T)
  for idimx in range(dimx):
    for idimy in range(dimy):

      y = stimulus[:, idimx, idimy]
      A = (X.dot(y))
      decoder[idimx, idimy, :] = A

  return decoder

def learn_linear_decoder_unfiltered(stimulus, repeats, ttf_use, dimx=20, dimy=40):
  stim_filtered = filter_stimulus(stimulus, ttf_use[::-1])

  reps = np.reshape(repeats, [-1, repeats.shape[-1]])
  stims = np.repeat(np.expand_dims(stim_filtered, 0), repeats.shape[0], 0)
  stims = np.reshape(stims, [-1, 20, 40])
  decoder  = learn_linear_decoder_time_filtered(stims, reps)
  return decoder, stim_filtered



def plot_reconstruct_stimulus(response, rf, xlims=[18, 27], ylims=[3, 12]):
  """Reconstruct stimulus using responses and receptive fields """

  ss = 0 * rf[:, :, 0]
  for icell in range(rf.shape[-1] -1 ):
    ss += rf[:, :, icell]*response[icell]
  ss += rf[:, :, -1]

  plt.imshow(ss, interpolation='nearest', cmap='gray', clim=(-0.01, 0.01))
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.xlim(xlims)
  plt.ylim(ylims)

  print(plt.gci().get_clim())
  #plt.xlim([18*2, 27*2])
  #plt.ylim([3*2, 12*2])

  return ss


# reconstruct stimulus
def reconstruct_stimulus(response, rf):
  """Reconstruct stimulus using responses and receptive fields """

  ss = 0 * rf[:, :, 0]
  for icell in range(rf.shape[-1] - 1):
    ss += rf[:, :, icell]*response[icell]
  ss += rf[:, :, -1]

  return ss


def plot_electrodes(elec_loc, stim_elec, elec_range):
  plt.plot(elec_loc[:, 0], elec_loc[:, 1], '.', markersize=10)
  plt.axis('image')
  plt.xlim(elec_range[0])
  plt.ylim(elec_range[1])
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.plot(elec_loc[stim_elec, 0], elec_loc[stim_elec, 1], 'r.', markersize=20)
  plt.title('Stimulation electrode %d' % stim_elec)
