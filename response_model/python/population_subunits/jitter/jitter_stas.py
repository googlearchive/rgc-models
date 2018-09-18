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
"""Jitter STAs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import sys
import os.path
import collections
import tensorflow as tf
from absl import app
from absl import flags
from absl import gfile
from tensorflow.python.profiler.model_analyzer import PrintModelAnalysis
import cPickle as pickle
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import ndimage
import retina.response_model.python.jitter.distributed.jitter_model as jitter_model
import retina.response_model.python.jitter.distributed.get_data_mat as get_data_mat
import random

# setup dataset
flags.DEFINE_string('data_location',
                    '/home/retina_data/datasets/2016-04-21-1/data006(2016-04-21-1_data006_data006)/',
                    'where to take data from?')

flags.DEFINE_integer('n_chunks',1793, 'number of data chunks') # should be 216

flags.DEFINE_integer('n_cells', 1, 'number of cells')
FLAGS = flags.FLAGS


def main(unused_argv):

  '''
  print('start code')
  response, cids_select, n_cells, tc_select, tc_mean = get_data_mat.setup_dataset()
  selected_cid = cids_select[23]
  cid_idx = 23
  print('data summary loaded')
  print('selected_cid %d' % selected_cid)
  compute_sta(selected_cid, response, cid_idx)
  print('sta computed')
  '''
  import pickle;
  sta_frame = pickle.load(open('/home/bhaishahster/Downloads/jitter_sta.pkl','rb'))
  print(np.shape(sta_frame))
  for ichannel in range(3):
    plt.imshow(sta_frame[:,:,ichannel]);
    plt.show()
    plt.draw()

def compute_sta(cell_id, response, cid_idx, delay=4):

  # compute STA for one cell - load a chunk, compute STA, repeat
  sta_frame = np.zeros((640, 320, 3))
  for ichunk in np.arange(1000)+1:
    print('Loading chunk: %d' % ichunk)
    # get a stimulus chunk
    stim_path = FLAGS.data_location + 'Stimulus/'
    stim_file = sio.loadmat(gfile.Open(stim_path+'stim_chunk_' + str(ichunk) + '.mat'))
    chunk_start = np.squeeze(stim_file['chunk_start'])
    chunk_end = np.squeeze(stim_file['chunk_end'])
    jump = stim_file['jump']
    stim_chunk = stim_file['stimulus_chunk']
    stim_chunk = np.transpose(stim_chunk, [3,0,1,2])
    print(np.shape(stim_chunk))

    print(chunk_start, chunk_end)
    # find non-zero time points and
    for itime in np.arange(29, chunk_end - chunk_start + 1):
      if response[chunk_start + itime - 1, cid_idx] > 0:
        sta_frame = sta_frame +  np.squeeze(stim_chunk[itime - delay, :, :, :])

  # compute STA
  sta_frame = sta_frame / np.sum(response[:,cid_idx])
  import pdb; pdb.set_trace()

  # plot STA
  plt.imshow(sta_frame[:,:,2])
  plt.show()
  plt.draw()

if __name__ == '__main__':
  app.run()
