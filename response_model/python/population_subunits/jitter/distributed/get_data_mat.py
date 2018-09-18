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
"""Get jitter data (stimulus and response) in batches
"""

import sys
import os.path
import tensorflow as tf
from absl import app
from absl import flags
from absl import gfile
import cPickle as pickle
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import ndimage
import random

FLAGS = flags.FLAGS


response = np.array([])
test_chunks = np.array([])
train_chunks = np.array([])
train_counter = np.array([])
test_counter = np.array([])

def init_chunks(n_chunks):
  # initialize the chunks, called at the beginning of the code.
  global test_chunks
  global train_chunks
  global train_counter
  global response
  global test_counter

  random_chunks = np.arange(n_chunks)+1
  test_chunks = random_chunks[0:20]
  train_chunks = random_chunks[22:]
  train_chunks = np.random.permutation(train_chunks)
  train_counter = 0
  test_counter = 0
  response, cids_select, n_cells, tc_select, tc_mean = setup_dataset()
  return tc_mean


def get_stim_resp(data_type='train', num_chunks=1):
  # this function gets you the training and testing chunks!
  # permute chunks and decide the training and test chunks
  global train_chunks
  global test_chunks
  global train_counter
  global test_counter
  global response

  #print(FLAGS.data_location)

  def get_stimulus_batch(ichunk):
    stim_path = FLAGS.data_location + 'Stimulus/'
    stim_file = sio.loadmat(gfile.Open(stim_path+'stim_chunk_' + str(ichunk) + '.mat'))
    chunk_start = np.squeeze(stim_file['chunk_start'])
    chunk_end = np.squeeze(stim_file['chunk_end'])
    jump = stim_file['jump']
    stim_chunk = stim_file['stimulus_chunk']
    stim_chunk = np.transpose(stim_chunk, [3,0,1,2])
    return stim_chunk, chunk_start, chunk_end

  if(data_type=='test'):
    print('Loading test')
    #print(test_chunks)
    if test_counter>=test_chunks.shape[0]:
      test_counter = 0
    start_chunk_id = np.squeeze(np.array(test_chunks[test_counter]).astype('int'))
    chunk_ids = [start_chunk_id]
    test_counter = test_counter + 1

  if(data_type=='train'):
    #print('Loading train')
    if(train_counter>=train_chunks.shape[0]):
      train_chunks = np.random.permutation(train_chunks)
      train_counter=0
    start_chunk_id = np.squeeze(np.array(train_chunks[train_counter]).astype('int'))
    if start_chunk_id+num_chunks > FLAGS.n_chunks:
      start_chunk_id = np.random.randint(22,FLAGS.n_chunks-num_chunks-1)
    chunk_ids = np.arange(start_chunk_id,start_chunk_id + num_chunks)
    train_counter =train_counter + 1
    
  stim_total = np.zeros((0,640,320,3))
  resp_total =np.zeros((0,FLAGS.n_cells))
  data_len_total = 0

  print(chunk_ids)
  for ichunk in chunk_ids:
    #print('Loading chunk:' + str(ichunk))
    # get chunks
    if(ichunk==chunk_ids[0]):
      #print('first chunk')
      # first entry into the chunk
      stim_chunk, chunk_start, chunk_end  = get_stimulus_batch(ichunk)
      resp_chunk = response[chunk_start+29:chunk_end+1,:]
    else:
      #print('subsequent chunks')
      stim_chunk, chunk_start, chunk_end  = get_stimulus_batch(ichunk)
      stim_chunk = stim_chunk[30:,:,:,:]
      resp_chunk = response[chunk_start+30-1:chunk_end,:]

    data_len = resp_chunk.shape[0]
    #print(chunk_start, chunk_end)
    #print(np.shape(stim_chunk), np.shape(resp_chunk))

    # remove overlapping parts of chunks and then append them!
    stim_total = np.append(stim_total, stim_chunk, axis=0)
    resp_total = np.append(resp_total, resp_chunk, axis=0)
    data_len_total = data_len_total + data_len
  return stim_total, resp_total, data_len_total


def setup_dataset():
  # initialize paths, get dataset properties, etc
  path = FLAGS.data_location

  # load cell response
  response_path = path + 'response.mat'
  response_file = sio.loadmat(gfile.Open(response_path))
  resp_mat = response_file['binned_spikes']
  resp_file_cids = np.squeeze(response_file['cell_ids'])

  # load off parasol cell IDs
  cids_path = path + 'cell_ids/cell_ids_OFF parasol.mat'
  cids_file = sio.loadmat(gfile.Open(cids_path))
  cids_select = np.squeeze(cids_file['cids'])

  # find index of cells to choose from resp_mat
  resp_file_choose_idx = np.array([])
  for icell in np.array(cids_select):
    idx = np.where(resp_file_cids == icell)
    resp_file_choose_idx = np.append(resp_file_choose_idx, idx[0])
    

  # finally, get selected cells from resp_mat
  global response
  response = resp_mat[resp_file_choose_idx.astype('int'),:].T
  print(cids_select)
  print(resp_file_choose_idx.astype('int'))

  # load population time courses
  time_c_file_path = path + 'cell_ids/time_courses.mat'
  time_c_file = sio.loadmat(gfile.Open(time_c_file_path))
  tc_mat = time_c_file['time_courses']
  tm_cids = np.squeeze(time_c_file['cids'])

  # find average time course of cells of interest
  tc_file_choose_idx = np.array([])
  for icell in np.array(cids_select):
    idx = np.where(tm_cids == icell)
    tc_file_choose_idx = np.append(tc_file_choose_idx, idx[0])
  tc_select = tc_mat[tc_file_choose_idx.astype('int'),:,:]
  tc_mean = np.squeeze(np.mean(tc_select,axis=0))
  n_cells = cids_select.shape[0]
  FLAGS.n_cells = n_cells

  #  'response', cell ids are 'cids_select' with 'n_cells' cells, 'tc_select' are timecourses, 'tc_mean' for mean time course
  return response, cids_select, n_cells, tc_select, tc_mean



def main(argv):
  pass


if __name__ == '__main__':
  app.run()

