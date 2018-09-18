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
"""Get jitter data (stimulus and response) batches.

First,load the coarse stimulus matrix, and jitter x&y positions for each point in time.

When the data is requested, coarse stimulus is upsampled and
shifted based on jitter positions to give fine resolution stimulus.
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


'''
flags.DEFINE_string('data_location',
                    '/home/retina_data/datasets/2016-04-21-1/data006(2016-04-21-1_data006_data006)/',
                    'where to take data from?')

flags.DEFINE_string('data_location',
                    '/home/bhaishahster/jitter_data/data006(2016-04-21-1_data006_data006)/',
                    'where to take data from?')
## Dataset specific
flags.DEFINE_float('n_cells',1, 'number of cells in the dataset')
flags.DEFINE_integer('n_chunks',1793, 'number of data chunks') # should be 216
'''
FLAGS = flags.FLAGS

# store data to generate stimulus frames on the fly (during runtime).
response = np.array([]) # all of cell responses (Time x number of cells)
# chunk ids for training and testing dataset
test_chunks = np.array([])
train_chunks = np.array([])
# counters to keep track of what chunks already returned
train_counter = np.array([])
test_counter = np.array([])

# data needed to generate fine stimulus frames on the fly.
coarse_stimulus = np.array([])
jitter_x = np.array([])
jitter_y = np.array([])

frames_needed = np.array([]) # mapping from coarse 'time' points to fine time points
batch_sz = np.array([])
mask = np.array([])
sta_coarse = np.array([])

def init_chunks(batch_sz_in):
  # fill in the global fields with data, called at the beginning of the code.
  global test_chunks
  global train_chunks
  global train_counter
  global response
  global test_counter
  global coarse_stimulus
  global jitter_x
  global jitter_y
  global frames_needed
  global batch_sz
  global mask
  global sta_coarse

  batch_sz = batch_sz_in

  n_chunks = np.floor(215200 / batch_sz)-2
  print('Number of chunks is: ' + str(n_chunks))
  FLAGS.n_chunks = n_chunks
  random_chunks = np.arange(n_chunks)+1 # TODO(bhaishahster) Randomness not applied yet
  print('The random permutation of chunks is ' + str(random_chunks))
  # TODO(bhaishahster) training and testing should be non-overlapping
  test_chunks = random_chunks[0]
  train_chunks = random_chunks
  train_chunks = np.random.permutation(train_chunks)
  train_counter = 0
  test_counter = 0
  response, cids_select, n_cells, tc_select, tc_mean, coarse_stimulus, jitter_x,jitter_y, frames_needed, mask, sta_coarse = setup_dataset()
  return tc_mean


def setup_dataset():
  # initialize paths and load dataset
  path = FLAGS.data_location

  # load cell response
  response_path = path + 'response.mat'
  response_file = sio.loadmat(gfile.Open(response_path))
  resp_mat = response_file['binned_spikes']
  resp_file_cids = np.squeeze(response_file['cell_ids'])

  # load coarse stimulus chunks and jitter positions
  stim_compress_path = path + 'stimulus_compress.mat'
  stim_compress = sio.loadmat(gfile.Open(stim_compress_path))
  coarse_stimulus = stim_compress['real_frame']
  jitter_x = np.squeeze(stim_compress['jitter_x'])
  jitter_y = np.squeeze(stim_compress['jitter_y'])
  frames_needed = np.squeeze(stim_compress['frames_needed'])
  print(np.shape(coarse_stimulus), np.shape(jitter_x), np.shape(jitter_y), np.shape(frames_needed))

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

  # get strong pixels in coarse STA, and also get the STAs
  mask_path = path + 'masks_off_parasol.mat'
  mask_file = sio.loadmat(gfile.Open(mask_path))
  mask = mask_file['mask']
  stas_mat = mask_file['stas_mat']

  #  'response', cell ids are 'cids_select' with 'n_cells' cells,
  # 'tc_select' are timecourses, 'tc_mean' for mean time course
  # 'coarse_stimulus' is coarse stimulus frame,
  # 'jitter_x' & 'jitter_y' are frame-wise jitter directions in x and y,
  # 'frames_needed' are mapping from fine movie frames to coarse movie frames(matlab indexing)
  return response, cids_select, n_cells, tc_select, tc_mean, coarse_stimulus, jitter_x, jitter_y, frames_needed, mask, stas_mat


def get_stim_resp(data_type='train', num_chunks=1):
  # return the 'next' training and testing chunk
  global train_chunks
  global test_chunks
  global train_counter
  global test_counter
  global response

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
      # if cycled through all the chunks once, reorder the chunk
      train_chunks = np.random.permutation(train_chunks)
      train_counter=0

    start_chunk_id = np.squeeze(np.array(train_chunks[train_counter]).astype('int'))
    if start_chunk_id+num_chunks > FLAGS.n_chunks:
      start_chunk_id = np.random.randint(0,FLAGS.n_chunks-num_chunks-1)
    chunk_ids = np.arange(start_chunk_id,start_chunk_id + num_chunks)
    train_counter =train_counter + 1

  stim_total = np.zeros((0,640,320,3))
  resp_total =np.zeros((0,FLAGS.n_cells))
  data_len_total = 0

  print(chunk_ids)
  for ichunk in chunk_ids:
    # get chunk by its chunk ID
    stim_chunk, resp_chunk = get_stim_resp_batch(ichunk)
    data_len = resp_chunk.shape[0]
    #print(chunk_start, chunk_end)
    #print(np.shape(stim_chunk), np.shape(resp_chunk))

    # remove overlapping parts of chunks and then append them!
    stim_total = np.append(stim_total, stim_chunk, axis=0)
    resp_total = np.append(resp_total, resp_chunk, axis=0)
    data_len_total = data_len_total + data_len
  return stim_total, resp_total, data_len_total



def get_stim_resp_batch(ichunk):
  # gets a chunk of fine resolution stimulus and response data
  start_id = batch_sz * ichunk
  end_id = batch_sz * (ichunk +1) + 29
  stim_chunk = get_frames(start_id, end_id) # get fine resolution stimulus in this range
  print(np.shape(stim_chunk))
  stim_chunk = np.transpose(stim_chunk, (3,0,1,2))
  resp_chunk = response[start_id: end_id, :]
  stim_chunk = np.array(stim_chunk, dtype='float32')
  resp_chunk = np.array(resp_chunk, dtype='float32')
  print('Loaded stimulus and response chunk, each from times %d, %d:' %(start_id, end_id))
  
  return stim_chunk, resp_chunk


def get_frames(start_id = 40, end_id = 120):
  # get frames of fine resolution stimulus between these times.
  # this will upsample 'coarse_stimulus', and
  # apply shifts as specified by jitter_x and jitter_y.
  height = image_height = 40;
  width = image_width = 80;
  num_colors=3;
  stixel_size = 8;

  chunk_len = end_id - start_id # total movie length
  movie_len = np.shape(frames_needed)[0];
  stim_chunk = np.array(np.zeros((image_width*stixel_size,
                                  image_height*stixel_size, num_colors,
                                  chunk_len)), dtype='int16')

  for iframe_in_chunk , i in enumerate(np.arange(start_id, end_id, 1)):
    # compute the movie
    movie = np.zeros((image_width*stixel_size, image_height*stixel_size, num_colors), dtype='int16');
    true_frame = np.zeros((width*stixel_size, height*stixel_size), 'int16');

    F = coarse_stimulus[:, :, :, frames_needed[i]-1];
    for icol in np.arange(num_colors):
      shaped_frame = F[:,:,icol];
      scale = np.array([stixel_size, stixel_size]); # The resolution scale factors: [rows columns]
      oldSize = np.shape(shaped_frame); # Get the size of your image
      newSize = scale * oldSize;  # Compute the new image size

      #Compute an upsampled set of indices:

      rowIndex = np.minimum(np.round((np.arange(1, newSize[0]+1)-0.5)/scale[0]+0.5)-1, oldSize[0]-1);
      colIndex = np.minimum(np.round((np.arange(1, newSize[1]+1)-0.5)/scale[1]+0.5)-1, oldSize[1]-1);
      rowIndex = np.expand_dims(np.array(rowIndex, dtype='int'), 1)
      colIndex = np.expand_dims(np.array(colIndex, dtype='int'), 0)
      # Index old image to get new image:
      sized_frame = shaped_frame[rowIndex,colIndex];
      sized_frame = sized_frame[(stixel_size/2):( - stixel_size/2), (stixel_size/2):( - stixel_size/2)];
      position = [jitter_x[frames_needed[i]-1] + 1 + stixel_size/2, jitter_y[frames_needed[i]-1] + 1 + stixel_size/2];
      position = np.squeeze(np.array(position, dtype='int'))
      true_frame[position[0]-1: np.shape(sized_frame)[0]+position[0]-1, position[1]-1:np.shape(sized_frame)[1] + position[1]-1] = sized_frame
      movie[:,:,icol] = true_frame;

    stim_chunk[:,:,:,iframe_in_chunk] = movie

  return stim_chunk




def compute_stas_coarse(icell=23):
  # compute STA for icell using coarse stimulus (coarse_stimulus)
  global coarse_stimulus
  global response

  coarse_stimulus_up = np.repeat(coarse_stimulus, 2, axis=3)
  coarse_stimulus_up_flatten = np.reshape(coarse_stimulus_up, (80*40*3, -1)) # use only green channel to compute STA
  rf_uf_short = np.array(coarse_stimulus_up_flatten[:,0:215199], dtype='float32') # make it same size as response
  response = np.array(response, dtype='float32')



  # In numpy :
  print('STA calculated for:' + str(icell))
  sta  =  np.dot(rf_uf_short[:, 0:200000], response[4:200004,icell])/200000
  sta3d = np.reshape(sta, (80, 40, 3))
  sta3d = np.repeat(np.repeat(sta3d,8,0),8,1)

  '''
  for icol in np.arange(3):
    plt.subplot(1, 3, icol+1)
    plt.imshow(sta3d[:,:,icol]);
  plt.show()
  plt.draw()
  '''
  return sta3d

def get_cell_masks():
  # mask represents which pixels are in STA of a given cell.
  global mask
  return mask

def get_cell_stas_coarse():
  # returns STA computed from coarse resolution stimulus
  global sta_coarse
  return sta_coarse


def get_cell_weights():
  # Get relative weights (related to inverse total number of spikes)
  # for different cells in log-likelihood.
  # We use inverse the total number of spikes.
  raw_spikes = np.double(np.squeeze(np.sum(response,0))) # response is Time x cells
  n_cells = response.shape[1]
  weights = 1 / raw_spikes
  weights = n_cells * weights / np.sum(weights)
  return weights


def main(argv):
  # to test the code, load datasets, and compute STA
  print('Uncomment FLAGS to run this')
  print('starting main')

  batch_sz_in = 100
  # initialize dataset

  init_chunks(batch_sz_in)
  print('intialized')

  # get a chunk of training data
  #stim_total, resp_total, data_len_total = get_stim_resp(data_type='train', num_chunks=1)
  #print('got a chunk of data')


  # get STAs from coarse stimulus
  # stas = compute_stas_coarse()

  # get masks and plot them! , remove bad cells?
  # global mask, sta_coarse
  import time
  start = time.time()
  frames = get_frames(start_id = 1, end_id = 100)
  duration = time.time() - start
  print('Loading 100 frames, duratino %.3f' % duration)


  start = time.time()
  frames = get_frames(start_id = 1, end_id = 215900)
  duration = time.time() - start
  print('Loading 215900 frames, duratino %.3f' % duration)
  
  #from IPython.terminal.embed import InteractiveShellEmbed
  #ipshell = InteractiveShellEmbed()
  #ipshell()

  if __name__ == '__main__':
    app.run()

if __name__ == '__main__':
  app.run()

