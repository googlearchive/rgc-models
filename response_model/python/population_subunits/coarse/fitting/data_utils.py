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
""" Classes to store and access data for subunit learning algorithm
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import collections
import tensorflow as tf
from absl import gfile

import cPickle as pickle
import numpy as np
import scipy.io as sio
from scipy import ndimage
import random

class DataUtils(object):
  # interface for all data I/O functions

  def __init__(self):
    pass

  def get_test_data(self):
    # get stimulus and response test data
    pass

  def get_next_training_batch(self):
    # one batch of stimulus and response dara
    pass

  def get_all_data(self):
    # dump all stimulus and response data,
    # useful when data stored as constants in tensorflow,
    # or data converted into TFRecords
    pass

  def convert_to_TFRecords(self, name,  save_location):
    # converts the stimulus-response data into a TFRecords file.
    pass


class CoarseDataUtils(DataUtils):

  def __init__(self, data_location, batch_sz_in=1000,
               masked_stimulus=False, chosen_cells = None,
               total_samples = 216000, storage_chunk_sz = 1000,
               data_chunk_prefix =  'Off_par_data_', stimulus_dimension = 3200,
               test_length=20000,
               sr_key={'stimulus': 'maskedMovdd_part', 'response':'Y_part'}):

    # loads dataset, sets up variables.
    # data_location: where data is stored, after splitting into chunks,
    # with each chunk having prefix 'data_chunk_prefix'
    # having 'storage_chunk_sz' number of samples out of 'total_samples'
    # 'masked_stimulus': if we want to clip the stimulus to relevant pixels for selected cells
    # chosen_cells : which cells to include in 'response'. If None, include all cells
    # sr_key: what variables in stored data correspond to stimulus and response matrices
    # the stored matrices should have shape:
    # stimulus (# stimulus_dimension x time) and response (# chosen_cells x time)
    # batch_sz_in : number of examples in each batch of training data
    # test_length : number of examples in test data


    if chosen_cells is None:
      all_cells = True

    self.batch_sz = batch_sz_in

    print('Initializing datasets')
    # Load summary of datasets
    data_filename = data_location + 'data_details.mat'
    tf.logging.info('Loading summary file: ' + data_filename)
    summary_file = gfile.Open(data_filename, 'r')

    data_summary = sio.loadmat(summary_file)
    #data_summary = sio.loadmat(data_filename)

    cells = np.squeeze(data_summary['cells'])
    print('Dataset details loaded')

    # choose cells to build model for
    if all_cells:
      self.cells_choose = np.array(np.ones(np.shape(cells)), dtype='bool')
    else:
      cells_choose = np.zeros(len(cells))
      for icell in chosen_cells:
        cells_choose+= np.array(cells == icell).astype(np.int)
      self.cells_choose = cells_choose>0
    n_cells = np.sum(self.cells_choose) # number of cells
    self.stas = np.array(data_summary['stas'])
    self.stas = self.stas[:,self.cells_choose]
    print('Cells selected: %d' % n_cells)

    # count total number of spikes for chosen cells
    tot_spks = np.squeeze(data_summary['tot_spks'])
    tot_spks_chosen_cells = np.array(tot_spks[self.cells_choose] ,dtype='float32')
    self.total_spikes_chosen_cells = tot_spks_chosen_cells
    print('Total number of spikes loaded')

    # choose what stimulus mask to use
    # self.chosen_mask = which pixels to learn subunits over
    if masked_stimulus:
      total_mask = np.squeeze(data_summary['totalMaskAccept_log']).T
      self.cell_mask = total_mask
      self.chosen_mask = np.array(np.sum(total_mask[self.cells_choose,:],0)>0,
                                  dtype='bool')
    else:
      total_mask = np.squeeze(data_summary['totalMaskAccept_log']).T
      self.cell_mask = total_mask
      self.chosen_mask = np.array(np.ones(stimulus_dimension).astype('bool'))
    stim_dim = np.sum(self.chosen_mask) # stimulus dimensions
    print('Stimulus mask size: %d' % np.sum(stim_dim))
    print('Stimulus mask loaded')

    # load stimulus and response from .mat files
    # python cant read too big .mat files,
    # so have broken it down into smaller pieces and stitch the data later
    self.stimulus = np.zeros((total_samples, np.sum(self.chosen_mask)))
    self.response = np.zeros((total_samples, n_cells))
    n_chunks_load = np.int(total_samples / storage_chunk_sz)
    for ichunk in range(216):
      print('Loading %d' %ichunk)
      filename = data_location + data_chunk_prefix + str(ichunk + 1) + '.mat'
      tf.logging.info('Trying to load: ' + filename)
      file_r = gfile.Open(filename, 'r')

      data = sio.loadmat(file_r)
      #data = sio.loadmat(filename)

      X = data[sr_key['stimulus']].T
      Y = data[sr_key['response']].T
      self.stimulus[ichunk*storage_chunk_sz:
               (ichunk+1)*storage_chunk_sz, :] = X[:, self.chosen_mask]
      self.response[ichunk*storage_chunk_sz:
               (ichunk+1)*storage_chunk_sz, :] = Y[:, self.cells_choose]


    # set up training and testing chunk IDs
    n_chunks = np.int(total_samples / self.batch_sz)
    test_num_chunks = test_length/self.batch_sz
    self.test_chunks = np.arange(test_num_chunks)
    self.train_chunks = np.random.permutation(np.arange(test_num_chunks+1, n_chunks))
    self.ichunk_train = 0

  def get_test_data(self):
    # get testing data, it will consist of multiple batches, so join them and return

    batch_sz = self.batch_sz
    stimulus_test = np.zeros((len(self.test_chunks)*batch_sz,
                              self.stimulus.shape[1]))
    response_test = np.zeros((len(self.test_chunks)*batch_sz,
                              self.response.shape[1]))
    for icnt, ichunk in enumerate(self.test_chunks):
      stimulus_test[ichunk*batch_sz :
                    (ichunk+1)*batch_sz, :] = self.stimulus[ichunk*batch_sz :
                                                            (ichunk+1)*batch_sz, :]
      response_test[ichunk*batch_sz
                    : (ichunk+1)*batch_sz, :] = self.response[ichunk*batch_sz :
                                                              (ichunk+1)*batch_sz, :]

    stimulus_test = stimulus_test.astype(np.float32)
    response_test = response_test.astype(np.float32)
    return stimulus_test, response_test


  def get_next_training_batch(self):
    # Returns a new batch of training data : stimulus and response arrays
    # use train_chunks to shuffle the data
    # ichunk_train is the index of which chunk we have already read

    chunk = self.train_chunks[self.ichunk_train]
    stimulus_train = self.stimulus[chunk*self.batch_sz: (chunk+1)*self.batch_sz, :]
    response_train = self.response[chunk*self.batch_sz: (chunk+1)*self.batch_sz, :]
    stimulus_train = stimulus_train.astype(np.float32)
    response_train = response_train.astype(np.float32)
    print('Training chunk: ' + str(chunk))

    self.ichunk_train += 1
    if self.ichunk_train >= len(self.train_chunks):
      print('Reshuffling')
      self.ichunk_train=0
      self.train_chunks = np.random.permutation(self.train_chunks)
    return stimulus_train, response_train


  def convert_to_TFRecords(self, name, save_location='~/tmp'):
    # converts the stimulus-response data into a TFRecords file.

    stimulus = self.stimulus.astype(np.float32)
    response = self.response.astype(np.float32)
    num_examples = stimulus.shape[0]

    def _int64_feature(value):
      # value: number to convert to tf.train.Feature
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
      # value: bytes to convert to tf.train.Feature
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    filename = os.path.join(save_location, name)
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):

      stimulus_raw = stimulus[index,:].tostring()
      response_raw = response[index,:].tostring()
      #print(index, stimulus[index,:].shape, response[index,:].shape)
      example = tf.train.Example(features=tf.train.Features(feature={
          'stimulus': _bytes_feature(stimulus_raw),
          'response': _bytes_feature(response_raw)}))
      writer.write(example.SerializeToString())
    writer.close()

  def get_stas(self):
    #stas = (self.stimulus.T.dot(self.response))/np.sum(response, 0)
    return self.stas

  def convert_to_TFRecords_chunks(self, prefix, save_location='~/tmp',
                                  examples_per_file=1000):
    # converts the stimulus-response data into a TFRecords file.

    tf.logging.info('making TFRecords chunks')
    stimulus = self.stimulus.astype(np.float32)
    response = self.response.astype(np.float32)
    num_examples = stimulus.shape[0]
    num_files = np.ceil(num_examples / examples_per_file).astype(np.int)
    tf.logging.info('Number of files: %d, examples: %d' % (num_files, num_examples))
    def _bytes_feature(value):
      # value: bytes to convert to tf.train.Feature
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # make folder for storing .tfrecords files
    folder_tfrec = os.path.join(save_location, prefix)
    # Make folder if it does not exist.
    if not gfile.IsDirectory(folder_tfrec):
      tf.logging.info('making folder to store tfrecords')
      gfile.MkDir(folder_tfrec)
    else:
      tf.logging.info('folder exists, will overwrite results')

    index = -1
    for ifile in range(num_files):
      filename = os.path.join(folder_tfrec, 'chunk_' + str(ifile) + '.tfrecords')
      tf.logging.info('Writing %s , starting index %d' % (filename, index+1))
      writer = tf.python_io.TFRecordWriter(filename)
      for iexample in range(examples_per_file):
        index += 1
        stimulus_raw = stimulus[index,:].tostring()
        response_raw = response[index,:].tostring()
        #print(index, stimulus[index,:].shape, response[index,:].shape)
        example = tf.train.Example(features=tf.train.Features(feature={
            'stimulus': _bytes_feature(stimulus_raw),
            'response': _bytes_feature(response_raw)}))
        writer.write(example.SerializeToString())
      writer.close()


def read_and_decode(filename_queue, stim_dim, resp_dim):
  # read one example and decode the stimulus and response example

  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'stimulus': tf.FixedLenFeature([], tf.string),
          'response': tf.FixedLenFeature([], tf.string),
      })

  # Convert from a scalar string tensor
  stimulus = tf.decode_raw(features['stimulus'], tf.float32)
  response = tf.decode_raw(features['response'], tf.float32)

  stimulus.set_shape([stim_dim])
  response.set_shape([resp_dim])

  return stimulus, response


def inputs(name, data_location, batch_size, num_epochs, stim_dim, resp_dim):
  # gives a batch of stimulus and responses from a .tfrecords file
  # works for .tfrecords file made using CoarseDataUtils.convert_to_TFRecords

  # Get filename queue.
  # Actual name is either 'name', 'name.tfrecords' or
  # folder 'name' with list of .tfrecords files.
  with tf.name_scope('input'):
    filename = os.path.join(data_location, name)
    filename_extension = os.path.join(data_location, name + '.tfrecords')
    if gfile.Exists(filename) and not gfile.IsDirectory(filename):
      tf.logging.info('%s Exists' % filename)
      filenames = [filename]
    elif gfile.Exists(filename_extension) and not gfile.IsDirectory(filename_extension):
      tf.logging.info('%s Exists' % filename_extension)
      filenames = [filename_extension]
    elif gfile.IsDirectory(filename):
      tf.logging.info('%s Exists and is a directory' % filename)
      filenames_short = gfile.ListDirectory(filename)
      filenames = [os.path.join(filename, ifilenames_short) for ifilenames_short in filenames_short ]
    tf.logging.info(filenames)
    filename_queue = tf.train.string_input_producer(filenames,
                                                    num_epochs=num_epochs,
                                                    capacity=10000)

  # Even when reading in multiple threads, share the filename
  # queue.
  stimulus, response = read_and_decode(filename_queue, stim_dim, resp_dim)

  # Shuffle the examples and collect them into batch_size batches.
  # (Internally uses a RandomShuffleQueue.)
  # We run this in two threads to avoid being a bottleneck.

  stimulus_batch, response_batch = tf.train.shuffle_batch(
      [stimulus, response], batch_size=batch_size, num_threads=30,
      capacity = 5000 + 3 * batch_size,
      # Ensures a minimum amount of shuffling of examples.
      min_after_dequeue=2000)

  '''
  stimulus_batch, response_batch = tf.train.batch(
      [stimulus, response], batch_size=batch_size, num_threads=30,
      capacity = 50000 + 3 * batch_size)
  '''
  return  stimulus_batch, response_batch
