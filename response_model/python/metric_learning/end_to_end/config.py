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
"""Set global parameters for learning/evaluating the joint embedding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


flags = tf.app.flags

# Only white noise (WN): '/home/bhaishahster/stim-resp_collection2/'
# Both WN and natural scenes (NSEM) :
# '/home/bhaishahster/stim-resp_collection/'
flags.DEFINE_string('src_dir',
                    '/home/bhaishahster/stim-resp_collection/',
                    'Where is the datasets are')

flags.DEFINE_string('partition_file', '/home/bhaishahster/'
                    'stim-resp_collection/partitions.txt',
                    'Training and testing partitions for different tasks')

flags.DEFINE_string('tmp_dir',
                    '/home/bhaishahster/Downloads/',
                    'Temporary folder on machine for better I/O')

flags.DEFINE_string('save_folder',
                    '/home/bhaishahster/end_to_end_refrac',
                    'Where to store model and analysis')

flags.DEFINE_bool('minimize_disk_usage',
                  False,
                  'Deletes data from local HDD if its no longer needed.')

flags.DEFINE_integer('max_iter', 2000000,
                     'Maximum number of iterations')

flags.DEFINE_integer('batch_neg_train_sz', 100,
                     'Size of negative batch')

flags.DEFINE_integer('batch_train_sz', 100,
                     'Size of training batch')

# Partition of data into training and testing

flags.DEFINE_integer('test_min', 0,
                     'Minimum time for test window')

flags.DEFINE_integer('test_max', 30000,
                     'Maximum time for test window')

flags.DEFINE_integer('train_min', 32000,
                     'Minimum time for train window')

flags.DEFINE_integer('train_max', 21600000,
                     'Maximum time for train window')

flags.DEFINE_integer('taskid', 0,
                     'Maximum number of iterations')

flags.DEFINE_integer('valid_cells_boundary', 0,
                     'How far valid cells should be from border.')

flags.DEFINE_integer('mode', 0,
                     'If testing for prosthesis(2), encoding/decoding(1)'
                     ' or training(0)')

flags.DEFINE_string('save_suffix',
                    '',
                    'Suffix to save files')

## Set up embeddings.
flags.DEFINE_string('sr_model',
                    'convolutional_embedding',
                    'How to learn the embeddings.')

flags.DEFINE_string('ttf_file',
                    '/home/bhaishahster/stim-resp_collection/'
                    'ttf_log.pkl',
                    'Location of averaged time course from different'
                    ' populations. Used for linear metrics for blind retina.')

# Flags for convolutional model.
flags.DEFINE_string('resp_layers',
                    '3, 128, 1, 3, 128, 1, 3, 128, 1, 3, '
                    '128, 2, 3, 128, 2, 3, 1, 1',
                    'Response embedding layers.')

flags.DEFINE_string('stim_layers',
                    '1, 5, 1, 3, 128, 1, 3, 128, 1, 3, 128, '
                    '1, 3, 128, 2, 3, 128, 2, 3, 1, 1',
                    'Stimulus embedding layers.')

flags.DEFINE_bool('batch_norm',
                  True,
                  'Do we apply batch-norm regularization between layers?')

flags.DEFINE_float('beta',
                   10,
                   'Temperature parameters for log-sum-exp loss.')

# Flags for training.
flags.DEFINE_float('learning_rate',
                   0.001,
                   'Learning rate for optimization.')

flags.DEFINE_bool('subsample_cells_train', False,
                  'Subsample continguous cell populations '
                  'for data augmentation while training.')

# Flags for auto-embedding model
flags.DEFINE_float('scale_triplet', 1, 'How much to weigh the triplet loss')
flags.DEFINE_float('scale_encode', 0.5, 'How much to weigh the encoding loss')
flags.DEFINE_float('scale_decode', 0.005, 'How much to weigh the decoding loss')
flags.DEFINE_float('scale_regularization', 0.005, 'How much to weigh the regularization')

# Used for models that embedding EIs as well
flags.DEFINE_bool('use_EIs', False, 'Use the blind retina?')
flags.DEFINE_bool('batch_norm_ei',
                  True,
                  'Do we apply batch-norm regularization between EI embedding layers?')

flags.DEFINE_string('ei_layers',
                    '1, 5, 1, 3, 128, 1, 3, 128, 1, 3, 128, '
                    '1, 3, 128, 2, 3, 128, 2, 3, 1, 1',
                    'EI embedding layers.')

flags.DEFINE_float('scale_encode_from_ei', 1, 'Encoding loss from EIs')
flags.DEFINE_float('scale_regularization_from_ei', 0, 'L2 regularization of EI embedding.')
flags.DEFINE_float('scale_match_embeddding', 0.5, 'Match retina embedding and EI embedding.')


FLAGS = tf.app.flags.FLAGS
