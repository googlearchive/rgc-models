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
r"""Generate partitions of different retinas."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf
import tensorflow.contrib.slim as slim
from absl import app
from absl import gfile
import numpy as np, h5py,numpy
import scipy.io as sio
from numpy.random import RandomState


def generate_partitions(n_datasets,
                        n_tasks,
                        log_file,
                        n_training_datasets_log=[1, 3, 5, 7, 9, 11, 12],
                        n_tasks_per_num_train=4, frequency_test=4):

    # open file for writing
    f = open(log_file, 'wb')

    # make test datasets
    datasets_all = np.arange(n_datasets)
    testing_datasets = datasets_all[::frequency_test]
    training_datasets_all = np.setdiff1d(datasets_all, testing_datasets)
    print('Testing datasets', testing_datasets)

    # make train datasets for different tasks
    for taskid in range(n_tasks):
      if (np.floor(taskid / n_tasks_per_num_train)).astype(np.int) < len(n_training_datasets_log):

        prng = RandomState(23)
        n_training_datasets = n_training_datasets_log[(np.floor(taskid / n_tasks_per_num_train)).astype(np.int)]
        for _ in range(10 * taskid):
          xx = (prng.choice(training_datasets_all,
                            n_training_datasets, replace=False))
        training_datasets = prng.choice(training_datasets_all,
                                        n_training_datasets, replace=False)

      else:
        datasets_all = np.arange(n_datasets)
        training_datasets = [datasets_all[taskid -
                                          (n_tasks_per_num_train *
                                           len(n_training_datasets_log))]]

      print('Task ID, Training datasets', (taskid, training_datasets))
      train_string = ",".join(str(e) for e in training_datasets)
      test_string = ",".join(str(e) for e in testing_datasets)
      f.write('%d:%s:%s\n' % (taskid, train_string, test_string))

    f.close()

def get_partitions(partition_file, taskid):
  """Reads partition file and return training/testing paritions for the taskid.

  The partition file has each task in differnet rows, with each row format -
  taskid: training dataset list: testing dataset list
  Args:
    partition_file : File containing all the partitions.
    taskid : Index of partition to load.

  Returns:
    training_datasets : dataset ids to train on.
    testing_datasets: dataset ids to test on.
  """
  # Load partitions
  with gfile.Open(partition_file, 'r') as f:
    content = f.readlines()

  for iline in content:
    tokens = iline.split(':')
    tokens = [i.replace(' ', '') for i in tokens]
    if taskid == int(tokens[0]):
      training_datasets = tokens[1].split(',')
      testing_datasets = tokens[2].split(',')

      training_datasets = [int(i) for i in training_datasets]
      testing_datasets = [int(i) for i in testing_datasets]
      break

  return training_datasets, testing_datasets


