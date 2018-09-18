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
"""Stimulation algorithm for prosthesis.

Build using:
 build -c opt --copt=-mavx --config=cuda learning/brain/public/tensorflow_std_server{,_gpu} //experimental/retina/prosthesis:stimulation_pgd_vary_conditions

/retina/prosthesis/stimulation_pgd_vary_conditions --logtostderr

# Test :
 --delta=100 --t_max=20 --learning_rate=5  --normalization='C'

 --delta=100 --t_max=20 --learning_rate=5  --normalization='C'

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import pickle
from absl import gfile
import copy
import os
import cvxpy
import tensorflow as tf
from tensorflow.python.platform import gfile
from retina.prosthesis.stimulation_pgd import greedy_stimulation
from retina.prosthesis.stimulation_pgd import simultaneous_planning
from retina.prosthesis.stimulation_pgd import simultaneous_planning_interleaved_discretization
from retina.prosthesis.stimulation_pgd import simultaneous_planning_interleaved_discretization_exp_gradient

tf.flags.DEFINE_integer('taskid', 0,
                    'Task ID determines what dictionary to use')

tf.flags.DEFINE_string('src_dict',
                    '/home/bhaishahster/dictionary_diverse/',
                    'Directory to store results.')


FLAGS = tf.flags.FLAGS


def main(argv):
  np.random.seed(23)

  # Figure out dictionary path.
  dict_list = gfile.ListDirectory(FLAGS.src_dict)
  dict_path = os.path.join(FLAGS.src_dict, dict_list[FLAGS.taskid])

  # Load the dictionary
  if dict_path[-3:] == 'pkl':
    data = pickle.load(gfile.Open(dict_path, 'r'))
  if dict_path[-3:] == 'mat':
    data = sio.loadmat(gfile.Open(dict_path, 'r'))

  #FLAGS.save_dir = '/home/bhaishahster/stimulation_algos/dictionaries/' + dict_list[FLAGS.taskid][:-4]
  FLAGS.save_dir = FLAGS.save_dir + dict_list[FLAGS.taskid][:-4]
  if not gfile.Exists(FLAGS.save_dir):
    gfile.MkDir(FLAGS.save_dir)

  # S_collection = data['S']  # Target
  A = data['A']  # Decoder
  D = data['D'].T  # Dictionary

  # clean dictionary
  thr_list = np.arange(0, 1, 0.01)
  dict_val = []
  for thr in thr_list:
    dict_val += [np.sum(np.sum(D.T > thr, 1)!=0)]
  plt.ion()
  plt.figure()
  plt.plot(thr_list, dict_val)
  plt.xlabel('Threshold')
  plt.ylabel('Number of dictionary elements with \n atleast one element above threshold')
  plt.title('Please choose threshold')
  thr_use = float(input('What threshold to use?'));
  plt.axvline(thr_use)
  plt.title('Using threshold: %.5f' % thr_use)

  dict_valid = np.sum(D.T > thr_use, 1)>0
  D = D[:, dict_valid]
  D = np.append(D, np.zeros((D.shape[0], 1)), 1)
  print('Appending a "dummy" dictionary element that does not activate any cell')

  # Vary stimulus resolution
  for itarget in range(20):
    n_targets = 1
    for stix_resolution in [32, 64, 16, 8]:

      # Get the target
      x_dim = int(640 / stix_resolution)
      y_dim = int(320 / stix_resolution)
      targets = (np.random.rand(y_dim, x_dim, n_targets) < 0.5) - 0.5
      upscale = stix_resolution / 8
      targets = np.repeat(np.repeat(targets, upscale, axis=0), upscale, axis=1)
      targets = np.reshape(targets, [-1, targets.shape[-1]])
      S_actual = targets[:, 0]

      # Remove null component of A from S
      S = A.dot(np.linalg.pinv(A).dot(S_actual))

      # Run Greedy first to initialize
      x_greedy = greedy_stimulation(S, A, D, max_stims = FLAGS.t_max * FLAGS.delta,
                       file_suffix='%d_%d' % (stix_resolution, itarget),
                                    save=True, save_dir=FLAGS.save_dir)

      # Load greedy output from previous run
      #data_greedy = pickle.load(gfile.Open('/home/bhaishahster/greedy_2000_32_0.pkl', 'r'))
      #x_greedy = data_greedy['x_chosen']

      # Plan for multiple time points
      x_init = np.zeros((x_greedy.shape[0], FLAGS.t_max))
      for it in range(FLAGS.t_max):
        print((it + 1) * FLAGS.delta - 1)
        x_init[:, it] = x_greedy[:, (it + 1) * FLAGS.delta - 1]

      #simultaneous_planning(S, A, D, t_max=FLAGS.t_max, lr=FLAGS.learning_rate,
      #                     delta=FLAGS.delta, normalization=FLAGS.normalization,
      #                    file_suffix='%d_%d_normal' % (stix_resolution, itarget), x_init=x_init, save_dir=FLAGS.save_dir)

      from IPython import embed; embed()
      simultaneous_planning_interleaved_discretization(S, A, D,
                                                       t_max=FLAGS.t_max,
                                                       lr=FLAGS.learning_rate,
                                                       delta=FLAGS.delta,
                                                       normalization=FLAGS.normalization,
                          file_suffix='%d_%d_pgd' % (stix_resolution, itarget),
                                                       x_init=x_init,
                                                       save_dir=FLAGS.save_dir,
                                                       freeze_freq=np.inf, steps_max=500*20 - 1)

      # Interleaved discretization.
      simultaneous_planning_interleaved_discretization(S, A, D,
                                                       t_max=FLAGS.t_max,
                                                       lr=FLAGS.learning_rate,
                                                       delta=FLAGS.delta,
                                                       normalization=FLAGS.normalization,
                          file_suffix='%d_%d_pgd_od' % (stix_resolution, itarget),
                                                       x_init=x_init,
                                                       save_dir=FLAGS.save_dir, freeze_freq=500, steps_max=500*20 - 1)



      # Exponential weighing.
      simultaneous_planning_interleaved_discretization_exp_gradient(S, A, D,
                                                       t_max=FLAGS.t_max,
                                                       lr=FLAGS.learning_rate,
                                                       delta=FLAGS.delta,
                                                       normalization=FLAGS.normalization,
                          file_suffix='%d_%d_ew' % (stix_resolution, itarget),
                                                       x_init=x_init,
                                                       save_dir=FLAGS.save_dir, freeze_freq=np.inf, steps_max=500*20 - 1)

      # Exponential weighing with interleaved discretization.
      simultaneous_planning_interleaved_discretization_exp_gradient(S, A, D,
                                                       t_max=FLAGS.t_max,
                                                       lr=FLAGS.learning_rate,
                                                       delta=FLAGS.delta,
                                                       normalization=FLAGS.normalization,
                          file_suffix='%d_%d_ew_od' % (stix_resolution, itarget),
                                                       x_init=x_init,
                                                       save_dir=FLAGS.save_dir, freeze_freq=500, steps_max=500*20 - 1)

      '''
      # Plot results
      data_fractional = pickle.load(gfile.Open('/home/bhaishahster/2012-09-24-0_SAD_fr1/pgd_20_5.000000_C_100_32_0_normal.pkl', 'r'))
      plt.ion()
      start = 0
      end = 20

      error_target_frac = np.linalg.norm(S - data_fractional['S'])
      #error_target_greedy = np.linalg.norm(S - data_greedy['S'])
      print('Did target change? \n Err from fractional: %.3f ' % error_target_frac)
      #'\n Err from greedy %.3f' % (error_target_frac,
      #                             error_target_greedy))
      # from IPython import embed; embed()
      normalize = np.sum(data_fractional['S'] ** 2)
      plt.ion()
      plt.plot(np.arange(120, len(data_fractional['f_log'])), data_fractional['f_log'][120:] / normalize, 'b')
      plt.axhline(data_fractional['errors'][5:].mean() / normalize, color='m')
      plt.axhline(data_fractional['errors_ht_discrete'][5:].mean() / normalize, color='y')
      plt.axhline(data_fractional['errors_rr_discrete'][5:].mean() / normalize, color='r')
      plt.axhline(data_greedy['error_curve'][120:].mean() / normalize, color='k')
      plt.legend(['fraction_curve', 'fractional error', 'HT', 'RR', 'Greedy'])
      plt.pause(1.0)
      '''

  from IPython import embed; embed()


if __name__ == '__main__':
  app.run(main)
