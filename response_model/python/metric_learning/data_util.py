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
"""Data Util for metric learning project."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.python.platform import gfile  # tf.gfile does NOT work with big mat files.
import retina.response_model.python.metric_learning.analyse_metric as analyse


class DataUtilsMetric(object):
  """Get repeats, triplets for metric learning."""

  def __init__(self, path):
    """Load responses from 'repeats'."""
    print('Loading from: '+ str(path))
    data_file = gfile.Open(path, 'r')
    data = sio.loadmat(data_file)
    self._responses = data['repeats']
    self.n_cells = self._responses.shape[2]
    self.n_trials = self._responses.shape[0]
    self.n_times = self._responses.shape[1]
    mean_response = np.mean(np.reshape(self._responses,
                                       [-1, self._responses.shape[2]]), 0)
    self.mean_response = np.squeeze(mean_response)

    # Check if the data also contains stimulus.
    if 'stimulus' in data.keys():
      self._stimulus = data['stimulus']
    else:
      self._stimulus = None

    if 'centers' in data.keys():
      self._centers = data['centers']
    else:
      self._centers = None

    if 'ttf' in data.keys():
      self.ttf = np.squeeze(data['ttf'])
    else:
      self.ttf = None

    if 'cell_type' in data.keys():
      self.cell_type = np.squeeze(data['cell_type'])
    else:
      self.cell_type = None

  def get_mean_response(self):
    """Return mean response for each cell."""
    return self.mean_response

  def get_cell_type(self):
    """Return cell type of cells. (+1 for type 1, -1 for type 2)."""
    return self.cell_type

  def get_centers(self):
    """Returns center location of cells."""
    return self._centers
  
  def get_repeats(self):
    """Returns response matrix: (Trials x Time x Cells)."""
    return self._responses

  def get_stimulus(self):
    """Returns stimulus matrix: None or (Time x Dimension X x Dimension Y)."""
    return self._stimulus

  def get_all_responses(self, time_window):
    """Return all the response (flatten repeats) in same format as triplets.

    Args :
        time_window (int) : The number of continuous time bins for
                              each response.
    Returns :
        all_responses : All the responses (batch x cells x time_window).
        stim_time : Time index of corresponding stimulus.
    """

    n_trials = self._responses.shape[0]
    n_times = self._responses.shape[1]
    n_cells = self._responses.shape[2]
    all_responses = np.zeros((n_trials * (n_times - time_window + 1),
                              n_cells, time_window))
    icnt = 0
    stim_time = []
    for itrial in range(n_trials):
      for itime in range(n_times - time_window + 1):
        all_responses[icnt, :, :] = self._responses[itrial,
                                                    itime: itime+time_window,
                                                    :].T
        stim_time += [itime]

        icnt += 1

    stim_time = np.array(stim_time)
    return all_responses, stim_time

  def get_triplets(self, batch_size=1000, time_window=50):
    """Get a batch of triplets (anchor, positive, negative).

    'anchor' and 'positive' are responses to same stimulus in different trials
    and 'negative' is response to another stimulus.
    Args:
        batch_size (int) : batch size of triplets
        time_window (int) : number of continuous time bins to
                           include in each example
    Returns:
        anchor : Numpy array of anchor (batch x cells x time_window).
        pos : positive examples - near anchor
        neg : negative examples - far from anchor
        time_log : Times for anchor/positive and negative examples (batch x 2)
        trial_log : Trials for anchor/negative and positive examples (batch x 2)
        0: A dummy output to make number of outputs to 6
    """

    # setup basic parameters
    n_trials = self._responses.shape[0]
    response_length = self._responses.shape[1]
    n_cells = self._responses.shape[2]

    anchor = np.zeros((batch_size, n_cells, time_window))
    pos = np.zeros((batch_size, n_cells, time_window))
    neg = np.zeros((batch_size, n_cells, time_window))
    time_log = np.zeros((batch_size, 2))
    trial_log = np.zeros((batch_size, 2))

    for iexample in range(batch_size):

      # sample random trials
      random_trials = np.random.choice(n_trials, 2, replace=False)

      # sample random times which are atleast time_window apart.
      # time for anchor and positive example.
      itime1 = np.random.randint(response_length - time_window)
      # time for negative example.
      itime2 = np.random.randint(response_length - time_window)

      # time for anchor and negative not too close.
      while np.abs(itime1 - itime2) < time_window:
        itime2 = np.random.randint(response_length - time_window)

      anchor[iexample, :, :] = self._responses[random_trials[0],
                                               itime1: itime1 + time_window,
                                               :].T
      pos[iexample, :, :] = self._responses[random_trials[1],
                                            itime1: itime1 + time_window, :].T
      neg[iexample, :, :] = self._responses[random_trials[0],
                                            itime2: itime2 + time_window, :].T

      time_log[iexample, :] = np.array([itime1, itime2])
      trial_log[iexample, :] = random_trials

    return anchor, pos, neg, time_log, trial_log, 0

  def get_tripletsB(self, batch_size=1000, time_window=50):
    """Get batch of triplets with negatives of same stimulus, no correlations.

    'anchor' and 'positive' are responses to a stimulus in different trials
    and 'negative' is response to same stimulus, but randomly sampled trials
    across cells.

    Args:
        batch_size (int) : batch size of triplets
        time_window (int) : number of continuous time bins to
                           include in each example
    Returns:
        anchor : Numpy array of anchor (batch x cells x time_window).
        pos : positive examples - near anchor
        neg : negative examples - far from anchor
        time_log : Times for anchor/positive and negative examples (batch x 2)
        trial_log : Trials for anchor/positive and positive examples (batch x 2)
        trial_log_negatives : Trials for negatives (batch x n_cells)
    """

    # setup basic parameters
    n_trials = self._responses.shape[0]
    response_length = self._responses.shape[1]
    n_cells = self._responses.shape[2]

    anchor = np.zeros((batch_size, n_cells, time_window))
    pos = np.zeros((batch_size, n_cells, time_window))
    neg = np.zeros((batch_size, n_cells, time_window))
    time_log = np.zeros((batch_size))
    trial_log = np.zeros((batch_size, 2))
    trial_log_negatives = np.zeros((batch_size, n_cells))

    for iexample in range(batch_size):

      # sample random trials
      random_trials = np.random.choice(n_trials, 2, replace=False)
      remaining_trials = np.setdiff1d(np.arange(n_trials), random_trials)
      if remaining_trials == []:
        tf.logging.info('No trials left for negatives.')
        continue
      negative_trials = np.random.choice(remaining_trials, n_cells)

      # sample random times which are atleast time_window apart.
      # time for examples.
      itime1 = np.random.randint(response_length - time_window)

      # time for anchor and negative not too close.
      anchor[iexample, :, :] = self._responses[random_trials[0],
                                               itime1: itime1 + time_window,
                                               :].T
      pos[iexample, :, :] = self._responses[random_trials[1],
                                            itime1: itime1 + time_window, :].T

      for icell in range(n_cells):
        neg[iexample, icell, :] = self._responses[negative_trials[icell],
                                                  itime1:
                                                  itime1 + time_window, icell].T

      time_log[iexample] = np.array(itime1)
      trial_log[iexample, :] = random_trials
      trial_log_negatives[iexample, :] = negative_trials

    return anchor, pos, neg, time_log, trial_log, trial_log_negatives


  def get_response_all_trials(self, n_stims, time_window, random_seed=234):
    """ Get population responses for all repeats of few stimuli.

    Args :
      n_stims (int) : Number of stimuli.
      time_window (int) : Number of successive time bins for each response.

    Returns :
      r : Collection of responses (# responses x # cells x time_window)
      stim_id (int): The stimulus time of each response point (# responses)
    """

    from numpy.random import RandomState
    prng = RandomState(random_seed)
    print('Setting local pseudo-random number generator')

    n_trials = self._responses.shape[0]
    response_length = self._responses.shape[1]
    n_cells = self._responses.shape[2]
    tms = prng.randint(0, response_length, n_stims)

    n_responses = n_trials * n_stims
    r = np.zeros((n_responses, n_cells, time_window))

    for itm_cnt, itm in enumerate(tms):
       r[itm_cnt*n_trials :
         (itm_cnt+1)*n_trials, :, :] = np.transpose(self._responses[:, itm: itm + time_window, :], [0, 2, 1])

    stim_id = np.repeat(tms, n_trials, 0)
    return r, stim_id

  def get_all_response_few_trials(self, n_trials, time_window):
    """Get population responses for few repeats of all stimuli.

    Args :
      n_trials (int) : Number of trials for which to give resposne.
      time_window (int) : Number of successive time bins for each response.

    Returns :
      r : Collection of responses. (# trials x # cells x time_window)
      trials_sample (int): ID of trials selected. (# trials)
    """

    from numpy.random import RandomState
    prng = RandomState(250)
    print('Setting local pseudo-random number generator')

    response_length = self._responses.shape[1]
    n_cells = self._responses.shape[2]
    total_cells = self._responses.shape[0]

    r = np.zeros((n_trials, response_length, n_cells, time_window))
    trials_sample = prng.randint(0, total_cells, n_trials)

    for itrial_cnt, itrial in enumerate(trials_sample):
      for istim in range(response_length):
        r[itrial_cnt, istim, :, :] = np.transpose(np.expand_dims(
            self._responses[itrial, istim: istim + time_window, :], 0),
                                                  [0, 2, 1])

    return r, trials_sample

  # TODO(bhaishahster) : Add triplets by methods (b,c) with negatives generated by mixing trials

  # Additional triplet methods
  def get_tripletsC(self, batch_size=1000, time_window=50):
    """Get batch of triplets with negatives of same stimulus, no correlations.

    'anchor' and 'positive' are responses to different stimuli
    and 'negative' is response to different stimulus, and different repeats
    (d(X,X-) < d(X,Xh-))

    Args:
        batch_size (int) : batch size of triplets
        time_window (int) : number of continuous time bins to
                           include in each example
    Returns:
        anchor : Numpy array of anchor (batch x cells x time_window).
        pos : positive examples - near anchor
        neg : negative examples - far from anchor
        time_log : Times for anchor/positive and negative examples (batch x 2)
        trial_log : Trials for anchor/positive and positive examples (batch x 2)
        trial_log_negatives : Trials for negatives (batch x n_cells)
    """

    # setup basic parameters
    n_trials = self._responses.shape[0]
    response_length = self._responses.shape[1]
    n_cells = self._responses.shape[2]

    anchor = np.zeros((batch_size, n_cells, time_window))
    pos = np.zeros((batch_size, n_cells, time_window))
    neg = np.zeros((batch_size, n_cells, time_window))
    time_log = np.zeros((batch_size))
    trial_log = np.zeros((batch_size, 2))
    trial_log_negatives = np.zeros((batch_size, n_cells))

    for iexample in range(batch_size):

      # sample random trials
      random_trials = np.random.choice(n_trials, 2, replace=False)
      remaining_trials = np.setdiff1d(np.arange(n_trials), random_trials)
      if remaining_trials == []:
        tf.logging.info('No trials left for negatives.')
        continue
      negative_trials = np.random.choice(remaining_trials, n_cells)

      # sample random times which are atleast time_window apart.
      # time for examples.
      itime1 = np.random.randint(response_length - time_window)

      # time for negative example.
      itime2 = np.random.randint(response_length - time_window)

      # time for anchor and negative not too close.
      while np.abs(itime1 - itime2) < time_window:
        itime2 = np.random.randint(response_length - time_window)

      # time for anchor and negative not too close.

      anchor[iexample, :, :] = self._responses[random_trials[0],
                                               itime1: itime1 + time_window,
                                               :].T
      pos[iexample, :, :] = self._responses[random_trials[1],
                                            itime2: itime2 + time_window, :].T

      for icell in range(n_cells):
        neg[iexample, icell, :] = self._responses[negative_trials[icell],
                                                  itime2:
                                                  itime2 + time_window, icell].T

      time_log[iexample] = np.array(itime1)
      trial_log[iexample, :] = random_trials
      trial_log_negatives[iexample, :] = negative_trials

    return anchor, pos, neg, time_log, trial_log, trial_log_negatives

  def get_tripletsD(self, batch_size=1000, time_window=50):
    """Get batch of triplets with negatives of same stimulus, no correlations.

    'anchor' and 'positive' are responses to same stimuli,
    but positive has trials for different cells all mixed up
    and 'negative' is response to different stimulus,
    and trials for different cells mixed up
    (d(X,Xh+)<d(X, Xh-))

    Args:
        batch_size (int) : batch size of triplets
        time_window (int) : number of continuous time bins to
                           include in each example
    Returns:
        anchor : Numpy array of anchor (batch x cells x time_window).
        pos : positive examples - near anchor
        neg : negative examples - far from anchor
        time_log : Times for anchor/positive and negative examples (batch x 2)
        trial_log : Trials for anchor/positive and positive examples (batch x 2)
        trial_log_negatives : Trials for negatives (batch x n_cells)
    """

    # setup basic parameters
    n_trials = self._responses.shape[0]
    response_length = self._responses.shape[1]
    n_cells = self._responses.shape[2]

    anchor = np.zeros((batch_size, n_cells, time_window))
    pos = np.zeros((batch_size, n_cells, time_window))
    neg = np.zeros((batch_size, n_cells, time_window))
    time_log = np.zeros((batch_size))
    trial_log = np.zeros((batch_size, 2))
    trial_log_negatives = np.zeros((batch_size, n_cells))

    for iexample in range(batch_size):

      # sample random trials
      random_trials = np.random.choice(n_trials, 1, replace=False)
      remaining_trials = np.setdiff1d(np.arange(n_trials), random_trials)
      if remaining_trials == []:
        tf.logging.info('No trials left for negatives.')
        continue
      positive_trials = np.random.choice(remaining_trials, n_cells)
      negative_trials = np.random.choice(remaining_trials, n_cells)

      # sample random times which are atleast time_window apart.
      # time for examples.
      itime1 = np.random.randint(response_length - time_window)

      # time for negative example.
      itime2 = np.random.randint(response_length - time_window)

      # time for anchor and negative not too close.
      while np.abs(itime1 - itime2) < time_window:
        itime2 = np.random.randint(response_length - time_window)

      # time for anchor and negative not too close.

      anchor[iexample, :, :] = self._responses[random_trials[0],
                                               itime1: itime1 + time_window,
                                               :].T

      for icell in range(n_cells):
        pos[iexample, icell, :] = self._responses[positive_trials[icell],
                                                  itime1:
                                                  itime1 + time_window, icell].T

      for icell in range(n_cells):
        neg[iexample, icell, :] = self._responses[negative_trials[icell],
                                                  itime2:
                                                  itime2 + time_window, icell].T

      time_log[iexample] = np.array(itime1)
      trial_log[iexample, :] = random_trials
      trial_log_negatives[iexample, :] = negative_trials

    return anchor, pos, neg, time_log, trial_log, trial_log_negatives

  def get_triplets_mix(self, batch_size, time_window, score):
    """Return hard triplets which are hard for the score function."""

    # Get triplets A
    outputs_a = self.get_triplets(batch_size, time_window)
    anchor_batch_a, pos_batch_a, neg_batch_a, _, _, _ = outputs_a
    outputs_a = list(outputs_a) + [None]
    _, _, accuracy_a = analyse.compute_distances(score, anchor_batch_a,
                                                  pos_batch_a, neg_batch_a)
    error_a = 1 - accuracy_a

    # Get triplets B
    outputs_b = self.get_tripletsB(batch_size, time_window)
    anchor_batch_b, pos_batch_b, neg_batch_b, _, _, _ = outputs_b
    _, _, accuracy_b = analyse.compute_distances(score, anchor_batch_b,
                                             pos_batch_b, neg_batch_b)
    error_b = 1 - accuracy_b

    # Get triplets C
    outputs_c = self.get_tripletsC(batch_size, time_window)
    anchor_batch_c, pos_batch_c, neg_batch_c, _, _, _ = outputs_c
    _, _, accuracy_c = analyse.compute_distances(score, anchor_batch_c,
                                             pos_batch_c, neg_batch_c)
    error_c = 1 - accuracy_c

    # Get triplets D
    outputs_d = self.get_tripletsD(batch_size, time_window)
    anchor_batch_d, pos_batch_d, neg_batch_d, _, _, _ = outputs_d
    _, _, accuracy_d = analyse.compute_distances(score, anchor_batch_d,
                                                 pos_batch_d, neg_batch_d)
    error_d = 1 - accuracy_d

    errs = np.array([error_a, error_b, error_c, error_d])
    probs = np.exp(errs)/np.sum(np.exp(errs))

    # print(score.iter , probs)

    xx = np.random.random()
    if xx < probs[0] :
      return outputs_a
    elif  xx < probs[0] + probs[1]:
      return outputs_b
    elif xx < probs[0] + probs[1] + probs[2]:
      return outputs_c
    else :
      return outputs_d

  def get_stimulus_response_samples(self, batch_size, time_window):
    """Get a few samples of stimulus and response

    Args :
        batch_size (int) : number of examples.
        time_window (int) : number of continuous time bins to
                           include in each example
    Returns :
        stimulus_examples : Numpy array of stimulus samples
                            (batch x dimx x dimy).
        response_examples : Numpy array of response samples
                            (batch x cells x time_window).
        time_log : Times for anchor/positive and negative examples (batch)
        trial_log : Trials for anchor/negative and positive examples (batch)
    """

    if self._stimulus is None:
      tf.logging.info('Stimulus not found.')
      return None

    # setup basic parameters
    n_trials = self._responses.shape[0]
    response_length = self._responses.shape[1]
    n_cells = self._responses.shape[2]

    dimx = self._stimulus.shape[1]
    dimy = self._stimulus.shape[2]

    response_examples = np.zeros((batch_size, n_cells, time_window))
    stimulus_examples = np.zeros((batch_size, dimx, dimy))
    time_log = np.zeros((batch_size, 1))
    trial_log = np.zeros((batch_size, 1))

    for iexample in range(batch_size):

      # sample random trials
      random_trial = np.random.choice(n_trials, 1, replace=False)

      # sample random times which are atleast time_window apart.
      # time for anchor and positive example.
      itime = np.random.randint(response_length - time_window)

      stimulus_examples[iexample, :, :] = self._stimulus[itime, :, :]
      response_examples[iexample, :, :] = self._responses[random_trial[0],
                                                          itime: itime +
                                                          time_window,
                                                          :].T
      time_log[iexample] = np.array(itime)
      trial_log[iexample] = random_trial

    return stimulus_examples, response_examples, time_log, trial_log

  def get_triplets_batch(self, batch_size=1000, time_window=50):
    """Get pairs of postitives (anchor, positive) and a batch of negatives.

    'anchor' and 'positive' are responses to same stimulus in different trials
    and 'negative' is response to another stimulus.
    Args:
        batch_size (int) : batch size of triplets
        time_window (int) : number of continuous time bins to
                           include in each example
    Returns:
        anchor : Numpy array of anchor (batch x cells x time_window).
        pos : positive examples - near anchor
        neg : negative examples - far from anchor
        time_log : Times for anchor/positive and negative examples (batch x 2)
        trial_log : Trials for anchor/negative and positive examples (batch x 2)
        0: A dummy output to make number of outputs to 6
    """

    # setup basic parameters
    n_trials = self._responses.shape[0]
    response_length = self._responses.shape[1]
    n_cells = self._responses.shape[2]

    anchor = np.zeros((batch_size, n_cells, time_window))
    pos = np.zeros((batch_size, n_cells, time_window))
    neg = np.zeros((batch_size, n_cells, time_window))

    # generate postitive pairs
    pos_times = []
    for iexample in range(batch_size):

      # sample random trials
      random_trials = np.random.choice(n_trials, 2, replace=False)

      # sample random times which are atleast time_window apart.
      # time for anchor and positive example.
      itime = np.random.randint(response_length - time_window)
      pos_times += [itime]

      anchor[iexample, :, :] = self._responses[random_trials[0],
                                               itime: itime + time_window,
                                               :].T
      pos[iexample, :, :] = self._responses[random_trials[1],
                                            itime: itime + time_window, :].T
    pos_times = np.array(pos_times)

    for iexample in range(batch_size):

      # sample random trials
      random_trials = np.random.choice(n_trials, 1)

      # time for negative example.
      itime = np.random.randint(response_length - time_window)

      # time for anchor and negative not too close.
      while np.min(np.abs(itime - pos_times)) < time_window:
        itime = np.random.randint(response_length - time_window)

      neg[iexample, :, :] = self._responses[random_trials[0],
                                            itime: itime + time_window, :].T


    return anchor, pos, neg, 0, 0, 0

