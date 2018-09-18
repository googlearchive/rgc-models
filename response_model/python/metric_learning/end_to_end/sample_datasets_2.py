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
r"""Utils for sampling stimulus-response batches for learning the embeddings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import skimage.transform
from retina.response_model.python.metric_learning.end_to_end.config import FLAGS


def get_triplets(stimulus, responses, batch_size=100, batch_neg_resp=100,
                 stim_history=30, min_window=10,
                 batch_type='test', negative_stim=False, negative_resp=True,
                 if_continuous=False, rng=None):
  """Get a batch of triplets for form (stimulus, resp_positive, resp_neg).

  For training, the a common set of negatives for all
  stimulus-positive pairs is returned.
  For testing, each positive pair has a randomly chosen negative example,
  but not necessarilty negative for other positive examples.

  Args:
    stimulus : Visual stimulus (T, X, Y).
    responses : Responses of a retinal preparation (T, # cells).
    batch_size : Number of positive stimulus-response pairs.
    batch_neg_resp : Number of negative responses.
    stim_history : Stimulus history for each stimulus example.
    min_window : minimum number of time bins between any two positive pairs.
    batch_type : If 'train' or 'test' triplets
                (common negatives, see above).
    negative_stim : If negative stimulus is returned (default: False).
    negative_resp : If negative response is returned (defualt: True).
    if_continuous : If the returned stim-responses belong to a continuous set.
    rng : Random number generator.

  Returns:
    stim_batch : Stimulus batch (Batch size, X, Y, stim_history)
    resp_batch : Positive response batch (Batch size, # cells, 1)
    resp_batch_neg : Negative response batch (Negative batch size, # cells, 1)

  Raises:
    ValueError : If batch_size != batch_neg_resp for test triplets.
  """

  stim = stimulus
  resp = responses
  if rng is None:
    rng = np.random

  if batch_type == 'test':
    t_min = FLAGS.test_min
    t_max = FLAGS.test_max
  elif batch_type == 'train':
    t_min = FLAGS.train_min
    t_max = FLAGS.train_max

  t_max = np.min([stim.shape[0], resp.shape[0], t_max])
  t_min = np.max([stim_history, t_min])

  # Get positive stimulus-response pairs.
  stim_batch = np.zeros((batch_size, stim.shape[1],
                         stim.shape[2], stim_history))
  resp_batch = np.zeros((batch_size, resp.shape[1]))

  if not if_continuous:
    random_times = rng.randint(t_min, t_max - 1, batch_size)
  if if_continuous:
    start_time = rng.randint(t_min, t_max - 1 - batch_size, 1)
    random_times = np.arange(start_time, start_time + batch_size)

  for isample in range(batch_size):
    itime = random_times[isample]
    stim_batch[isample, :, :, :] = np.transpose(stim[itime:
                                                     itime-stim_history:-1,
                                                     :, :],
                                                [1, 2, 0])
    resp_batch[isample, :] = resp[itime, :]

  # Get negative responses.
  if negative_stim :
    stim_batch_neg = np.zeros((batch_neg_resp, stim.shape[1],
                               stim.shape[2], stim_history))
  if negative_resp:
    resp_batch_neg = np.zeros((batch_neg_resp, resp.shape[1]))

  if batch_type == 'train':
    # Common set of negatives for training data.
    for isample in range(batch_neg_resp):
      itime = rng.randint(t_min, t_max - 1, 1)
      while np.min(np.abs(itime - random_times)) < min_window:
        itime = rng.randint(t_min, t_max - 1, 1)

      if negative_stim :
        stim_batch_neg[isample, :, :, :] = np.transpose(stim[itime[0]:
                                                     itime[0]-stim_history:-1,
                                                     :, :],
                                                [1, 2, 0])
      if negative_resp:
        resp_batch_neg[isample, :] = resp[itime, :]

  elif batch_type == 'test':
    # Independent negative for each positive pair in data.
    # batch_neg_resp and batch_size must be of same size
    if batch_neg_resp != batch_size:
      raise ValueError('Positive and Negative batch size must be equal '
                       'for test data')

    for isample in range(batch_neg_resp):
      itime = rng.randint(t_min, t_max - 1, 1)
      while (np.min(np.abs(itime - random_times[isample])) < min_window) or (itime < stim_history):
        itime = rng.randint(t_min, t_max - 1, 1)

      if negative_stim:
        stim_batch_neg[isample, :, :, :] = np.transpose(stim[itime[0]:
                                                        itime[0]-stim_history:-1,
                                                        :, :],
                                                [1, 2, 0])
      if negative_resp:
        resp_batch_neg[isample, :] = resp[itime, :]

  return_dict = {'stim_batch': stim_batch, 'resp_batch': resp_batch}
  if negative_stim :
    return_dict.update({'stim_batch_neg': stim_batch_neg})

  if negative_resp:
    return_dict.update({'resp_batch_neg': resp_batch_neg})
  return return_dict


def batch(stimuli, responses, dataset_id, sr_model,
          batch_pos_sz, batch_neg_sz, batch_type='test',
          frac_cells=1.0, if_continuous=False, rng=None):
  """Get triplets and auxillary information for a dataset.

  Auxillary information includes cell types, cell locations and
    mean firing rates.
  Assigns these values to a model placeholders and returns feed_dict.

  Args :
    stimuli : Dictionary of different stimuli
               (key(string), numpy array (T x X, Y))
    responses : List of responses for different retinas.
    dataset_id : The index of dataset in 'responses'
                   from which to generate triplets.
    sr_model : Stimulus-response embedding model.
    batch_pos_sz : Size of positive stimulus-response pairs.
    batch_neg_sz : Size of negatives.
    batch_type : If training or testing mode.
    delta_t : Time difference between positive and negative responses
    negative_stim : If true, negative is a stimulus rather than a response
    if_continuous : If the positives have continuous index
    rng : Random number generator

  Returns:
    feed_dict : Dictionary of model placeholders assigned to triplets and
                  auxillary variables.
  """

  # Make triplets
  resp = responses[dataset_id]
  stimulus = stimuli[resp['stimulus_key']]
  if rng is None:
    rng = np.random

  if hasattr(sr_model, 'neg_model'):
    negative_resp = True
  else:
    negative_resp = False

  return_dict = get_triplets(stimulus, resp['responses'],
                          batch_size=batch_pos_sz,
                          batch_neg_resp=batch_neg_sz,
                          stim_history=30, min_window=10,
                          batch_type=batch_type,
                          negative_stim=False, negative_resp=negative_resp,
                          if_continuous=if_continuous, rng=rng)


  ## Select cells
  if frac_cells > 1.0 or frac_cells < 0.0:
    raise ValueError('Fraction of cells must be between 0 and 1')

  n_cells = np.floor(resp['centers'].shape[0]).astype(np.int)
  if frac_cells != 1.0:
    n_cells_choose = np.floor(resp['centers'].shape[0] * frac_cells).astype(np.int)

    # choose cells
    center_cell = rng.randint(0, high=n_cells, size=1)
    distances = np.sum((resp['centers'] - resp['centers'][center_cell, :]) ** 2 , 1)
    sorted_cell_ids = np.argsort(distances)
    selected_cell_ids = sorted_cell_ids[:n_cells_choose]

  else:
    selected_cell_ids = np.arange(n_cells)

  ## Make feed_dict
  ## S, R, R triplet
  stim_batch = return_dict['stim_batch']
  resp_batch_pos = return_dict['resp_batch']
  resp_batch_pos_selected = resp_batch_pos[:, selected_cell_ids]

  map_grid = resp['map_cell_grid'][:, :, selected_cell_ids]
  ctype_1hot = resp['ctype_1hot'][selected_cell_ids, :]
  mean_firing_rate = resp['mean_firing_rate'][selected_cell_ids]

  feed_dict = {sr_model.stim_tf: stim_batch,
               sr_model.anchor_model.responses_tf:
               np.expand_dims(resp_batch_pos_selected, 2),

               sr_model.anchor_model.map_cell_grid_tf:map_grid ,
               sr_model.anchor_model.cell_types_tf: ctype_1hot,
               sr_model.anchor_model.mean_fr_tf: mean_firing_rate}

  if negative_resp:
    resp_batch_neg = return_dict['resp_batch_neg']
    resp_batch_neg_selected = resp_batch_neg[:, selected_cell_ids]
    feed_dict.update({sr_model.neg_model.responses_tf:
               np.expand_dims(resp_batch_neg_selected, 2),

               sr_model.neg_model.map_cell_grid_tf:map_grid ,
               sr_model.neg_model.cell_types_tf: ctype_1hot,
               sr_model.neg_model.mean_fr_tf: mean_firing_rate})

  if hasattr(sr_model.anchor_model, 'dist_nn'):
    dist_nn = np.array([resp['dist_nn_cell_type'][1],
                        resp['dist_nn_cell_type'][2]]).astype(np.float32)
    feed_dict.update({sr_model.anchor_model.dist_nn: dist_nn})
    if negative_resp:
      feed_dict.update({sr_model.neg_model.dist_nn: dist_nn})
  if hasattr(sr_model, 'retina_indicator'):
    n_retinas = len(responses)
    retina_indicator = np.array(np.arange(n_retinas) == dataset_id).astype(np.float32)
    feed_dict.update({sr_model.retina_indicator: retina_indicator})

  # make stimulus 80 x 40, if needed
  attrs_list = [sr_model.stim_tf]

  for attrs in attrs_list:
    if attrs in feed_dict.keys():
      feed_dict[attrs] = verify_stimulus_dimensions(feed_dict[attrs],
                                                    dimx=resp['dimx_final'],
                                                    dimy=resp['dimy_final'])
  if hasattr(sr_model, 'ei_image'):
    feed_dict.update({sr_model.ei_image : resp['ei_image']})

  return feed_dict


def verify_stimulus_dimensions(stimulus, dimx, dimy):
  """Verfiy (and scale, if needed) that stimulus dimensions are dimx, dimy.

  Args :
    stimulus : stimulus array (T x X x Y x depth))
    dimx : X dimension
    dimy : Y dimension

  Returns:
    stimulus : stimulus array with desired dimensions (T x dimx x dimy x depth).
  """

  if stimulus.shape[0] != dimx or stimulus.shape[1] != dimy:
    rsz_fcn = skimage.transform.resize
    stimulus_new = np.zeros((stimulus.shape[0], dimx, dimy, stimulus.shape[3]))
    for ibatch in range(stimulus_new.shape[0]):
      for idepth in range(stimulus_new.shape[3]):
        stimulus_new[ibatch, :, :, idepth] = rsz_fcn(stimulus[ibatch, :,
                                                              :, idepth],
                                                     [dimx, dimy], order=1)

    stimulus = stimulus_new
  return stimulus
