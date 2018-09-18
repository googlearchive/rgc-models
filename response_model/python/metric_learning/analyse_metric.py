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
"""Multiple functions to analyse the metric learnt."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def plot_pos_neg_distances(metric, anchor, pos, neg):
  """Plot distances anchor-positive v/s anchor-negative."""

  distances_pos = metric.get_distance(anchor, pos)
  distances_neg = metric.get_distance(anchor, neg)

  # Plot
  plt.figure()
  plt.plot(distances_pos, distances_neg, 'r.')
  plt.hold(True)
  plt.plot([0, np.max(distances_neg)], [0, np.max(distances_neg)], 'g')
  plt.xlim([0, np.max(distances_neg)])
  plt.ylim([0, np.max(distances_neg)])
  plt.xlabel(['Positive'])
  plt.ylabel(['Negative'])
  plt.show()
  plt.draw()

def compute_distances(metric, anchor, pos, neg):

  batchsz = 100
  distances_pos = np.zeros(anchor.shape[0])
  distances_neg = np.zeros(anchor.shape[0])
  for ibatch_start in np.arange(0, anchor.shape[0], batchsz):
    print(ibatch_start)
    distances_pos[ibatch_start: ibatch_start+batchsz] = metric.get_distance(anchor[ibatch_start: ibatch_start+batchsz, :, :],
                                                                            pos[ibatch_start: ibatch_start+batchsz, :, :])
    distances_neg[ibatch_start: ibatch_start+batchsz] = metric.get_distance(anchor[ibatch_start: ibatch_start+batchsz, :, :],
                                                                            neg[ibatch_start: ibatch_start+batchsz, :, :])

  # compute accuracy for correctly identified orderings
  nnz = np.logical_or(distances_pos != 0 , distances_neg != 0)
  num_correct = np.sum(distances_pos[nnz] <= distances_neg[nnz])
  num_wrong = np.sum(distances_pos[nnz] > distances_neg[nnz])
  accuracy = num_correct / (num_correct + num_wrong)

  return distances_pos, distances_neg, accuracy

def precision_recall(metric, anchor, pos, neg, toplot=False):


  batchsz = 10
  distances_pos = np.zeros(anchor.shape[0])
  distances_neg = np.zeros(anchor.shape[0])
  for ibatch_start in np.arange(0, anchor.shape[0], batchsz):
    print(ibatch_start)
    distances_pos[ibatch_start: ibatch_start+batchsz] = metric.get_distance(anchor[ibatch_start: ibatch_start+batchsz, :, :],
                                                                            pos[ibatch_start: ibatch_start+batchsz, :, :])
    distances_neg[ibatch_start: ibatch_start+batchsz] = metric.get_distance(anchor[ibatch_start: ibatch_start+batchsz, :, :],
                                                                            neg[ibatch_start: ibatch_start+batchsz, :, :])

  pr_data = {'anchor': anchor, 'pos': pos, 'neg': neg,
             'distances_ap': distances_pos, 'distances_an': distances_neg}
  nnz = np.logical_and(distances_pos != 0, distances_neg != 0)

  if np.sum(nnz)==0:
    tf.logging.info('All distances 0')
    precision_log = None
    recall_log = None
    F1_log = None
    FPR_log = None
    TPR_log = None
    return precision_log, recall_log, F1_log, FPR_log, TPR_log, pr_data

  distances_pos = distances_pos[nnz]
  distances_neg = distances_neg[nnz]

  all_distances = np.append(distances_pos, distances_neg)
  precision_log = []
  recall_log = []
  F1_log = []
  TPR_log = []
  FPR_log = []

  for iprc in np.arange(0,100,1):
    ithr = np.percentile(all_distances, iprc)
    TP = np.sum(distances_pos <= ithr)
    FP = np.sum(distances_neg <= ithr)
    FN = np.sum(distances_pos > ithr)
    TN = np.sum(distances_neg > ithr)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    precision_log += [precision]
    recall_log += [recall]
    F1_log += [F1]
    TPR_log += [TPR]
    FPR_log += [FPR]

  if toplot:
    # precision recall curve
    plt.figure()
    plt.plot(precision_log, recall_log, 'r')
    plt.title('Precision-Recall')
    plt.show()
    plt.draw()

    # ROC curve
    plt.figure()
    plt.plot(FPR_log, TPR_log, 'r')
    plt.title('ROC curve')
    plt.hold(True)
    plt.plot([0, 1], [0, 1], 'g')
    plt.show()
    plt.draw()


  return precision_log, recall_log, F1_log, FPR_log, TPR_log, pr_data

def get_pairwise_distances(metric, resp_all_trials):
  """Does not work when we have batch norm in embedding!! """

  n_responses = resp_all_trials.shape[0]
  # from IPython import embed; embed()
  resp_order0 = np.random.permutation(np.arange(n_responses))
  resp_order1 = np.random.permutation(np.arange(n_responses))

  resp_order0 = np.expand_dims(resp_order0, 1)
  resp_order1 = np.expand_dims(resp_order1, 0)

  resp_order0 = np.repeat(resp_order0, n_responses, 1)
  resp_order1 = np.repeat(resp_order1, n_responses, 0)

  resp_order0 = np.ndarray.flatten(resp_order0)
  resp_order1 = np.ndarray.flatten(resp_order1)

  batch_sz = 100

  distances = np.zeros((n_responses, n_responses))
  for itime in range(0, resp_order0.shape[0], batch_sz):
    print(itime)
    ti = resp_order0[np.arange(itime, itime + batch_sz)]
    tj = resp_order1[np.arange(itime, itime + batch_sz)]
    dists = metric.get_distance(resp_all_trials[ti, :, :], resp_all_trials[tj, :, :])
    for ii in range(batch_sz):
      distances[ti[ii], tj[ii]] = dists[ii]

  '''
  distances = np.zeros((n_responses, n_responses))
  for iresp in range(n_responses):
    print(iresp)
    a1 = np.repeat(np.expand_dims(resp_all_trials[iresp, :, :], 0),
                   n_responses, 0)

    # tf.logging.info(iresp)
  '''
  return distances

def topK_retrieval(distance_pairs, K, stim_id):
  """Do retrieval analysis given all pairwise distances.

    Given distances between all pairs of responses,
    retrive top K other responses for each response point,
    and compute how many of them are generated by the same stimulus.

    Args :
        distance_pairs (np.float32): Square matrix of pairwise distances
                                       between responses.
        K (int) : Number of responses to retrieve for each probe response.
        stim_id (np.int) : The stimulus corresponding to each response.
  """
  precision_log = np.zeros(distance_pairs.shape[0])
  recall_log = np.zeros(distance_pairs.shape[0])
  for iresponse in range(distance_pairs.shape[0]):
    idx_sorted = np.argsort(distance_pairs[iresponse, :])[1:K+1]
    precision_log[iresponse] = np.sum(stim_id[idx_sorted] == stim_id[iresponse])/K

    true_pos_idx = np.where(stim_id[iresponse] == stim_id)[0]
    true_pos_idx = [x for x in true_pos_idx if x != iresponse]
    true_pos_retrieved = [x for x in true_pos_idx if (x in idx_sorted)]
    recall_log[iresponse] = len(true_pos_retrieved)/len(true_pos_idx)

  return precision_log, recall_log

def topK_retrieval_probes(corpus, corpus_stim, probes, K, met):
  """ Retrieve nearest responses in corpus for probe responses.

  Args :
      corpus :
      corpus_stim : Stimulus identifier index of each resposne in corpus.
      probes :
      K :
      met :
  Returns :
      retrieved : Top K retrieved responses.
      retrieved_stim_idx : Corresponding stimulus index of retrieved.
  """

  retrieved = []
  retrieved_stim_idx = []
  for iprobe in range(probes.shape[0]):
    # print(iprobe)
    in2 = np.repeat(np.expand_dims(probes[iprobe, :, :], 0),
                    corpus.shape[0], axis=0)
    distances = met.get_distance(corpus, in2)
    I = np.argsort(distances)
    retrieved += [corpus[I[:K], :, :]]
    retrieved_stim_idx += [corpus_stim[I[:K]]]

  return retrieved, retrieved_stim_idx

def compute_all_distances(corpus, probes, met):
  """ Compute distances for responses in corpus to those in probes using met.

  Args :
      corpus : Collection of responses to compute distances from.
                 Most generally, its either ALL possible population resposnes or
                 ALL responses in training data.
      probes : compute distances of probe responses to all in corpus.
      met : A score/metric function
  Returns :
      distances : List of distances for each probe to all responses in corpus.

  """

  distances = []
  for iprobe in range(probes.shape[0]):
    # print(iprobe)
    in2 = np.repeat(np.expand_dims(probes[iprobe, :, :], 0),
                    corpus.shape[0], axis=0)
    distances += [met.get_distance(corpus, in2)]


  return distances


  # TODO(bhaishahster) : Score v/s MSE of images.
def compare_stimulus_score_similarity(data_wn, stimuli_score,
                                      resp_score):
  """Relate distances between responses and corresponding stimuli. """

  batch0 = data_wn.get_stimulus_response_samples(batch_size=
                                                 FLAGS.batch_size_test,
                                                 time_window=
                                                 FLAGS.time_window)

  stimulus_examples0, response_examples0, time_log0, trial_log0 = batch0

  batch1 = data_wn.get_stimulus_response_samples(batch_size=
                                                 FLAGS.batch_size_test,
                                                 time_window=
                                                 FLAGS.time_window)
  stimulus_examples1, response_examples1, time_log1, trial_log1 = batch1

  """Compare similarity of stimuli with score on corresponding responses."""
  stim_distance = stimuli_score.get_distance(stimulus_examples0,
                                             stimulus_examples1)
  resp_distance = resp_score.get_distance(response_examples0,
                                          response_examples1)
  times = [time_log0, time_log1]
  responses = [response_examples0, response_examples1]
  return stim_distance, resp_distance, times, responses

def compare_responses_across_time(metric, data, n_trial_pairs=10):
  """Compare responses generated for whole stimulus across trials."""

  distances = []
  for _ in range(n_trial_pairs):
    n_trials = 2
    resps_few_trials, _ = data.get_all_response_few_trials(n_trials,
                                                           FLAGS.time_window)

    distances += [metric.get_distance(resps_few_trials[0, :, :, :],
                                      resps_few_trials[1, :, :, :])]

  return np.array(distances)

  # TODO(bhaishahster) : Pairwise distances - see clustering.

  # TODO(bhaishahster) : tSNE embedding

  # TODO(bhaishahster) : Precision - recall analysis

  # TODO(bhaishahster) : Retrieval analysis

 # TODO(bhaishahster) : 
