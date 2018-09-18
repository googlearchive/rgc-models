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
r""""Run analyses on learnt metric/score function.

We load a learnt metric and test responses which are repeated
presentations of a short stimuli, and perform various analyses such as:
* Accuracy of triplet ordering.
* Precision recall analysis of triplet ordering.
* Evaluating clustering of responses generated due to same stimulus.
* Retrieval of nearest responses in training data and
    using it to decode the stimulus corresponding to test responses.
The output of all the analyses is stored in a pickle file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import pickle
import numpy as np
import tensorflow as tf
from absl import app
from absl import gfile
import retina.response_model.python.metric_learning.analyse_metric as analyse
import retina.response_model.python.metric_learning.config as config
import retina.response_model.python.metric_learning.data_util as du
import retina.response_model.python.metric_learning.score_fcns.quadratic_score as quad
import retina.response_model.python.metric_learning.score_fcns.mrf as mrf
import retina.response_model.python.metric_learning.score_fcns.hamming_distance as hamming
import retina.response_model.python.metric_learning.encoding_model as encoding_model

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=()):

  # set random seed
  np.random.seed(121)
  print('random seed reset')

  # Get details of stored model.
  model_savepath, model_filename = config.get_filepaths()

  # Load responses to two trials of long white noise.
  data_wn = du.DataUtilsMetric(os.path.join(FLAGS.data_path, FLAGS.data_test))

  # Quadratic score function.
  with tf.Session() as sess:

    # Define and restore/initialize the model.
    tf.logging.info('Model : %s ' % FLAGS.model)

    if FLAGS.model == 'quadratic':
      met = quad.QuadraticScore(sess, model_savepath,
                                model_filename,
                                n_cells=data_wn.n_cells,
                                time_window=FLAGS.time_window,
                                lr=FLAGS.learning_rate,
                                lam_l1=FLAGS.lam_l1)

    if FLAGS.model == 'mrf':
      met = mrf.MRFScore(sess, model_savepath, model_filename,
                         n_cells=data_wn.n_cells,
                         time_window=FLAGS.time_window,
                         lr=FLAGS.learning_rate,
                         lam_l1=FLAGS.lam_l1,
                         cell_centers=data_wn.get_centers(),
                         neighbor_threshold=FLAGS.neighbor_threshold)

    if FLAGS.model == 'hamming':
      met = hamming.HammingScore(model_savepath)

    # Get different types of triplets.
    np.random.seed(45)
    tf.logging.info('Random seed set')
    triplet_fcns = [data_wn.get_triplets, data_wn.get_tripletsB,
                    data_wn.get_tripletsC, data_wn.get_tripletsD]
    triplet_names = ['triplet A', 'triplet B', 'triplet C', 'triplet D']
    test_triplets = []
    for triplet_fcn in triplet_fcns:
      outputs = triplet_fcn(batch_size=FLAGS.batch_size_test,
                            time_window=FLAGS.time_window)
      anchor_test = outputs[0]
      pos_test = outputs[1]
      neg_test = outputs[2]
      test_triplets += [[anchor_test, pos_test, neg_test]]
      tf.logging.info('Got triplets')

    # from IPython import embed; embed()
    analysis_results = {}  # collect analysis results in a dictionary

    # 1. Plot distances between positive and negative pairs.
    # analyse.plot_pos_neg_distances(met, anchor_test, pos_test, neg_test)
    # tf.logging.info('Distances plotted')

    # 2. Accuracy of triplet orderings - fraction of triplets where
    # distance with positive is smaller than distance with negative.

    distances_triplets = {}
    for itest, test_data in enumerate(test_triplets):
      dist_pos, dist_neg, accuracy = analyse.compute_distances(met, *test_data)
      distances = {'pos': dist_pos,
                   'neg': dist_neg,
                   'accuracy': accuracy}
      distances_triplets.update({triplet_names[itest]: distances})
    analysis_results.update({'distances': distances_triplets})
    tf.logging.info('Accuracy computed')

    # 3. Precision-Recall analysis : declare positive if s(x,y)<t and
    # negative otherwise. Vary threshold t, and plot precision-recall and
    # ROC curves.
    pr_roc_triplets = {}
    for itest, test_data in enumerate(test_triplets):
      output = analyse.precision_recall(met, *test_data, toplot=False)
      precision_log, recall_log, f1_log, fpr_log, tpr_log = output
      pr = {'precision': precision_log, 'recall': recall_log}
      roc = {'TPR': tpr_log, 'FPR': fpr_log}
      combined_results = {'PR': pr, 'F1': f1_log, 'ROC': roc}
      pr_roc_triplets.update({triplet_names[itest]: combined_results})
    analysis_results.update({'PR_ROC': pr_roc_triplets})
    tf.logging.info('Precision Recall, F1 score and ROC curves computed')

    # 4. Distance between responses for same stimuli in different repeats.
    distances = analyse.compare_responses_across_time(met, data_wn,
                                                      n_trial_pairs=50)
    analysis_results.update({'distances_across_trials': distances})

    # TODO(bhaishahter) : 5. Is similarity in images implicitly learnt in the metric ?
    '''
    # Use NSEM! 
    from IPython import embed; embed()
    stim = data_wn.get_stimulus()
    reps = data_wn.get_repeats()
    # choose random times
    '''


    # save analysis in a pickle file
    pickle_file = (os.path.join(model_savepath, model_filename) +
                   '_analysis.pkl')
    pickle.dump(analysis_results, gfile.Open(pickle_file, 'w'))
    tf.logging.info('File: ' + pickle_file)
    tf.logging.info('Analysis results saved')

if __name__ == '__main__':
  app.run(main)
