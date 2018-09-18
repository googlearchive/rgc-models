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
"""Train the joint embedding model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import retina.response_model.python.metric_learning.end_to_end.bookkeeping as bookkeeping

FLAGS = tf.app.flags.FLAGS


def training(start_iter, sess, embedding, summary_writers,
             summary_ops, saver_var,
             training_datasets, testing_datasets,
             responses, stimuli, file_name, sample_fcn,
             summary_freq=10,
             save_freq=10):
  """Train the joint embedding with frequent summaries and saves.

  Args :
    start_iter : Starting iteration number,
                  might not be 0 if a previously trained model loaded.
    sess : Tensorflow session.
    embedding : Collection of TF Ops for embedding.
    summary_writers : TF summary writers.
    summary_ops : TF Ops to to evaluate loss and accuracy for different retinas.
    saver_var : TF Op for saving model parameters.
    training_datasets : index of retinas in 'responses' to use in training.
    testing_datasets : index of retinas in 'responses' to use in testing/eval.
    responses : List of responses.
    stimuli : Dictionary of differnet stimuli used.
    file_name : Filename to save the model and summaries.
    summary_freq : Frequency of writing the summaries.
    save_freq : Frequency of saving the model.
  """

  # Start training.
  batch_neg_train_sz = FLAGS.batch_neg_train_sz
  batch_train_sz = FLAGS.batch_train_sz

  # Keep track of how many times testing done as we
  # want to evaluate a different retina evertime.
  test_iiter = 0
  frac_cells_train = 1.0
  for iiter in range(start_iter, FLAGS.max_iter):
    if FLAGS.subsample_cells_train:
      frac_cells_train = np.random.uniform(low=0.7, high=1.0)

    # Get a new batch.
    train_dataset = training_datasets[iiter % len(training_datasets)]

    print('Train: %d' % train_dataset)
    feed_dict_train = sample_fcn.batch(stimuli, responses,
                                       train_dataset, embedding,
                                       batch_train_sz,
                                       batch_neg_train_sz,
                                       batch_type='train',
                                       frac_cells=frac_cells_train)

    # Training step.
    _, l_tr = sess.run([embedding.train_op, embedding.loss],
                    feed_dict=feed_dict_train)
    #print('Retina: %d, l_tr: %.3f:, retina_params, accuracy: %s' % (train_dataset, l_tr,
    #                                                      sess.run([embedding.retina_params, embedding.accuracy_tf], feed_dict=feed_dict_train)))

    # Write summaries.
    if iiter % summary_freq == 0:

      test_iiter += 1

      # Write summary on test data in training datasets.
      for train_dataset in training_datasets: # [test_iiter % len(training_datasets)]
        print('Testing, Train dataset: %d' % train_dataset)
        feed_dict_train = sample_fcn.batch(stimuli, responses,
                                           train_dataset, embedding,
                                           batch_train_sz,
                                           batch_neg_train_sz,
                                           batch_type='test')
        summary_train = sess.run(summary_ops[train_dataset],
                                 feed_dict=feed_dict_train)
        summary_writers[0].add_summary(summary_train, iiter)

      # Write summary on test data in testing datasets.
      for test_dataset in testing_datasets: # [test_iiter % len(testing_datasets)]
        print('Testing, Test dataset: %d' % test_dataset)
        feed_dict_test = sample_fcn.batch(stimuli, responses,
                                          test_dataset, embedding,
                                          batch_train_sz,
                                          batch_neg_train_sz,
                                          batch_type='test')

        l_test, summary_test = sess.run([embedding.loss,
                                         summary_ops[test_dataset]],
                                        feed_dict=feed_dict_test)
        summary_writers[1].add_summary(summary_test, iiter)
        print('Iteration: %d, Test retina: %d, loss: %.3f' % (iiter,
                                                              test_dataset,
                                                              l_test))

    # save model
    if iiter % save_freq == 0:
      bookkeeping.save_model(saver_var, FLAGS.save_folder,
                             file_name, sess, iiter)


