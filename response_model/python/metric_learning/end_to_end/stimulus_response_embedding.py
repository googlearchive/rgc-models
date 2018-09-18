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
r"""Jointly embed stimulus, response from multiple retina.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import numpy as np
import random
import tensorflow as tf
from tensorflow.python.client.model_analyzer import PrintModelAnalysis
from absl import app
from absl import gfile
import retina.response_model.python.metric_learning.end_to_end.bookkeeping as bookkeeping
# pylint: disable-unused-import
import retina.response_model.python.metric_learning.end_to_end.config as config  # defines all the flags
# pylint: enable-unused-import
import retina.response_model.python.metric_learning.end_to_end.data_util as data_util
import retina.response_model.python.metric_learning.end_to_end.partitions as partitions
import retina.response_model.python.metric_learning.end_to_end.prosthesis as prosthesis
import retina.response_model.python.metric_learning.end_to_end.sr_embedding_models as sr_models
import retina.response_model.python.metric_learning.end_to_end.sr_embedding_models_experimental as sr_models_expt
import retina.response_model.python.metric_learning.end_to_end.encoding_models_experimental as encoding_models_expt
import retina.response_model.python.metric_learning.end_to_end.sr_embedding_baseline_models as sr_baseline_models
import retina.response_model.python.metric_learning.end_to_end.testing as testing
import retina.response_model.python.metric_learning.end_to_end.training as training
import retina.response_model.python.metric_learning.end_to_end.sample_datasets as sample_datasets
import retina.response_model.python.metric_learning.end_to_end.sample_datasets_2 as sample_datasets_2

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=()):

  np.random.seed(23)
  tf.set_random_seed(1234)
  random.seed(50)

  # Load stimulus-response data.
  # Collect population response across retinas in the list 'responses'.
  # Stimulus for each retina is indicated by 'stim_id',
  # which is found in 'stimuli' dictionary.
  datasets = gfile.ListDirectory(FLAGS.src_dir)
  stimuli = {}
  responses = []
  for icnt, idataset in enumerate(datasets):
    fullpath = os.path.join(FLAGS.src_dir, idataset)
    if gfile.IsDirectory(fullpath):
      key = 'stim_%d' % icnt
      op = data_util.get_stimulus_response(FLAGS.src_dir, idataset, key,
                                           boundary=FLAGS.valid_cells_boundary,
                                           if_get_stim=True)
      stimulus, resp, dimx, dimy, _ = op
      stimuli.update({key: stimulus})
      responses += resp

  # Get training and testing partitions.
  # Generate partitions
  # The partitions for the taskid should be listed in partition_file.

  op = partitions.get_partitions(FLAGS.partition_file, FLAGS.taskid)
  training_datasets, testing_datasets = op

  with tf.Session() as sess:

    # Get stimulus-response embedding.
    if FLAGS.mode == 0:
      is_training = True
    if FLAGS.mode == 1:
      is_training = True
    if FLAGS.mode == 2:
      is_training = True
      print('NOTE: is_training = True in test')
    if FLAGS.mode == 3:
      is_training = True
      print('NOTE: is_training = True in test')

    sample_fcn = sample_datasets
    if (FLAGS.sr_model == 'convolutional_embedding'):
      embedding = sr_models.convolutional_embedding(FLAGS.sr_model, sess,
                                                    is_training,
                                                    dimx, dimy)

    if (FLAGS.sr_model == 'convolutional_embedding_expt' or
        FLAGS.sr_model == 'convolutional_embedding_margin_expt' or
        FLAGS.sr_model == 'convolutional_embedding_inner_product_expt' or
        FLAGS.sr_model == 'convolutional_embedding_gauss_expt' or
        FLAGS.sr_model == 'convolutional_embedding_kernel_expt'):
      embedding = sr_models_expt.convolutional_embedding_experimental(
          FLAGS.sr_model, sess, is_training, dimx, dimy)

    if FLAGS.sr_model == 'convolutional_autoembedder':
      embedding = sr_models_expt.convolutional_autoembedder(sess, is_training,
                                                       dimx, dimy)

    if FLAGS.sr_model == 'convolutional_autoembedder_l2':
      embedding = sr_models_expt.convolutional_autoembedder(sess, is_training,
                                                            dimx, dimy,
                                                            loss='log_sum_exp')

    if FLAGS.sr_model == 'convolutional_encoder' or FLAGS.sr_model == 'convolutional_encoder_2':
      embedding = encoding_models_expt.convolutional_encoder(sess, is_training,
                                                             dimx, dimy)

    if FLAGS.sr_model == 'convolutional_encoder_using_retina_id':
      model = encoding_models_expt.convolutional_encoder_using_retina_id
      embedding = model(sess, is_training, dimx, dimy, len(responses))
      sample_fcn = sample_datasets_2

    if (FLAGS.sr_model == 'residual') or (FLAGS.sr_model == 'residual_inner_product') :
      embedding = sr_models_expt.residual_experimental(FLAGS.sr_model,
                                                       sess, is_training,
                                                       dimx, dimy)


    if FLAGS.sr_model == 'lin_rank1' or FLAGS.sr_model == 'lin_rank1_blind':
      if ((len(training_datasets) != 1) and
          (training_datasets != testing_datasets)):
        raise ValueError('Identical training/testing data'
                         ' (exactly 1) supported')

      n_cells = responses[training_datasets[0]]['responses'].shape[1]
      cell_locations = responses[training_datasets[0]]['map_cell_grid']
      cell_masks = responses[training_datasets[0]]['mask_cells']
      firing_rates = responses[training_datasets[0]]['mean_firing_rate']
      cell_type = responses[training_datasets[0]]['cell_type'].squeeze()

      model_fn = sr_baseline_models.linear_rank1_models
      embedding = model_fn(FLAGS.sr_model, sess, dimx, dimy, n_cells,
                           center_locations=cell_locations,
                           cell_masks=cell_masks,
                           firing_rates=firing_rates,
                           cell_type=cell_type, time_window=30)

    # print model graph
    PrintModelAnalysis(tf.get_default_graph())

    # Get filename, initialize model
    file_name = bookkeeping.get_filename(training_datasets,
                                         testing_datasets,
                                         FLAGS.beta, FLAGS.sr_model)
    tf.logging.info('Filename: %s' % file_name)
    saver_var, start_iter = bookkeeping.initialize_model(FLAGS.save_folder,
                                                         file_name, sess)

    # Setup summary ops.
    # Save separate summary for each retina (both training/testing).
    summary_ops = []
    for iret in np.arange(len(responses)):
      r_list = []
      r1 = tf.summary.scalar('loss_%d' % iret, embedding.loss)
      r_list += [r1]

      if hasattr(embedding, 'accuracy_tf'):
        r2 = tf.summary.scalar('accuracy_%d' % iret, embedding.accuracy_tf)
        r_list += [r2]

      if FLAGS.sr_model == 'convolutional_autoembedder' or FLAGS.sr_model =='convolutional_autoembedder_l2':
        r3 = tf.summary.scalar('loss_triplet_%d' % iret, embedding.loss_triplet)
        r4 = tf.summary.scalar('loss_stim_decode_from_resp_%d' % iret,
                               embedding.loss_stim_decode_from_resp)
        r5 = tf.summary.scalar('loss_stim_decode_from_stim_%d' % iret,
                               embedding.loss_stim_decode_from_stim)
        r6 = tf.summary.scalar('loss_resp_decode_from_resp_%d' % iret,
                               embedding.loss_resp_decode_from_resp)
        r7 = tf.summary.scalar('loss_resp_decode_from_stim_%d' % iret,
                               embedding.loss_resp_decode_from_stim)
        r_list += [r3, r4, r5, r6, r7]
        
        '''
        chosen_stim = 2
        bound = FLAGS.valid_cells_boundary
        
        r8 = tf.summary.image('stim_decode_from_stim_%d' % iret,
                              tf.expand_dims(tf.expand_dims(embedding.stim_decode_from_stim[chosen_stim, bound:80-bound, bound:40-bound, 3], 0), 3))

        r9 = tf.summary.image('stim_decode_from_resp_%d' % iret,
                              tf.expand_dims(tf.expand_dims(embedding.stim_decode_from_resp[chosen_stim, bound:80-bound, bound:40-bound, 3], 0), 3))

        r10 = tf.summary.image('resp_decode_from_stim_chann0_%d' % iret,
                              tf.expand_dims(tf.expand_dims(embedding.resp_decode_from_stim[chosen_stim, bound:80-bound, bound:40-bound, 0], 0), 3))

        r11 = tf.summary.image('resp_decode_from_resp_chann0_%d' % iret,
                              tf.expand_dims(tf.expand_dims(embedding.resp_decode_from_resp[chosen_stim, bound:80-bound, bound:40-bound, 0], 0), 3))

        r12 = tf.summary.image('resp_decode_from_stim_chann1_%d' % iret,
                              tf.expand_dims(tf.expand_dims(embedding.resp_decode_from_stim[chosen_stim, bound:80-bound, bound:40-bound, 1], 0), 3))

        r13 = tf.summary.image('resp_decode_from_resp_chann1_%d' % iret,
                              tf.expand_dims(tf.expand_dims(embedding.resp_decode_from_resp[chosen_stim, bound:80-bound, bound:40-bound, 1], 0), 3))

        r14 = tf.summary.image('resp_chann0_%d' % iret,
                              tf.expand_dims(tf.expand_dims(embedding.anchor_model.responses_embed_1[chosen_stim, bound:80-bound, bound:40-bound, 0], 0), 3))

        r15 = tf.summary.image('resp_chann1_%d' % iret,
                              tf.expand_dims(tf.expand_dims(embedding.anchor_model.responses_embed_1[chosen_stim, bound:80-bound, bound:40-bound, 1], 0), 3))

        r_list += [r8, r9, r10, r11, r12, r13, r14, r15]
        '''

      summary_ops += [tf.summary.merge(r_list)]

    # Setup summary writers.
    summary_writers = []
    for loc in ['train', 'test']:
      summary_location = os.path.join(FLAGS.save_folder, file_name,
                                      'summary_' + loc)
      summary_writer = tf.summary.FileWriter(summary_location, sess.graph)
      summary_writers += [summary_writer]

    # Separate tests for encoding or metric learning,
    #  prosthesis usage or just neuroscience usage.
    if FLAGS.mode == 3:
      testing.test_encoding(training_datasets, testing_datasets,
                            responses, stimuli,
                            embedding, sess, file_name, sample_fcn)

    elif FLAGS.mode == 2:
      prosthesis.stimulate(embedding, sess, file_name, dimx, dimy)

    elif FLAGS.mode == 1:
      testing.test_metric(training_datasets, testing_datasets,
                          responses, stimuli,
                          embedding, sess, file_name)

    else:
      training.training(start_iter, sess, embedding, summary_writers,
                        summary_ops, saver_var,
                        training_datasets, testing_datasets,
                        responses, stimuli, file_name, sample_fcn,
                        summary_freq=500, save_freq=500)


if __name__ == '__main__':
  app.run(main)
