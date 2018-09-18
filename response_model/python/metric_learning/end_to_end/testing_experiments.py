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
r"""Run experiments on stimulus-response embedding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import retina.response_model.python.metric_learning.end_to_end.sample_datasets as sample_datasets
import retina.response_model.python.metric_learning.end_to_end.analysis_utils as analysis_utils
import retina.response_model.python.metric_learning.end_to_end.bookkeeping as bookkeeping
from absl import gfile
import pickle
import os.path

FLAGS = tf.app.flags.FLAGS


def stimulus_embedding_expt(stimuli, sr_graph, n_stims_per_type=2000):
  """Embedd n_stims_per_type examples for each different type of stimuli.

  Args:
    stimuli : Dictionary of different stimuli (key, numpy array).
    sr_graph : TF graph which embeds stimuli.
    n_stims_per_type : Number of stimuli of each type to embed.

  Returns :
    stim_embedding_dict : Dictionary embeddings of stimulus and other params.
  """

  ## Jointly embed NSEM and WN stimuli
  # 100 random test examples for NSEM and WN each

  stimulus_test, stim_type = analysis_utils.sample_stimuli(stimuli,
                                                           n_stims_per_type=
                                                           n_stims_per_type,
                                                           t_min=FLAGS.test_min,
                                                           t_max=FLAGS.test_max,
                                                           dimx=80, dimy=40,
                                                           tlen=30)

  # random permutation of stimuli
  stim_test_embed = analysis_utils.embed_stimuli(stimulus_test, sr_graph,
                                                 embed_batch=100)

  # write tensorboard summary
  '''
  bookkeeping.write_tensorboard(sr_graph.sess,
                                np.reshape(stim_test_embed, [-1, 200]),
                                stim_type, stimulus_test[:, :, :, 4],
                                embedding_name='stim_embed',
                                log_dir=
                                '/home/bhaishahster/tb')
  '''

  stim_test_embed_2d = np.reshape(stim_test_embed,
                                  [stim_test_embed.shape[0], -1])

  # get PCA embedding
  pca_embedding = analysis_utils.get_pca(stim_test_embed_2d, n_components=2)

  # get tSNE embedding
  tSNE_embedding = analysis_utils.get_tsne(stim_test_embed_2d, n_components=2)

  # save results in a dictionary
  stim_embedding_dict = {'stim_type': stim_type,
                         'stim_test_embed': stim_test_embed,
                         'tSNE_embedding': tSNE_embedding,
                         'pca_embedding' : pca_embedding,
                         'n_stims_per_type': n_stims_per_type}

  return stim_embedding_dict


def stimulus_transformations_expt(stimuli, sr_graph,
                                  n_stims_per_type_transform=50,
                                  n_stims_per_type_bkg=2000):

  stimulus_test, stim_type = analysis_utils.sample_stimuli(stimuli,
                                                           n_stims_per_type=
                                                           n_stims_per_type_transform,
                                                           t_min=FLAGS.test_min,
                                                           t_max=FLAGS.test_max,
                                                           dimx=80, dimy=40,
                                                           tlen = 30)
  # Apply transformations
  transform_dict = {'Luminance': 1, 'Contrast': 2, 'Translation': 3, 'Original': 0}
  transform_levels = {}
  stimulus_transformed = np.copy(stimulus_test)
  stimulus_orig_idx = np.arange(stimulus_test.shape[0])
  transform_type = transform_dict['Original'] + np.zeros(stimulus_test.shape[0])

  # Luminance transformation
  luminance_levels = [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]
  transform_levels.update({'Luminance': luminance_levels})
  stimulus_luminance, original_idx = analysis_utils.transform_stimulus(stimulus_test,
                                                                       levels=luminance_levels,
                                                                       transform_type='luminance')
  stimulus_transformed = np.append(stimulus_transformed, stimulus_luminance, axis=0)
  stimulus_orig_idx = np.append(stimulus_orig_idx, original_idx)
  transform_type = np.append(transform_type, transform_dict['Luminance'] + np.zeros_like(original_idx))

  # Contrast transformation
  contrast_levels = [0.5, 0.7, 0.9, 0.95, 1.05, 1.1, 1.3, 1.5]
  transform_levels.update({'Contrast': contrast_levels})
  stimulus_contrast, original_idx = analysis_utils.transform_stimulus(stimulus_test,
                                                                      levels=contrast_levels,
                                                                      transform_type='contrast')
  stimulus_transformed = np.append(stimulus_transformed, stimulus_contrast, axis=0)
  stimulus_orig_idx = np.append(stimulus_orig_idx, original_idx)
  transform_type = np.append(transform_type, transform_dict['Contrast'] + np.zeros_like(original_idx))

  # Translation transformation
  translate_levels = [-7, -5, -1, 1, 5, 7]
  transform_levels.update({'Translation': translate_levels})
  stimulus_translate, original_idx = analysis_utils.transform_stimulus(stimulus_test,
                                                                       levels=translate_levels,
                                                                       transform_type='translate')
  stimulus_transformed = np.append(stimulus_transformed, stimulus_translate, axis=0)
  stimulus_orig_idx = np.append(stimulus_orig_idx, original_idx)
  transform_type = np.append(transform_type, transform_dict['Translation'] + np.zeros_like(original_idx))

  # Get stimuli which form the background
  stimulus_test_background, stim_type_background = analysis_utils.sample_stimuli(stimuli,
                                                           n_stims_per_type=
                                                           n_stims_per_type_bkg,
                                                           t_min=0,
                                                           t_max=30000,
                                                           dimx=80, dimy=40,
                                                           tlen = 30)
  stim_type = np.append(stim_type, stim_type_background)
  stimulus_combined = np.append(stimulus_transformed, stimulus_test_background, axis=0)
  stimulus_untransformed = np.append(stimulus_test, stimulus_test_background, axis=0)
  stimulus_orig_idx = np.append(stimulus_orig_idx, np.arange(np.max(stimulus_orig_idx),
                                                             stimulus_test_background.shape[0] +
                                                             np.max(stimulus_orig_idx)))
  transform_type = np.append(transform_type, np.zeros(stimulus_test_background.shape[0]))

  # Embed stimuli
  stim_embed = analysis_utils.embed_stimuli(stimulus_combined, sr_graph,
                                             embed_batch=100)
  stim_embed_2d = np.reshape(stim_embed,
                             [stim_embed.shape[0], -1])

  # Get PCA embedding
  pca_embedding = analysis_utils.get_pca(stim_embed_2d, n_components=2)


  # Get tSNE embedding
  tSNE_embedding = analysis_utils.get_tsne(stim_embed_2d, n_components=2)

  # Save results
  stim_embedding_dict = {'stim_type': stim_type,
                         'transform_dict': transform_dict,
                         'stimulus_orig_idx': stimulus_orig_idx,
                         'transform_type': transform_type,
                         'stim_embed': stim_embed,
                         'tSNE_embedding': tSNE_embedding,
                         'pca_embedding': pca_embedding,
                         'transform_levels': transform_levels}

  return stim_embedding_dict


def stimulus_response_embedding_expt(stimuli, responses, sr_graph,
                                     stim_key='stim_2', n_samples=1000,
                                     if_continuous=False):

  # from IPython import embed; embed()
  
  ## Embed multiple retinas
  t_min = FLAGS.test_min
  t_max = FLAGS.test_max

  # Random times
  if if_continuous:
    rand_sample = np.random.randint(30, t_max - t_min, 1)
    rand_sample = np.squeeze(rand_sample)
    rand_sample = np.arange(rand_sample, rand_sample + n_samples)
    rand_sample = np.random.permutation(rand_sample)
  else:
    rand_sample = np.random.randint(30, t_max - t_min, n_samples)

  ## Embed responses across multiple retina.
  embed_x = sr_graph.stim_embed.shape[-3].value
  embed_y = sr_graph.stim_embed.shape[-2].value
  embed_t = sr_graph.stim_embed.shape[-1].value
  resp_embed = np.zeros((0, embed_x, embed_y, embed_t))
  stim_decode_from_resp_test = np.zeros((0, 80, 40, 30))

  retinas = [i for i in np.arange(len(responses)) if
             responses[i]['stimulus_key'] == stim_key]
  retina_labels = np.repeat(retinas, n_samples, 0)

  resp_sampled_log = []
  centers_log = []
  cell_type_log = []
  for iretina in retinas:
    print(iretina)
    resp = responses[iretina]
    resp_sampled_log += [resp['responses'][rand_sample, :]]
    centers_log += [resp['centers']]
    cell_type_log += [resp['cell_type']]

    #rr, decoded_resp = sr_graph.sess.run([sr_graph.anchor_model.responses_embed,
    #                                      sr_graph.stim_decode_from_resp],
    #                       feed_dict=feed_dict)

    rr = np.zeros((n_samples, embed_x, embed_y, embed_t))
    for istart in np.arange(0, n_samples, 100):

      feed_dict={sr_graph.anchor_model.map_cell_grid_tf: resp['map_cell_grid'],
               sr_graph.anchor_model.cell_types_tf: resp['ctype_1hot'],
               sr_graph.anchor_model.mean_fr_tf: resp['mean_firing_rate'],
               sr_graph.anchor_model.responses_tf:
               np.expand_dims(resp['responses'][rand_sample[istart: istart + 100], :], 2)}

      if hasattr(sr_graph.anchor_model, 'dist_nn'):
        dist_nn = np.array([resp['dist_nn_cell_type'][1],
                          resp['dist_nn_cell_type'][2]]).astype(np.float32)
        feed_dict.update({sr_graph.anchor_model.dist_nn: dist_nn,
                        sr_graph.neg_model.dist_nn: dist_nn})

      rr[istart: istart + 100, : ] = sr_graph.sess.run(sr_graph.anchor_model.responses_embed,
                                                       feed_dict=feed_dict)
    resp_embed = np.append(resp_embed, rr.squeeze(), 0)

    # collect stimuli decoded from responses
    #stim_decode_from_resp_test = np.append(stim_decode_from_resp_test,
    #                                       decoded_resp, 0)

  # linear discriminator to see how responses separate in embedded space
  '''
  resp_embed_2d_discriminator = analysis_utils.get_linear_discriminator(resp_embed,
                                                         retina_labels,
                                                         n_components=2)
  '''
  ## Embed stimuli
  stimulus_test = np.zeros((n_samples, 80, 40, 30))
  for istim in range(n_samples):
    stim_sample = stimuli[stim_key][rand_sample[istim]: rand_sample[istim]-30:-1, :, :]
    stim_sample = np.expand_dims(np.transpose(stim_sample, [1, 2, 0]), 0)
    stim_sample = sample_datasets.verify_stimulus_dimensions(stim_sample, dimx=80, dimy=40)
    stimulus_test[istim, :, :, :] = stim_sample

  #stim_embed, stim_decode_from_stim_test = sr_graph.sess.run([sr_graph.stim_embed,sr_graph.stim_decode_from_stim],
  #                               feed_dict = {sr_graph.stim_tf: stimulus_test})
  stim_embed = sr_graph.sess.run(sr_graph.stim_embed,
                                 feed_dict = {sr_graph.stim_tf: stimulus_test})

  stim_embed = np.squeeze(stim_embed)

  embedding = np.append(resp_embed, stim_embed, 0)

  labels = np.append(retina_labels, np.repeat(-1, n_samples))  # -1 for stimulus


  # write tensorboard summary
  '''
  bookkeeping.write_tensorboard(sr_graph.sess, np.reshape(embedding, [-1, 200]),
                                labels,
                                embedding_name='multiple_retina_embed',
                                log_dir='/home/bhaishahster/tb',
                                model_name='model_resp')
  '''

  stim_resp_embedding = {'stim_embed': stim_embed,
                         'resp_embed': resp_embed,
                         'retina_labels': retina_labels,
                         'rand_sample': rand_sample,
                         'resp_sampled_log': resp_sampled_log,
                         'centers_log': centers_log,
                         'cell_type_log': cell_type_log}
                         #'resp_embed_2d_discriminator':
                         #resp_embed_2d_discriminator,
                         #'stim_decode_from_resp_test':
                         #stim_decode_from_resp_test,
                         #'stim_decode_from_stim_test':
                         #stim_decode_from_stim_test}


  return stim_resp_embedding


def roc_analysis(dataset_dict, stimuli, responses,
                 sr_graph, num_examples=1000, delta_t=None, frac_cells=1.0, negative_stim=False):
  ## compute distances between s-r pairs - pos and neg.
  ## negative_stim - if the negative is a stimulus or a response
  if num_examples % 100 != 0:
    raise ValueError('Only supports examples which are multiples of 100.')

  save_dict = {}
  for dat_key, datasets in dataset_dict.items():
    test_retina = []

    for iretina in range(len(datasets)):
      batch_type_dict = {}
      for batch_type in ['test', 'train']:
        # stim-resp log
        d_pos_log = np.array([])
        d_neg_log = np.array([])
        for ibatch in range(np.floor(num_examples / 100).astype(np.int)):
          print(ibatch)
          feed_dict = sample_datasets.batch(stimuli, responses,
                                            datasets[iretina], sr_graph,
                                            batch_pos_sz=100,
                                            batch_neg_sz=100,
                                            batch_type=batch_type,
                                            delta_t=delta_t,
                                            frac_cells=frac_cells,
                                            negative_stim=negative_stim)
          if not negative_stim:  # S, R, R triplet
            d_pos, d_neg = sr_graph.sess.run([sr_graph.d_s_r_pos,
                                                       sr_graph.d_pairwise_s_rneg],
                                                      feed_dict=feed_dict)

          if negative_stim:  # R, S, S triplet
            d_pos, d_neg = sr_graph.sess.run([sr_graph.d_r_s_pos_rss,
                                                       sr_graph.d_pairwise_r_sneg_rss],
                                                      feed_dict=feed_dict)

          d_neg = np.diag(d_neg)  # np.mean(d_neg, 1) #
          d_pos_log = np.append(d_pos_log, d_pos)
          d_neg_log = np.append(d_neg_log, d_neg)

        op_ = analysis_utils.ROC(d_pos_log, d_neg_log)
        precision_log, recall_log, f1_log, fpr_log, tpr_log = op_
        accuracy = np.mean(d_pos_log < d_neg_log)

        print('Datasets: %s, Piece: %s, Batch_type: %s Accuracy : %.3f' %
              (dat_key, responses[datasets[iretina]]['piece'],
               batch_type,
               accuracy))

        test_sr = {'precision': precision_log, 'recall': recall_log,
                   'F1': f1_log, 'FPR': fpr_log, 'TPR': tpr_log,
                   'd_pos_log': d_pos_log, 'd_neg_log': d_neg_log,
                   'piece': responses[datasets[iretina]]['piece'],
                   'accuracy': accuracy}

        batch_type_dict.update({batch_type: test_sr})

      test_retina += [batch_type_dict]
      save_dict.update({dat_key: test_retina})

  return save_dict

def response_transformation_increase_nl(stimuli, responses, sr_graph,
                                        time_start_list,
                                        time_len=100,
                                        alpha_list=[1.5, 1.25, 0.8, 0.6]):
  # 1. Take an LN model and increase non-linearity.
  # How do the points move in response space?

  # Load LN models
  ln_save_folder = '/home/bhaishahster/stim-resp_collection_ln_model_exp'
  files = gfile.ListDirectory(ln_save_folder)

  ln_models = []
  for ifile in files:
    print(ifile)
    ln_models += [pickle.load(gfile.Open(os.path.join(ln_save_folder, ifile), 'r'))]

  t_start_dict = {}
  t_min = FLAGS.test_min
  t_max = FLAGS.test_max

  for time_start in time_start_list:

    print('Start time %d' % time_start)
    retina_log = []
    for iretina_test in range(3, len(responses)):

      print('Retina: %d' % iretina_test)
      piece_id = responses[iretina_test]['piece']

      # find piece in ln_models
      matched_ln_model = [ifile for ifile in range(len(files)) if  files[ifile][:12] == piece_id[:12]]
      if len(matched_ln_model) == 0 :
        print('LN model not found')
        continue
      if len(matched_ln_model) > 1:
        print('More than 1 LN model found')

      # Sample a sequence of stimuli and predict spikes
      iresp = responses[iretina_test]
      iln_model = ln_models[matched_ln_model[0]]
      stimulus_test = stimuli[iresp['stimulus_key']]

      stim_sample = stimulus_test[time_start: time_start + time_len, :, :]
      spikes, lam_np = analysis_utils.predict_responses_ln(stim_sample,
                                                           iln_model['k'],
                                                           iln_model['b'],
                                                           iln_model['ttf'],
                                                           n_trials=1)
      spikes_log = np.copy(spikes[0, :, :])
      alpha_log = np.ones(time_len)

      # Increase nonlinearity, normalize firing rate and embed.

      for alpha in alpha_list:
        _, lam_np_alpha = analysis_utils.predict_responses_ln(stim_sample,
                                                              alpha * iln_model['k'],
                                                              alpha * iln_model['b'],
                                                              iln_model['ttf'],
                                                              n_trials=1)
        correction_firing_rate = np.mean(lam_np)/ np.mean(lam_np_alpha)
        correction_b = np.log(correction_firing_rate)
        spikes_corrected, lam_np_corrected = analysis_utils.predict_responses_ln(stim_sample,
                                                           alpha * iln_model['k'],
                                                           alpha * iln_model['b'] + correction_b,
                                                           iln_model['ttf'],
                                                           n_trials=1)
        print(alpha, np.mean(lam_np), np.mean(lam_np_alpha), np.mean(lam_np_corrected))
        spikes_log = np.append(spikes_log, spikes_corrected[0, :, :], axis=0)
        alpha_log = np.append(alpha_log, alpha * np.ones(time_len), axis=0)

        # plt.figure()
        # analysis_utils.plot_raster(spikes_corrected[:, :, 23])
        # plt.title(alpha)

      # Embed responses
      try:
        resp_trans = np.expand_dims(spikes_log[:, iresp['valid_cells']], 2)
        feed_dict={sr_graph.anchor_model.map_cell_grid_tf: iresp['map_cell_grid'],
                   sr_graph.anchor_model.cell_types_tf: iresp['ctype_1hot'],
                   sr_graph.anchor_model.mean_fr_tf: iresp['mean_firing_rate'],
                   sr_graph.anchor_model.responses_tf:resp_trans}

        if hasattr(sr_graph.anchor_model, 'dist_nn'):
          dist_nn = np.array([iresp['dist_nn_cell_type'][1],
                              iresp['dist_nn_cell_type'][2]]).astype(np.float32)
          feed_dict.update({sr_graph.anchor_model.dist_nn: dist_nn,
                            sr_graph.neg_model.dist_nn: dist_nn})

        rr = sr_graph.sess.run(sr_graph.anchor_model.responses_embed,
                             feed_dict=feed_dict)


        retina_log += [{'spikes_log': spikes_log, 'alpha_log': alpha_log,
                      'resp_embed': rr, 'piece':  piece_id}]

      except:
        print('Error! ')
        retina_log += [np.nan]
        pass

    t_start_dict.update({time_start: retina_log})

  return t_start_dict


def resp_drop_cells_expt(stimuli, responses, sr_graph,
                         stim_key='stim_2', n_samples=1000):

  from IPython import embed; embed()

  ## Embed multiple retinas
  t_min = FLAGS.test_min
  t_max = FLAGS.test_max

  # Random times
  rand_sample = np.random.randint(30, t_max - t_min, n_samples)

  ## Embed responses across multiple retina.
  embed_x = sr_graph.stim_embed.shape[-3].value
  embed_y = sr_graph.stim_embed.shape[-2].value
  embed_t = sr_graph.stim_embed.shape[-1].value
  resp_embed = np.zeros((0, embed_x, embed_y, embed_t))

  retinas = [i for i in np.arange(len(responses)) if
             responses[i]['stimulus_key'] == stim_key]


  retina_labels = np.array([])
  cell_pop_label = np.array([])

  for iretina in retinas:
    print(iretina)
    resp = responses[iretina]

    I = np.eye(resp['responses'].shape[1])

    not_cell_type1 = np.where(resp['ctype_1hot'][:, 0] == 0)[0]
    I1 = np.copy(I)
    I1[not_cell_type1, not_cell_type1] = 0

    not_cell_type2 = np.where(resp['ctype_1hot'][:, 1] == 0)[0]
    I2 = np.copy(I)
    I2[not_cell_type2, not_cell_type2] = 0

    select_cell_mat = [I.astype(np.float32), I1.astype(np.float32),
                       I2.astype(np.float32)]

    for isel_mat, sel_mat in enumerate(select_cell_mat):
      feed_dict={sr_graph.anchor_model.map_cell_grid_tf: resp['map_cell_grid'],
                 sr_graph.anchor_model.cell_types_tf: resp['ctype_1hot'],
                 sr_graph.anchor_model.mean_fr_tf: resp['mean_firing_rate'],
                 sr_graph.anchor_model.responses_tf:
                 np.expand_dims(resp['responses'][rand_sample, :].dot(sel_mat), 2)}

      if hasattr(sr_graph.anchor_model, 'dist_nn'):
        dist_nn = np.array([resp['dist_nn_cell_type'][1],
                            resp['dist_nn_cell_type'][2]]).astype(np.float32)
        feed_dict.update({sr_graph.anchor_model.dist_nn: dist_nn,
                          sr_graph.neg_model.dist_nn: dist_nn})

      rr = sr_graph.sess.run(sr_graph.anchor_model.responses_embed,
                             feed_dict=feed_dict)
      resp_embed = np.append(resp_embed, rr.squeeze(), 0)
      retina_labels = np.append(retina_labels, iretina * np.ones(rr.shape[0]), 0)
      cell_pop_label = np.append(cell_pop_label, isel_mat * np.ones(rr.shape[0]), 0)


  stim_resp_embedding = {'cell_pop_label': cell_pop_label,
                         'resp_embed': resp_embed,
                         'retina_labels': retina_labels,
                         'rand_sample': rand_sample}


  return stim_resp_embedding

def resp_wn_repeats_multiple_retina(sr_graph, n_samples=100):

  ## Embed responses across multiple retina.
  embed_x = sr_graph.stim_embed.shape[-3].value
  embed_y = sr_graph.stim_embed.shape[-2].value
  embed_t = sr_graph.stim_embed.shape[-1].value
  resp_embed = np.zeros((0, embed_x, embed_y, embed_t))
  retina_labels = np.zeros((0))
  stimulus_labels = np.zeros((0))
  repeat_labels = np.zeros((0))

  # Get repeats data - prepare it!
  import retina.response_model.python.metric_learning.end_to_end.data_util as data_util
  op = data_util.get_stimulus_response('/home/bhaishahster/end_to_end_wn_repeats/', 'bw-8-4', 'bw-8-4-reps',
                                       boundary=FLAGS.valid_cells_boundary)
  stimulus, responses, dimx, dimy, _ = op

  # Random times
  t_len = responses[0]['repeats'].shape[1]
  rand_sample = np.random.randint(200, t_len, n_samples)
  repeats_sample = np.arange(30) #np.random.randint(0, 30, 15)

  for iretina in range(len(responses)):
    resp = responses[iretina]
    for irepeat in repeats_sample:
      print(iretina, irepeat)

      feed_dict={sr_graph.anchor_model.map_cell_grid_tf: resp['map_cell_grid'],
                 sr_graph.anchor_model.cell_types_tf: resp['ctype_1hot'],  # TODO(bhaishahster):Cell order verified!

                 sr_graph.anchor_model.mean_fr_tf: resp['mean_firing_rate'],
                 sr_graph.anchor_model.responses_tf:
                 np.expand_dims(resp['repeats'][irepeat, rand_sample, :], 2)}

      if hasattr(sr_graph.anchor_model, 'dist_nn'):
        dist_nn = np.array([resp['dist_nn_cell_type'][1],
                            resp['dist_nn_cell_type'][2]]).astype(np.float32)
        feed_dict.update({sr_graph.anchor_model.dist_nn: dist_nn,
                          sr_graph.neg_model.dist_nn: dist_nn})

      rr = sr_graph.sess.run(sr_graph.anchor_model.responses_embed,
                             feed_dict=feed_dict)
      resp_embed = np.append(resp_embed, rr.squeeze(), 0)
      retina_labels = np.append(retina_labels, iretina * np.ones(rr.shape[0]), 0)
      stimulus_labels = np.append(stimulus_labels, np.arange(rr.shape[0]), 0)
      repeat_labels = np.append(repeat_labels, irepeat * np.ones(rr.shape[0]), 0)


  stim_resp_embedding = {'rand_sample': rand_sample,
                         'repeats_sample': repeats_sample,
                         'resp_embed': resp_embed,
                         'retina_labels': retina_labels,
                         'stimulus_labels': stimulus_labels,
                         'repeat_labels': repeat_labels}



  return stim_resp_embedding

def resp_wn_repeats_multiple_retina_encoding(sr_graph, n_repeats=None):
  '''Embed responses and predict from embedding - 3 retinas seeing identical repeated WN.'''

  ## Embed responses across multiple retina.
  embed_x = sr_graph.stim_embed.shape[-3].value
  embed_y = sr_graph.stim_embed.shape[-2].value
  embed_t = sr_graph.stim_embed.shape[-1].value

  # Get repeats data - prepare it!
  import retina.response_model.python.metric_learning.end_to_end.data_util as data_util
  op = data_util.get_stimulus_response('/home/bhaishahster/end_to_end_wn_repeats/', 'bw-8-4', 'bw-8-4-reps',
                                       boundary=FLAGS.valid_cells_boundary)
  stimulus, responses, dimx, dimy, _ = op


  firing_rate_predictions_retina = {}
  response_embeddings = []
  cell_geometry_log = []
  if n_repeats is None:
    repeats_sample = np.arange(30)
  else:
    repeats_sample = np.random.randint(0, 30, n_repeats)

  for iretina in range(len(responses)):
    resp = responses[iretina]
    t_len = responses[iretina]['repeats'].shape[1]

    stimulus_times = np.arange(0, t_len)


    # Get cell locations
    cell_locations = []
    ct_log = []
    for icell in range(resp['map_cell_grid'].shape[-1]):
      cell_locations += [np.where(resp['map_cell_grid'][:, :, icell] > 0)]
      ict = np.squeeze(np.where(resp['ctype_1hot'][icell, :] == 1)[0])
      ct_log += [ict]
    cell_locations = np.array(cell_locations)[:, :, 0]
    ct_log = np.squeeze(np.array(ct_log))

    cell_geometry_log += [cell_locations, ct_log]

    firing_rate_pred_repeat = np.zeros((repeats_sample.shape[0], t_len, resp['repeats'].shape[-1]))
    response_embedding_repeat = np.zeros((repeats_sample.shape[0], t_len, embed_x, embed_y, embed_t))

    for iirepeat, irepeat in enumerate(repeats_sample):
      print(iretina, irepeat)

      resp_in = np.expand_dims(resp['repeats'][irepeat, stimulus_times, :], 2)
      batch_sz_embed = 100
      n_batches = np.array(t_len / batch_sz_embed).astype(np.int)
      permute_times = np.random.permutation(stimulus_times)

      for ibatch in range(n_batches):
        rand_sample = permute_times[ibatch * batch_sz_embed :
                                    (ibatch + 1) * batch_sz_embed]
        feed_dict={sr_graph.anchor_model.map_cell_grid_tf: resp['map_cell_grid'],
                   sr_graph.anchor_model.cell_types_tf: resp['ctype_1hot'],  # TODO(bhaishahster): Cell order verified!
                   sr_graph.anchor_model.mean_fr_tf: resp['mean_firing_rate'],
                   sr_graph.anchor_model.responses_tf:
                   np.expand_dims(resp['repeats'][irepeat, rand_sample, :], 2)}

        if hasattr(sr_graph.anchor_model, 'dist_nn'):
          dist_nn = np.array([resp['dist_nn_cell_type'][1],
                              resp['dist_nn_cell_type'][2]]).astype(np.float32)
          feed_dict.update({sr_graph.anchor_model.dist_nn: dist_nn,
                            sr_graph.neg_model.dist_nn: dist_nn})

        fr, r_embed = sr_graph.sess.run([sr_graph.resp_decode_from_resp,
                                         sr_graph.anchor_model.responses_embed],
                               feed_dict=feed_dict)

        response_embedding_repeat[iirepeat, rand_sample, :, :, :] = r_embed
        for icell in range(resp['repeats'].shape[-1]):
          ict = np.squeeze(np.where(resp['ctype_1hot'][icell, :] == 1)[0])
          firing_rate_pred_repeat[iirepeat, rand_sample, icell] = fr[:, cell_locations[icell, 0], cell_locations[icell, 1],  ict]

    firing_rate_predictions_retina.update({iretina: firing_rate_pred_repeat})
    response_embeddings += [response_embedding_repeat]

  resp_predictions = {'firing_rate_predictions': firing_rate_predictions_retina,
                      'recorded_data': responses}

  return resp_predictions, response_embeddings, cell_geometry_log


def resp_wn_repeats_interpolate_embeddings(sr_graph, response_embeddings,
                                           interpolation_retinas=[0, 1],
                                           alpha_list=[0, 0.2, 0.4, 0.6, 0.8, 1.0]):
  '''Take response embeddings across two retinas and interpolate between them.'''

  origin_embedding = response_embeddings[interpolation_retinas[0]]
  destination_embedding = response_embeddings[interpolation_retinas[1]]  # trials x t_len x embed_x x embed_y x embed_t
  n_trials = origin_embedding.shape[0]
  resp_predict_from_embedding = {}
  from IPython import embed; embed()
  for alpha in alpha_list:
    resp_predict = np.zeros((origin_embedding.shape[0], origin_embedding.shape[1], 80, 40, 2))

    for itrial in range(n_trials):
      print('alpha %.3f, itrial %d' % (alpha, itrial))
      embed_mixture = (1 - alpha) * origin_embedding[itrial, :] + alpha * destination_embedding[itrial, :]
      resp_predict[itrial, :] = sr_graph.sess.run(sr_graph.resp_decode_from_embedding, feed_dict={sr_graph.arbitrary_embedding: embed_mixture})

    resp_predict_from_embedding.update({alpha: resp_predict})

  resp_predict_from_embedding.update({'interpolation_retinas': interpolation_retinas})
  return resp_predict_from_embedding



## Subunit analysis - for how many subunits is the response closer?
def make_feed_dict(sr_model, retina, responses=None, stimulus=None):
  """Make feed_dict to evaluate ops later.

  Args :
      sr_graph : Tensorflow graph.
      retina : retina properties
      responses : Response vector (T x # cells)
      stimulus : T x X x Y x30

  """

  map_grid = retina['map_cell_grid']
  ctype_1hot = retina['ctype_1hot']
  mean_firing_rate = retina['mean_firing_rate']

  feed_dict = {sr_model.anchor_model.map_cell_grid_tf :map_grid ,
               sr_model.anchor_model.cell_types_tf: ctype_1hot,
               sr_model.anchor_model.mean_fr_tf: mean_firing_rate}

  if responses is not None:
    feed_dict.update({sr_model.anchor_model.responses_tf:
                      np.expand_dims(responses, 2)})

  if stimulus is not None:
    stimulus = sample_datasets.verify_stimulus_dimensions(stimulus,
                                                          dimx=80, dimy=40)
    feed_dict.update({sr_model.stim_tf: stimulus})

  if hasattr(sr_model.anchor_model, 'dist_nn'):
    dist_nn = np.array([retina['dist_nn_cell_type'][1],
                        retina['dist_nn_cell_type'][2]]).astype(np.float32)
    feed_dict.update({sr_model.anchor_model.dist_nn: dist_nn,
                      sr_model.neg_model.dist_nn: dist_nn})

  return feed_dict


def subunit_discriminability(dataset_dict, stimuli, responses,
                 sr_graph, num_examples=1000):
  ## compute distances between s-r pairs - pos and neg.
  ## negative_stim - if the negative is a stimulus or a response
  if num_examples % 100 != 0:
    raise ValueError('Only supports examples which are multiples of 100.')


  subunit_fit_loc = '/home/bhaishahster/stim-resp_collection_big_wn_retina_subunit_properties_train'
  subunits_datasets = gfile.ListDirectory(subunit_fit_loc)

  save_dict = {}

  datasets_log = {}
  for dat_key, datasets in dataset_dict.items():

    distances_log = {}
    distances_retina_sr_log = []
    distances_retina_rr_log = []
    for iretina in range(len(datasets)):
      # Find the relevant subunit fit
      piece = responses[iretina]['piece']
      matched_dataset = [ifit for ifit in subunits_datasets if piece[:12] == ifit[:12]]
      if matched_dataset == []:
        raise ValueError('Could not find subunit fit')

      subunit_fit_path = os.path.join(subunit_fit_loc, matched_dataset[0])

      # Get predicted spikes.
      dat_resp_su = pickle.load(gfile.Open(os.path.join(subunit_fit_path,
                                             'response_prediction.pkl'), 'r'))
      resp_su = dat_resp_su['resp_su']  # it has non-rejected cells as well.

      # Remove some cells.
      select_cells = [icell for icell in range(resp_su.shape[2])
                      if dat_resp_su['cell_ids'][icell] in
                      responses[iretina]['cellID_list'].squeeze()]

      select_cells = np.array(select_cells)
      resp_su = resp_su[:, :, select_cells].astype(np.float32)

      # Get stimulus
      stimulus = stimuli[responses[iretina]['stimulus_key']]
      stimulus_test = stimulus[FLAGS.test_min: FLAGS.test_max, :, :]
      responses_recorded_test = responses[iretina]['responses'][FLAGS.test_min: FLAGS.test_max, :]

      # Sample stimuli and responses.
      random_times = np.random.randint(40, stimulus_test.shape[0], num_examples)
      batch_size=100


      # Recorded response - predicted response distances.
      distances_retina = np.zeros((num_examples, 10)) + np.nan
      for Nsub in range(1, 11):
        for ibatch in range(np.floor(num_examples / batch_size).astype(np.int)):

          # construct stimulus tensor.
          stim_history = 30
          resp_pred_batch = np.zeros((batch_size, resp_su.shape[2]))
          resp_rec_batch = np.zeros((batch_size, resp_su.shape[2]))

          for isample in range(batch_size):
            itime = random_times[batch_size * ibatch + isample]
            resp_pred_batch[isample, :] = resp_su[Nsub - 1, itime, :]
            resp_rec_batch[isample, :] = responses_recorded_test[itime, :]

          # Embed predicted responses
          feed_dict = make_feed_dict(sr_graph, responses[iretina], responses=resp_pred_batch)
          embed_predicted = sr_graph.sess.run(sr_graph.anchor_model.responses_embed, feed_dict=feed_dict)

          # Embed recorded responses
          feed_dict = make_feed_dict(sr_graph, responses[iretina], responses=resp_rec_batch)
          embed_recorded = sr_graph.sess.run(sr_graph.anchor_model.responses_embed, feed_dict=feed_dict)

          dd = sr_graph.sess.run(sr_graph.distances_arbitrary, feed_dict={sr_graph.arbitrary_embedding_1: embed_predicted, sr_graph.arbitrary_embedding_2: embed_recorded})

          distances_retina[batch_size * ibatch: batch_size * (ibatch + 1), Nsub - 1] = dd
          print(iretina, Nsub, ibatch)

      distances_retina_rr_log += [distances_retina]


      # Stimulus - predicted response distances.
      distances_retina = np.zeros((num_examples, 10)) + np.nan
      for Nsub in range(1, 11):
        for ibatch in range(np.floor(num_examples / batch_size).astype(np.int)):

          # construct stimulus tensor.
          stim_history = 30
          stim_batch = np.zeros((batch_size, stimulus_test.shape[1],
                                 stimulus_test.shape[2], stim_history))
          resp_batch = np.zeros((batch_size, resp_su.shape[2]))

          for isample in range(batch_size):
            itime = random_times[batch_size * ibatch + isample]
            stim_batch[isample, :, :, :] = np.transpose(stimulus_test[itime: itime-stim_history:-1, :, :], [1, 2, 0])
            resp_batch[isample, :] = resp_su[Nsub - 1, itime, :]

          feed_dict = make_feed_dict(sr_graph, responses[iretina], resp_batch, stim_batch)

          # Get distances
          d_pos = sr_graph.sess.run(sr_graph.d_s_r_pos, feed_dict=feed_dict)

          distances_retina[batch_size * ibatch: batch_size * (ibatch + 1), Nsub - 1] = d_pos
          print(iretina, Nsub, ibatch)

      distances_retina_sr_log += [distances_retina]


    distances_log.update({'rr': distances_retina_rr_log})
    distances_log.update({'sr': distances_retina_sr_log})
    datasets_log.update({dat_key: distances_log})
  save_dict.update({'datasets_log': datasets_log,
                    'dataset_dict': dataset_dict})

  return save_dict

def subunit_discriminability_null(dataset_dict, stimuli, responses,
                 sr_graph, num_examples=1000):
  ## compute distances between s-r pairs - pos and neg.
  ## negative_stim - if the negative is a stimulus or a response
  if num_examples % 100 != 0:
    raise ValueError('Only supports examples which are multiples of 100.')


  subunit_fit_loc = '/home/bhaishahster/stim-resp_collection_big_wn_retina_subunit_properties_train'
  subunits_datasets = gfile.ListDirectory(subunit_fit_loc)

  save_dict = {}

  datasets_log = {}
  for dat_key, datasets in dataset_dict.items():

    distances_log = {}
    distances_retina_sr_log = []
    distances_retina_rr_log = []
    for iretina in range(len(datasets)):
      # Find the relevant subunit fit
      piece = responses[iretina]['piece']
      matched_dataset = [ifit for ifit in subunits_datasets if piece[:12] == ifit[:12]]
      if matched_dataset == []:
        raise ValueError('Could not find subunit fit')

      subunit_fit_path = os.path.join(subunit_fit_loc, matched_dataset[0])

      # Get predicted spikes.
      dat_resp_su = pickle.load(gfile.Open(os.path.join(subunit_fit_path,
                                             'response_prediction_null.pkl'), 'r'))
      resp_su = dat_resp_su['resp_su']  # it has non-rejected cells as well.

      # Remove some cells.
      select_cells = [icell for icell in range(resp_su.shape[2])
                      if dat_resp_su['cell_ids'][icell] in
                      responses[iretina]['cellID_list'].squeeze()]

      select_cells = np.array(select_cells)
      resp_su = resp_su[:, :, select_cells].astype(np.float32)

      # Get stimulus
      stimulus_test = dat_resp_su['stimulus_null']

      # Sample stimuli and responses.
      random_times = np.random.randint(40, stimulus_test.shape[0], num_examples)
      batch_size=100

      # Stimulus - predicted response distances.
      distances_retina = np.zeros((num_examples, 10)) + np.nan
      for Nsub in range(1, 11):
        for ibatch in range(np.floor(num_examples / batch_size).astype(np.int)):

          # construct stimulus tensor.
          stim_history = 30
          stim_batch = np.zeros((batch_size, stimulus_test.shape[1],
                                 stimulus_test.shape[2], stim_history))
          resp_batch = np.zeros((batch_size, resp_su.shape[2]))

          for isample in range(batch_size):
            itime = random_times[batch_size * ibatch + isample]
            stim_batch[isample, :, :, :] = np.transpose(stimulus_test[itime: itime-stim_history:-1, :, :], [1, 2, 0])
            resp_batch[isample, :] = resp_su[Nsub - 1, itime, :]

          feed_dict = make_feed_dict(sr_graph, responses[iretina], resp_batch, stim_batch)

          # Get distances
          d_pos = sr_graph.sess.run(sr_graph.d_s_r_pos, feed_dict=feed_dict)

          distances_retina[batch_size * ibatch: batch_size * (ibatch + 1), Nsub - 1] = d_pos
          print(iretina, Nsub, ibatch)

      distances_retina_sr_log += [distances_retina]

    distances_log.update({'sr': distances_retina_sr_log})
    datasets_log.update({dat_key: distances_log})
  save_dict.update({'datasets_log': datasets_log,
                    'dataset_dict': dataset_dict})

  return save_dict



