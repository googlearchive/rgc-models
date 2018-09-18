# Copyright 2018 Google LLC

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
r"""Run tests on joint embedding models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os.path
import tensorflow as tf
import tensorflow.contrib.slim as slim
from absl import app
from absl import gfile
import numpy as np, h5py,numpy
import scipy.io as sio
from numpy.random import RandomState
import pickle
import retina.response_model.python.metric_learning.end_to_end.sample_datasets as sample_datasets
import retina.response_model.python.metric_learning.end_to_end.encode as encode
import retina.response_model.python.metric_learning.end_to_end.decode as decode
import retina.response_model.python.metric_learning.end_to_end.data_util as data_util
import retina.response_model.python.metric_learning.end_to_end.bookkeeping as bookkeeping
import retina.response_model.python.metric_learning.end_to_end.analysis_utils as analysis_utils
import retina.response_model.python.metric_learning.end_to_end.testing_experiments as experiments

FLAGS = tf.app.flags.FLAGS

def test_metric(training_datasets, testing_datasets, responses, stimuli,
            sr_graph, sess, file_name):
  print('Testing for metric learning')
  tf.logging.info('Testing for metric learning')

  # saving filename.
  save_analysis_filename = os.path.join(FLAGS.save_folder,
                                        file_name + '_analysis_sample_resps')

  save_dict = {}
  #if FLAGS.taskid in [2, 8, 14, 20]:
  #  return

  retina_ids = [resp['piece'] for resp in responses]

  '''
  # NULL + SUBUNITS: Test subunits on null stimulus
  # Responses with different number of subunits - which is closer to stimulus?
  # Take embedding of population responses to same stimulus but
  # different number of subunits and see which is closer to the stimulus.
  rand_number = np.random.rand(2)
  dataset_dict = {'test_sr': testing_datasets, 'train_sr': training_datasets}
  su_analysis = experiments.subunit_discriminability_null(dataset_dict,
                                                          stimuli, responses,
                                                          sr_graph,
                                                          num_examples=10000)
  su_analysis.update({'rand_key': rand_number})
  su_analysis.update({'retina_ids': retina_ids})
  pickle.dump(su_analysis, gfile.Open(save_analysis_filename + '_test14_su_analysis_null.pkl', 'w'))

  

  # SUBUNITS: Responses with different number of subunits - which is closer to stimulus?
  # Take embedding of population responses to same stimulus but
  # different number of subunits and see which is closer to the stimulus.
  rand_number = np.random.rand(2)
  dataset_dict = {'test_sr': testing_datasets, 'train_sr': training_datasets}
  su_analysis = experiments.subunit_discriminability(dataset_dict,
                                          stimuli, responses, sr_graph,
                                          num_examples=10000)
  su_analysis.update({'rand_key': rand_number})
  su_analysis.update({'retina_ids': retina_ids})
  pickle.dump(su_analysis, gfile.Open(save_analysis_filename + '_test12_su_analysis.pkl', 'w'))

  '''
  # Embed stimuli-response + decode
  # WN
  for stim_key in stimuli.keys():
    expt = experiments.stimulus_response_embedding_expt
    stim_resp_embedding = expt(stimuli, responses, sr_graph,
                               stim_key=stim_key, n_samples=1000)
    save_dict.update({'stim_resp_embedding_wn': stim_resp_embedding})
    save_dict.update({'retina_ids': retina_ids})
    pickle.dump(stim_resp_embedding, gfile.Open((save_analysis_filename + '_test3_%s.pkl') %  stim_key, 'w'))

  '''
  # stimulus + response embed
  for stim_key in stimuli.keys():
    expt = experiments.stimulus_response_embedding_expt
    stim_resp_embedding = expt(stimuli, responses, sr_graph,
                               stim_key=stim_key, n_samples=1000, if_continuous=True)
    save_dict.update({'stim_resp_embedding_wn': stim_resp_embedding})
    save_dict.update({'retina_ids': retina_ids})
    pickle.dump(stim_resp_embedding, gfile.Open((save_analysis_filename + '_test3_continuous_%s.pkl') %  stim_key, 'w'))
  '''

  # ROC analysis on training and testing datasets
  rand_number = np.random.rand(2)
  dataset_dict = {'test_sr': testing_datasets, 'train_sr': training_datasets}
  roc_analysis = experiments.roc_analysis(dataset_dict,
                                          stimuli, responses, sr_graph,
                                          num_examples=10000)
  roc_analysis.update({'rand_key': rand_number})
  roc_analysis.update({'retina_ids': retina_ids})
  pickle.dump(roc_analysis, gfile.Open(save_analysis_filename + '_test4.pkl', 'w'))


  # ROC analysis with subsampling of cells
  # dataset_dict = {'train_sr': training_datasets, 'test_sr': testing_datasets}
  dataset_dict = {'test_sr': testing_datasets}
  if len(training_datasets) ==1 :
    dataset_dict.update({'train_sr': training_datasets})

  roc_frac_cells_dict = {}
  for frac_cells in [0.05, 0.1, 0.15,  0.2, 0.5, 0.8, 1.0]:
    print('frac_cells : %.3f' % frac_cells)
    roc_analysis = experiments.roc_analysis(dataset_dict,
                                            stimuli, responses, sr_graph,
                                            num_examples=10000,
                                            frac_cells=frac_cells)
    roc_frac_cells_dict.update({frac_cells: roc_analysis})
  roc_frac_cells_dict.update({'retina_ids': retina_ids})
  pickle.dump(roc_frac_cells_dict, gfile.Open(save_analysis_filename + '_test4_frac_cells_new.pkl', 'w'))
  print(save_analysis_filename + '_test4_frac_cells_new.pkl')

  # ROC analysis - RSS triplets - on training and testing datasets
  rand_number = np.random.rand(2)
  dataset_dict = {'test_sr': testing_datasets, 'train_sr': training_datasets}
  roc_analysis = experiments.roc_analysis(dataset_dict,
                                          stimuli, responses, sr_graph,
                                          num_examples=10000, negative_stim=True)
  roc_analysis.update({'rand_key': rand_number})
  roc_analysis.update({'retina_ids': retina_ids})
  pickle.dump(roc_analysis, gfile.Open(save_analysis_filename + '_test4_rss.pkl', 'w'))


  ## Response transformations
  expt = experiments.response_transformation_increase_nl
  nl_expt = expt(stimuli, responses, sr_graph,
                 time_start_list=[100, 4000, 10000], time_len=100,
                 alpha_list=[1.5, 1.25, 0.8, 0.6])
  nl_expt.update({'retina_ids': retina_ids})
  pickle.dump(nl_expt, gfile.Open(save_analysis_filename + '_test7_resp_transform_wn.pkl', 'w'))


  ## Embed all stimuli.
  '''
  stim_embedding_dict = experiments.stimulus_embedding_expt(stimuli, sr_graph,
                                                n_stims_per_type=1000)

  save_dict.update({'stimulus_embedding': stim_embedding_dict})

  pickle.dump(stim_embedding_dict, gfile.Open(save_analysis_filename + '_test1.pkl', 'w'))
  print('Saved after test 1')
  '''

  ## Explore invariances of the stimulus embedding.
  # Change luminance, contrast, geometrical changes (translate, rotate) and see
  # how they are reflected in embedded space.
  #
  # Get stimuli which are transformed
  stim_transformations_dict = experiments.stimulus_transformations_expt(stimuli,
                                                                        sr_graph,
                                                                        n_stims_per_type_transform=50,
                                                                        n_stims_per_type_bkg=500) # 2000
  save_dict.update({'stimulus_transformations': stim_transformations_dict})

  pickle.dump(stim_transformations_dict, gfile.Open(save_analysis_filename + '_test2.pkl', 'w'))
  print('Saved after test 2')


  # NSEM
  '''
  stim_resp_embedding = expt(stimuli, responses, sr_graph,
                             stim_key='stim_2', n_samples=100)
  save_dict.update({'stim_resp_embedding_nsem': stim_resp_embedding})
  save_dict.update({'retina_ids': retina_ids})
  pickle.dump(stim_resp_embedding, gfile.Open(save_analysis_filename + '_test3_nsem.pkl', 'w'))
  print('Saved after test 3')
  '''


  # Drop different cell types
  expt = experiments.resp_drop_cells_expt
  for stim_key in stimuli.keys():
    resp_drop_cells = expt(stimuli, responses, sr_graph,
                           stim_key=stim_key, n_samples=100)
    save_dict.update({'resp_drop_cells': resp_drop_cells})
    save_dict.update({'retina_ids': retina_ids})
    pickle.dump(resp_drop_cells, gfile.Open(save_analysis_filename + '_test8_%s.pkl' % stim_key, 'w'))


  ## TODO(bhaishahster): joint-auto-embed model response prediction for some stimuli
  # Rasters across different retinas embedded in same space.
  expt = experiments.resp_wn_repeats_multiple_retina
  resp_repeats = expt(sr_graph, n_samples=100)
  pickle.dump(resp_repeats, gfile.Open(save_analysis_filename + '_test9_repeats.pkl', 'w'))

  # Predict responses by embedding from rasters of different retina.
  expt = experiments.resp_wn_repeats_multiple_retina_encoding
  resp_repeats, response_embeddings, cell_geometry_log = expt(sr_graph, n_repeats=10)  # n_repeats=None for using all repeats
  pickle.dump(resp_repeats, gfile.Open(save_analysis_filename + '_test10_repeats_prediction.pkl', 'w'))

  # Interpolate between retinas
  expt = experiments.resp_wn_repeats_interpolate_embeddings
  interpolate_dict = expt(sr_graph, [ri[:7, 600:, :, :, :] for ri in response_embeddings])
  pickle.dump([interpolate_dict, cell_geometry_log], gfile.Open(save_analysis_filename + '_test11_repeats_pred_interpolation.pkl', 'w'))

  # Interpolate between retinas - mean of embedding across repeats
  expt = experiments.resp_wn_repeats_interpolate_embeddings
  interpolate_dict = expt(sr_graph, [np.expand_dims(ri[:, 600:, :, :, :].mean(0), 0) for ri in response_embeddings])
  pickle.dump([interpolate_dict, cell_geometry_log], gfile.Open(save_analysis_filename + '_test11_repeats_pred_interpolation_mean.pkl', 'w'))

  return

  '''
  # Load multiple check points and analyse accuracy
  # /retina/response_model/python/metric_learning/end_to_end/stimulus_response_embedding --logtostderr --mode=1 --taskid=2 --save_suffix='_stim-resp_wn_nsem' --stim_layers='1, 5, 1, 3, 64, 1, 3, 64, 1, 3, 64, 1, 3, 64, 2, 3, 64, 2, 3, 1, 1' --resp_layers='3, 64, 1, 3, 64, 1, 3, 64, 1, 3, 64, 2, 3, 64, 2, 3, 1, 1' --batch_norm=True --save_folder='//home/bhaishahster/end_to_end_feb_5_6PM' --learning_rate=0.001 --batch_train_sz=100 --batch_neg_train_sz=100 --sr_model='convolutional_embedding'
  for frac_cells in [0.2, 1.0, 0.5]:
    dataset_dict = {'train_sr': training_datasets}
    saver = tf.train.Saver()

    filename = bookkeeping.get_filename(training_datasets, testing_datasets,
                                        FLAGS.beta, FLAGS.sr_model)
    long_filename = os.path.join(FLAGS.save_folder, filename)
    checkpoints = gfile.Glob(long_filename + '*.meta')
    checkpoints = [cpts[:-5] for cpts in checkpoints]

    roc_dict = {}
    for cpts in checkpoints:
      try:
        print(cpts)
        saver.restore(sess, cpts)
        iteration = int(cpts.split('/')[-1].split('-')[-1])

        roc_analysis = experiments.roc_analysis(dataset_dict,
                                                stimuli, responses, sr_graph,
                                                num_examples=10000,
                                                frac_cells=frac_cells)
        roc_dict.update({iteration: roc_analysis})
      except:
        pass
    pickle.dump(roc_dict, gfile.Open((save_analysis_filename +
                                      '_test6_frac_cells_%.2f.pkl') % frac_cells, 'w'))




  # ROC analysis with negatives generated at small difference to positives.
  dataset_dict = {'train_sr': training_datasets}
  roc_delta_t_dict = {}
  for delta_t in [1, 2, 3, 4, 5]:
    print('Delta t : %d' % delta_t)
    roc_analysis = experiments.roc_analysis(dataset_dict,
                                            stimuli, responses, sr_graph,
                                            num_examples=10000,
                                            delta_t=delta_t)
    roc_delta_t_dict.update({delta_t: roc_analysis})
  pickle.dump(roc_delta_t_dict, gfile.Open(save_analysis_filename + '_test4_delta_t.pkl', 'w'))

  '''

  ## ROC curves of responses from repeats - dataset 1
  '''
  repeats_datafile = '/home/bhaishahster/metric_learning/datasets/2015-09-23-7.mat'
  repeats_data = sio.loadmat(gfile.Open(repeats_datafile, 'r'));
  repeats_data['cell_type'] = repeats_data['cell_type'].T
  # process repeats data
  num_cell_types = 2
  dimx = 80
  dimy = 40
  data_util.process_dataset(repeats_data, dimx, dimy, num_cell_types)
  # analyse and store the result
  test_reps = analyse_response_repeats(repeats_data,
                                       sr_graph.anchor_model,
                                       sr_graph.neg_model, sr_graph.sess)
  save_dict.update({'test_reps_2015-09-23-7': test_reps})
  pickle.dump(save_dict, gfile.Open(save_analysis_filename, 'w'))
  '''

  ## ROC curves of responses from repeats - dataset 2
  '''
  repeats_datafile = '/home/bhaishahster/metric_learning/examples_pc2005_08_03_0/data005_test.mat'
  repeats_data = sio.loadmat(gfile.Open(repeats_datafile, 'r'));
  data_util.process_dataset(repeats_data, dimx, dimy, num_cell_types)

  # analyse and store the result
  test_clustering = analyse_response_repeats_all_trials(repeats_data,
                                                        sr_graph.anchor_model,
                                                        sr_graph.neg_model,
                                                        sr_graph.sess)
  save_dict.update({'test_reps_2005_08_03_0': test_clustering})
  pickle.dump(save_dict, gfile.Open(save_analysis_filename, 'w'))
  '''
  #
  # get model params

  model_pars_dict = {'model_pars': sr_graph.sess.run(tf.trainable_variables())}
  pickle.dump(model_pars_dict, gfile.Open(save_analysis_filename + '_test5.pkl', 'w'))

  print(save_analysis_filename)


def test_encoding(training_datasets, testing_datasets, responses, stimuli,
                  sr_graph, sess, file_name, sample_fcn):

  print('Testing for encoding model')
  tf.logging.info('Testing for encoding model')
  #from IPython import embed; embed()

  # saving filename.
  save_analysis_filename = os.path.join(FLAGS.save_folder,
                                        file_name + '_analysis_sample_resps')

  retina_ids = [resp['piece'] for resp in responses]

  # 4) Find `retina_embed` for a test retina
  # Find mean of embedding of training retinas

  latent_dimensionality = np.int(FLAGS.resp_layers.split(',')[-2])
  batch_sz = 500
  ret_params_list = []
  for idataset in training_datasets:
    if idataset >= len(responses):
      continue
    print(idataset)
    rng = numpy.random.RandomState(23)
    feed_dict = sample_fcn.batch(stimuli, responses,
                                 idataset, sr_graph,
                                 batch_pos_sz=batch_sz,
                                 batch_neg_sz=batch_sz,
                                 batch_type='test',
                                 if_continuous=False, rng=rng)
    op  = sess.run(sr_graph.retina_params,
                   feed_dict=feed_dict)
    ret_params_list += [op]
  ret_params_list = np.array(ret_params_list)
  ret_params_init = np.mean(ret_params_list, 0)  # Use for initialization

  # Now optimize ret_params for each retina.
  loss_arbit_ret_params = sr_graph.loss_arbit_ret_params
  ret_params_grad = tf.gradients(loss_arbit_ret_params, sr_graph.retina_params_arbitrary)
  ret_params_dict = {}
  for idataset in testing_datasets:
    rng = numpy.random.RandomState(23)
    ret_params_new = np.copy(ret_params_init)
    lr = 0.001
    ret_log = []
    for iiter in range(100):
      feed_dict = sample_fcn.batch(stimuli, responses,
                                   idataset, sr_graph,
                                   batch_pos_sz=500, batch_neg_sz=0,
                                   batch_type='train',
                                   if_continuous=True, rng=rng)
      feed_dict.update({sr_graph.retina_params_arbitrary: ret_params_new})
      delta_ret_param, l_np = sess.run([ret_params_grad, loss_arbit_ret_params],
                                 feed_dict=feed_dict)
      print('Retina: %d, step: %d, loss: %.3f, Ret_params: %s' % (idataset, iiter, l_np, ret_params_new))
      ret_log += [np.copy(ret_params_new)]
      ret_params_new -= lr * delta_ret_param[0]

    dataset_log = {'final_embedding': np.copy(ret_params_new), 'path': np.copy(ret_log)}
    ret_params_dict.update({idataset: dataset_log})

  pickle.dump(ret_params_dict, gfile.Open((save_analysis_filename + '_test6.pkl'), 'w'))

  # Latent embedding of different retinas.
  # 3) Interpolate between latent representation and see how responses change.
  interpolation_retinas_log = [[63, 5], [73, 65], [41, 83], [38, 57], [72, 76], [48, 58], [2, 5]] # [[72, 76], [48, 58], [2, 5]]
  batch_sz = 500
  save_dict_log = []
  for interpolation_retinas in interpolation_retinas_log:
    retina_params_end_pts = []
    cell_info = []
    valid_cell_log = []

    # 3a) Find latent representation of each retina.
    for idataset in interpolation_retinas:
      rng = numpy.random.RandomState(23)
      feed_dict = sample_fcn.batch(stimuli, responses,
                                        idataset, sr_graph,
                                        batch_pos_sz=batch_sz, batch_neg_sz=batch_sz,
                                        batch_type='test',
                                        if_continuous=False, rng=rng)
      op  = sess.run([sr_graph.retina_params, sr_graph.stim_tf,
                      sr_graph.anchor_model.embed_locations_original,
                      sr_graph.anchor_model.map_cell_grid_tf],
                     feed_dict=feed_dict)
      ret_params_np, stim_np, cell_locs, map_cell_grid = op
      retina_params_end_pts += [ret_params_np]
      rcct_log = []
      for icell in range(map_cell_grid.shape[2]):
        r, c = np.where(map_cell_grid[:, :, icell] > 0)
        ct = np.where(np.squeeze(cell_locs[r, c, :]) > 0)[0]
        rcct_log += [[r[0], c[0], ct[0]]]
      cell_info += [np.array(rcct_log)]
      valid_cell_log += [responses[idataset]['valid_cells']]

    # 3b) Now, interpolate and check the responses.
    fr_interpolate_log = []
    alpha_log = np.arange(0, 1.1, 0.1)
    for alpha in alpha_log:
      retina_params_interpolate = (alpha * retina_params_end_pts[0] +
                                   (1 - alpha) * retina_params_end_pts[1])
      feed_dict = {sr_graph.stim_tf: stim_np,
                   sr_graph.retina_params_arbitrary: retina_params_interpolate}
      fr_interpolate = sess.run(sr_graph.response_pred_from_arbit_ret_params,
                                feed_dict=feed_dict)
      fr_interpolate_log += [fr_interpolate]

    fr_interpolate_log = np.array(fr_interpolate_log)
    save_dict = {'interpolation_retinas': interpolation_retinas,
                 'alpha_log': alpha_log, 'fr_interpolate_log': fr_interpolate_log,
                 'cell_info': cell_info, 'valid_cell_log': valid_cell_log,
                 'retina_params_end_pts': retina_params_end_pts}
    save_dict_log += [save_dict]

  pickle.dump(save_dict_log, gfile.Open((save_analysis_filename + '_test5.pkl'), 'w'))


  # 1) Predict responses of different retinas - training AND testing retinas.
  #from IPython import embed; embed()

  tag_list =['training_datasets', 'testing_datasets']
  results_tr_tst = {}
  for itr_tst, tr_test_datasets in enumerate([training_datasets, testing_datasets]):
    results_log = {}
    for idataset in tr_test_datasets:

      if idataset >= len(responses):
        continue
      print(idataset)
      rng = numpy.random.RandomState(23)
      feed_dict = sample_fcn.batch(stimuli, responses,
                                        idataset, sr_graph,
                                        batch_pos_sz=200, batch_neg_sz=200,
                                        batch_type='test',
                                        if_continuous=True, rng=rng)

      op  = sess.run([sr_graph.fr_predicted,
                      sr_graph.anchor_model.embed_locations_original,
                      sr_graph.anchor_model.map_cell_grid_tf,
                      sr_graph.anchor_model.responses_tf, sr_graph.retina_params],
                     feed_dict=feed_dict)
      fr_pred_np, cell_locs, map_cell_grid, responses_np, ret_params_np = op

      # r, c, z = np.where(cell_locs > 0)
      fr_pred_cell = np.squeeze(np.zeros_like(responses_np))
      for icell in range(responses_np.shape[1]):
        r, c = np.where(map_cell_grid[:, :, icell] > 0)
        ct = np.where(np.squeeze(cell_locs[r, c, :]) > 0)[0]
        r = r[0]
        c = c[0]
        ct = ct[0]
        fr_pred_cell[:, icell] = fr_pred_np[:, r, c, ct]

        '''
        tms = np.arange(responses_np.shape[0])
        plt.stem(tms, responses_np[:, icell, 0])
        plt.plot(3 * fr_pred_np[:, r, c, 0])
        plt.plot(3 * fr_pred_np[:, r, c, 1])
        '''

      # Find the stimulus
      stim_np = feed_dict[sr_graph.stim_tf]
      t_len, dx, dy, t_depth = stim_np.shape

      stim_np_compressed = np.zeros((t_len + t_depth - 1, dx, dy))  # 500, 80, 40, 30
      stim_np_compressed[:t_depth] = np.transpose(stim_np[0, :, :, :], [2, 0, 1])
      for itm in np.arange(1, t_len):
        stim_np_compressed[itm + t_depth - 1, :, :] = stim_np[itm, :, :, 0]

      save_dict = {'fr_pred_cell': fr_pred_cell,
                   'responses_recorded': np.squeeze(responses_np),
                   'valid_cells': responses[idataset]['valid_cells'],
                   'ctype_1hot' : responses[idataset]['ctype_1hot'],
                   'cell_locs': cell_locs, 'map_cell_grid': map_cell_grid,
                   'ret_params_np': ret_params_np, 'fr_pred_np': fr_pred_np,
                   'stimulus_key': responses[idataset]['stimulus_key']}
      results_tr_tst.update({responses[idataset]['stimulus_key']:
                                 stim_np_compressed})

      results_log.update({idataset: save_dict})

    results_tr_tst.update({tag_list[itr_tst]: results_log})
    results_tr_tst.update({'retina_ids': retina_ids})
  pickle.dump(results_tr_tst, gfile.Open((save_analysis_filename + '_test3.pkl'), 'w'))


  # 2) Is the latent representation consistent for each retina, across responses?
  latent_dimensionality = np.int(FLAGS.resp_layers.split(',')[-2])
  batch_sz_list = [500]
  n_repeats = 1
  tag_list =['training_datasets', 'testing_datasets']
  results_tr_tst = {}
  for itr_tst, tr_test_datasets in enumerate([training_datasets, testing_datasets]):
    results_log = {}
    for idataset in tr_test_datasets:
      if idataset >= len(responses):
        continue
      print(idataset)
      rng = numpy.random.RandomState(23)
      ret_params_np = np.zeros((len(batch_sz_list), n_repeats, latent_dimensionality))
      for ibatch_sz, batch_sz in enumerate(batch_sz_list):
        for iresp in range(n_repeats):
          print(idataset, ibatch_sz, iresp)
          feed_dict = sample_fcn.batch(stimuli, responses,
                                            idataset, sr_graph,
                                            batch_pos_sz=batch_sz,
                                            batch_neg_sz=batch_sz,
                                            batch_type='test',
                                            if_continuous=False, rng=rng)
          op  = sess.run(sr_graph.retina_params,
                         feed_dict=feed_dict)
          print(op)
          ret_params_np[ibatch_sz, iresp, :] = op

      save_dict = {'ret_params_np': ret_params_np,
                   'batch_sz_list': batch_sz_list,
                   'valid_cells': responses[idataset]['valid_cells']}
      results_log.update({idataset: save_dict})

    results_tr_tst.update({tag_list[itr_tst]: results_log})
    results_tr_tst.update({'retina_ids': retina_ids})
  pickle.dump(results_tr_tst, gfile.Open((save_analysis_filename + '_test4.pkl') , 'w'))

  # 3) Embedding of EIs
  if hasattr(sr_graph, 'retina_params_from_ei'):
    latent_dimensionality = np.int(FLAGS.resp_layers.split(',')[-2])
    batch_sz = 500
    tag_list =['training_datasets', 'testing_datasets']
    results_tr_tst = {}
    for itr_tst, tr_test_datasets in enumerate([training_datasets, testing_datasets]):
      results_log = {}
      for idataset in tr_test_datasets:
        if idataset >= len(responses):
          continue
        print(idataset)
        rng = numpy.random.RandomState(23)
        feed_dict = sample_fcn.batch(stimuli, responses,
                                     idataset, sr_graph,
                                     batch_pos_sz=batch_sz,
                                     batch_neg_sz=batch_sz,
                                     batch_type='test',
                                     if_continuous=False, rng=rng)
        op = sess.run([sr_graph.retina_params_from_ei, sr_graph.retina_params],
                       feed_dict=feed_dict)
        ret_params_from_ei_np, ret_params_np  = op
        print(op)

        save_dict = {'ret_params_np': ret_params_np,
                     'ret_params_from_ei_np': ret_params_from_ei_np,
                     'valid_cells': responses[idataset]['valid_cells']}
        results_log.update({idataset: save_dict})

      results_tr_tst.update({tag_list[itr_tst]: results_log})
      results_tr_tst.update({'retina_ids': retina_ids})
    pickle.dump(results_tr_tst, gfile.Open((save_analysis_filename + '_test4_ei.pkl') , 'w'))

  # 4) TODO(bhaishahster): Estimate LN model based on stimulus and observed responses.


def batch_few_cells(sr_graph, responses, stimulus,
                    batch_train_sz, batch_neg_train_sz):
  batch_train = sample_datasets.get_batch(stimulus,
                                          responses['responses'],
                                          batch_size=batch_train_sz,
                                          batch_neg_resp=batch_neg_train_sz,
                                          stim_history=30, min_window=10)
  stim_batch, resp_batch, resp_batch_neg = batch_train
  feed_dict = {sr_graph.stim_tf: stim_batch,
               sr_graph.anchor_model.responses_tf: np.expand_dims(resp_batch, 2),
               sr_graph.neg_model.responses_tf: np.expand_dims(resp_batch_neg, 2),

               sr_graph.anchor_model.map_cell_grid_tf: responses['map_cell_grid'],
               sr_graph.anchor_model.cell_types_tf: responses['ctype_1hot'],
               sr_graph.anchor_model.mean_fr_tf: responses['mean_firing_rate'],

               sr_graph.neg_model.map_cell_grid_tf: responses['map_cell_grid'],
               sr_graph.neg_model.cell_types_tf: responses['ctype_1hot'],
               sr_graph.neg_model.mean_fr_tf: responses['mean_firing_rate'],
               }

  return feed_dict





def analyse_response_repeats(repeats_data, anchor_model, neg_model, sess):
  # generate positive examples of resposnes - responses at same time, different repeats

  def get_feed_dict(responses1, responses2, anchor_model, neg_model, repeats_data):
    feed_dict = {anchor_model.responses_tf: np.expand_dims(responses1, 2),
                 neg_model.responses_tf: np.expand_dims(responses2, 2),

                 anchor_model.map_cell_grid_tf: repeats_data['map_cell_grid'],
                 anchor_model.cell_types_tf: repeats_data['ctype_1hot'],
                 anchor_model.mean_fr_tf: repeats_data['mean_firing_rate'],

                 neg_model.map_cell_grid_tf: repeats_data['map_cell_grid'],
                 neg_model.cell_types_tf: repeats_data['ctype_1hot'],
                 neg_model.mean_fr_tf: repeats_data['mean_firing_rate']}
    return feed_dict

  dist_ap_log = np.array([])
  dist_an_log = np.array([])
  for iibatch in range(100):
    print(iibatch)
    pos_batch = 100
    pos_times = np.random.randint(0, repeats_data['repeats'].shape[1], pos_batch)

    pos_trials = np.zeros((pos_batch, 2))
    pos_trials[:, 0] = np.random.randint(0, repeats_data['repeats'].shape[0], pos_batch)
    for isample in range(pos_batch):
      irep = np.random.randint(0, repeats_data['repeats'].shape[0], 1)
      while pos_trials[isample, 0] == irep:
        irep = np.random.randint(0, repeats_data['repeats'].shape[0], 1)
      pos_trials[isample, 1] = irep

    neg_times = np.random.randint(0, repeats_data['repeats'].shape[1], pos_batch)
    neg_trials = np.random.randint(0, repeats_data['repeats'].shape[0], pos_batch)

    # anchor, pos
    responses1 = repeats_data['repeats'][pos_trials[:, 0].astype(np.int), pos_times, :]
    responses2 = repeats_data['repeats'][pos_trials[:, 1].astype(np.int), pos_times, :]
    resp_anch, resp_pos = sess.run([anchor_model.responses_embed,
                               neg_model.responses_embed],
                              feed_dict=get_feed_dict(responses1, responses2,
                                                      anchor_model, neg_model,
                                                      repeats_data))
    dist_ap = np.sum((resp_anch - resp_pos)**2, (1, 2, 3))

    # anchor, neg
    responses1 = repeats_data['repeats'][pos_trials[:, 0].astype(np.int), pos_times, :]
    responses2 = repeats_data['repeats'][neg_trials.astype(np.int), neg_times, :]
    resp_anch2, resp_neg = sess.run([anchor_model.responses_embed,
                               neg_model.responses_embed],
                              feed_dict=get_feed_dict(responses1, responses2,
                                                      anchor_model, neg_model,
                                                      repeats_data))
    dist_an = np.sum((resp_anch - resp_neg)**2, (1, 2, 3))

    dist_ap_log = np.append(dist_ap_log, dist_ap)
    dist_an_log = np.append(dist_an_log, dist_an)

  precision_log, recall_log, F1_log, FPR_log, TPR_log = ROC(dist_ap_log, dist_an_log)

  print(np.sum(dist_ap_log < dist_an_log))
  print(np.sum(dist_ap_log > dist_an_log))
  print(np.sum(dist_ap_log == dist_an_log))
  test_reps = {'precision': precision_log, 'recall': recall_log,
               'F1': F1_log, 'FPR': FPR_log, 'TPR': TPR_log,
               'd_pos_log': dist_ap_log, 'd_neg_log': dist_an_log}
  return test_reps


def analyse_response_repeats_all_trials(repeats_data, anchor_model, neg_model, sess):
  # generate positive examples of resposnes - responses at same time, different repeats

  prng = RandomState(50)

  n_trials = repeats_data['repeats'].shape[0]
  n_random_times = 10
  random_times = prng.randint(0, repeats_data['repeats'].shape[1], n_random_times)
  responses = repeats_data['repeats'][:, random_times, :].astype(np.float32)
  responses = np.transpose(responses, [1, 0, 2])
  responses = np.reshape(responses, [n_trials * n_random_times, responses.shape[2]]).astype(np.float32)
  stim_idx = np.repeat(np.arange(n_random_times), n_trials, 0)

  # embed a sample response to get dimensions
  feed_dict = {anchor_model.map_cell_grid_tf: repeats_data['map_cell_grid'],
               anchor_model.cell_types_tf: repeats_data['ctype_1hot'],
               anchor_model.mean_fr_tf: repeats_data['mean_firing_rate'],
               anchor_model.responses_tf: np.expand_dims(responses[0:100, :], 2)}
  resp_test = sess.run(anchor_model.responses_embed, feed_dict=feed_dict)
  resp_embed = np.zeros((responses.shape[0],
                         resp_test.shape[1],
                         resp_test.shape[2], 1))

  # embed the responses
  # since we use batch norm in testing, we need to jumble the response to get correct estimate of batch norm statistics
  tms = np.arange(responses.shape[0])
  tms_jumble = np.random.permutation(tms)

  batch_sz = 100
  for itm in np.arange(0, tms_jumble.shape[0], batch_sz):
    print(itm)
    feed_dict = {anchor_model.map_cell_grid_tf: repeats_data['map_cell_grid'],
                 anchor_model.cell_types_tf: repeats_data['ctype_1hot'],
                 anchor_model.mean_fr_tf: repeats_data['mean_firing_rate'],
                 anchor_model.responses_tf: np.expand_dims(responses[tms_jumble[itm: itm+batch_sz], :], 2)}
    resp_embed[tms_jumble[itm: itm+batch_sz], :, :, :] = sess.run(anchor_model.responses_embed, feed_dict=feed_dict)

  # compute distance between pairs of responses
  distances = np.zeros((responses.shape[0], responses.shape[0]))
  distances_euclidean = np.zeros((responses.shape[0], responses.shape[0]))
  batch_dist = np.int(100)
  for iresp in np.arange(0, distances.shape[0], batch_dist):
    print(iresp)
    for jresp in np.arange(0, distances.shape[1], batch_dist):
      r1 = np.expand_dims(resp_embed[iresp: iresp+batch_dist], 1)
      r2 = np.expand_dims(resp_embed[jresp: jresp+batch_dist], 0)
      distances[iresp: iresp+batch_dist, jresp: jresp+batch_dist] = np.sum((r1-r2)**2, (2, 3, 4))

      rr1 = np.expand_dims(responses[iresp: iresp + batch_dist], 1)
      rr2 = np.expand_dims(responses[jresp: jresp + batch_dist], 0)
      distances_euclidean[iresp: iresp+batch_dist, jresp: jresp+batch_dist] = np.sum((rr1 - rr2)**2, 2)

  test_clustering = {'distances': distances,
                     'responses': responses,
                     'stim_idx': stim_idx,
                     'resp_embed': resp_embed,
                     'random_times': random_times,
                     'distances_euclidean': distances_euclidean}

  return test_clustering
