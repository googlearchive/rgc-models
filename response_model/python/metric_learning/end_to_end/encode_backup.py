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
""" Response prediction code (old version).

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

FLAGS = flags.FLAGS


  def raster(respMat,shift=0,color='r'):
    # for plotting stuff
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    # trials x spikes
    shp = np.shape(respMat)
    for itrial in range(shp[0]):
      iidx = np.arange(shp[1])
      ith = iidx[respMat[itrial,:]!=0]
      plt.vlines(ith, itrial+shift,itrial+1+shift, color=color)
      plt.hold(True)
    plt.xlim((0,respMat.shape[1]))

    
  #
  #     '''
  #     for iiter in range(15):
  #       # Use score functions
  #       theta = -2 * np.ones((t_len, n_cells))
  #       step_sz = 0.001
  #       eps = 1e-1
  #       dist_prev = np.inf
  #       d_log = []
  #
  #       for iiter in range(1000):
  #
  #         # Score function method
  #         lam = np.exp(theta)
  #         resp_sample = np.random.poisson(lam)
  #         resp_batch = np.expand_dims(resp_sample, 2)
  #
  #         feed_dict = {sr_graph.stim_tf: stim_batch,
  #                      sr_graph.anchor_model.map_cell_grid_tf: responses[iretina]['map_cell_grid'],
  #                      sr_graph.anchor_model.cell_types_tf: responses[iretina]['ctype_1hot'],
  #                      sr_graph.anchor_model.mean_fr_tf: responses[iretina]['mean_firing_rate'],
  #                      sr_graph.anchor_model.responses_tf: resp_batch}
  #         dist_np = sr_graph.sess.run(sr_graph.d_s_r_pos, feed_dict=feed_dict)
  #         if np.sum(np.abs(dist_prev - dist_np)) < eps:
  #           break
  #         print(np.sum(dist_np), np.sum(np.abs(dist_prev - dist_np)))
  #         dist_prev = dist_np
  #         d_log += [np.sum(dist_np)]
  #         theta = theta - step_sz * (np.expand_dims(dist_np, 1) * (resp_sample - lam))
  #         theta = np.minimum(theta, 2) # TODO(bhaishahster) : remove this hack!
  #         plt.ion()
  #         plt.cla()
  #         plt.plot(d_log)
  #         plt.show()
  #         plt.draw()
  #         plt.pause(0.05)
  #     '''
  #
  #
  #     '''
  #     for iiter in range(15):
  #       # Relaxed optimization.
  #       grad_resp = tf.gradients(sr_graph.d_s_r_pos, sr_graph.anchor_model.responses_tf)
  #       def sigmoid(x):
  #         return 1/(1 + np.exp(-x))
  #
  #       def d_sigmoid(x):
  #         return np.exp(x) /((1 + np.exp(x))**2)
  #
  #       theta = np.random.randn(t_len, n_cells) * 0.1
  #       step_sz = 1
  #       eps = 1e-1
  #       dist_prev = np.inf
  #       for iiter in range(100):
  #         """
  #         cells = [23, 54, 12, 200]
  #         plt.ion()
  #         for icell in range(4):
  #           plt.subplot(2, 2, icell + 1)
  #           plt.plot(sigmoid(theta[:, cells[icell]]))
  #         plt.draw()
  #         plt.show()
  #         plt.pause(0.05)
  #         """
  #
  #         resp_sample = sigmoid(theta)
  #         resp_batch = np.expand_dims(resp_sample, 2)
  #
  #         feed_dict = {sr_graph.stim_tf: stim_batch,
  #                      sr_graph.anchor_model.map_cell_grid_tf: responses[iretina]['map_cell_grid'],
  #                      sr_graph.anchor_model.cell_types_tf: responses[iretina]['ctype_1hot'],
  #                      sr_graph.anchor_model.mean_fr_tf: responses[iretina]['mean_firing_rate'],
  #                      sr_graph.anchor_model.responses_tf: resp_batch}
  #         dist_np, resp_grad_np = sr_graph.sess.run([sr_graph.d_s_r_pos, grad_resp], feed_dict=feed_dict)
  #         if np.sum(np.abs(dist_prev - dist_np)) < eps:
  #           break
  #         print(np.sum(dist_np), np.sum(np.abs(dist_prev - dist_np)))
  #         dist_prev = dist_np
  #         theta = theta - step_sz * (d_sigmoid(resp_sample) * resp_grad_np[0].squeeze())
  #     '''
  #
  #     '''
  #       # save results
  #       resp_batch = resp_batch.squeeze()
  #       resp_iters += [resp_batch]
  #
  #     sample_resp_retina += [{'sample_responses': resp_iters,
  #                             'stim_batch': stim_batch,
  #                             'iretina': iretina}]
  #   save_dict.update({'retina_samples': sample_resp_retina})
  #     '''
  #
  #   # from IPython import embed; embed()
  #   ## compute distances between s-r pairs for small number of cells
  #   '''
  #   test_retina = []
  #   for iretina in range(len(testing_datasets)):
  #     dataset_id = testing_datasets[iretina]
  #
  #     num_cells_total = responses[dataset_id]['responses'].shape[1]
  #     dataset_center = responses[dataset_id]['centers'].mean(0)
  #     dataset_cell_distances = np.sqrt(np.sum((responses[dataset_id]['centers'] -
  #                                      dataset_center), 1))
  #     order_cells = np.argsort(dataset_cell_distances)
  #
  #     test_sr_few_cells = {}
  #     for num_cells_prc in [5, 10, 20, 30, 50, 100]:
  #       num_cells = np.percentile(np.arange(num_cells_total),
  #                                 num_cells_prc).astype(np.int)
  #
  #       choose_cells = order_cells[:num_cells]
  #
  #       resposnes_few_cells = {'responses': responses[dataset_id]['responses'][:, choose_cells],
  #                              'map_cell_grid': responses[dataset_id]['map_cell_grid'][:, :, choose_cells],
  #                             'ctype_1hot': responses[dataset_id]['ctype_1hot'][choose_cells, :],
  #                             'mean_firing_rate': responses[dataset_id]['mean_firing_rate'][choose_cells]}
  #       # get a batch
  #       d_pos_log = np.array([])
  #       d_neg_log = np.array([])
  #       for test_iter in range(1000):
  #         print(iretina, num_cells_prc, test_iter)
  #         feed_dict = batch_few_cells(sr_graph, resposnes_few_cells,
  #                                     stimuli[responses[dataset_id]['stimulus_key']],
  #                                    batch_train_sz=100, batch_neg_train_sz=100)
  #
  #         d_pos, d_neg = sr_graph.sess.run([sr_graph.d_s_r_pos,
  #                                           sr_graph.d_pairwise_s_rneg],
  #                                          feed_dict=feed_dict)
  #         d_neg = np.diag(d_neg) # np.mean(d_neg, 1) #
  #         d_pos_log = np.append(d_pos_log, d_pos)
  #         d_neg_log = np.append(d_neg_log, d_neg)
  #
  #       precision_log, recall_log, F1_log, FPR_log, TPR_log = ROC(d_pos_log, d_neg_log)
  #
  #       print(np.sum(d_pos_log > d_neg_log))
  #       print(np.sum(d_pos_log < d_neg_log))
  #       test_sr= {'precision': precision_log, 'recall': recall_log,
  #                  'F1': F1_log, 'FPR': FPR_log, 'TPR': TPR_log,
  #                  'd_pos_log': d_pos_log, 'd_neg_log': d_neg_log,
  #                 'num_cells': num_cells}
  #
  #       test_sr_few_cells.update({'num_cells_prc_%d' % num_cells_prc : test_sr})
  #     test_retina += [test_sr_few_cells]
  #   save_dict.update({'few_cell_analysis': test_retina})
  #   '''
