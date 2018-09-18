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
r"""Playground of testing code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

FLAGS = flags.FLAGS



  # Isolate the stimulus at differnet frames
  # probes = np.array([[-5, -15], [20, -12], [10, 10], [-5, 8]])
  probes = np.array([[4, 1], [18, 14], [-14, -7], [-15, 2], [-6, 14]])
  plt.scatter(probes[:, 0], probes[:, 1], 40, 'g',  linewidths=0)
  plt.show()

  # Show the frames at t-4
  plt.figure()
  n_nearest = 10
  for iprobe in range(probes.shape[0]):
    probe_plot = probes[iprobe, :]
    distances_probe = np.sum((tSNE_embedding - probe_plot)**2, 1)
    nearest_responses = np.argsort(distances_probe)
    retrieve_idx = nearest_responses[:n_nearest]
    retrieved_stim_mean = np.mean(stimulus_test[retrieve_idx, :, :, 4], 0)

    plt.subplot(probes.shape[0], 5, 5*iprobe + 1)
    plt.imshow(retrieved_stim_mean.T, cmap='gray', interpolation='nearest')
    plt.axis('off')
    if iprobe == 0:
      plt.title('Mean of retrieved stimulus')

    for iret in [0, 1, 2]:
      plt.subplot(probes.shape[0], 5, 5*iprobe + iret + 2)
      plt.imshow(stimulus_test[retrieve_idx[iret], :, :, 4].T,
                 cmap='gray', interpolation='nearest')
      plt.axis('off')
      if iprobe == 0:
        plt.title('Retrieved stim %d' % iret)

    plt.subplot(probes.shape[0], 5, 5*iprobe + 5)
    retrieved_stim_mean_blur = scipy.ndimage.filters.gaussian_filter(retrieved_stim_mean, 2)
    plt.imshow(retrieved_stim_mean_blur.T, cmap='gray', interpolation='nearest')
    plt.axis('off')
    if iprobe == 0:
      plt.title('Mean of retrieved stimulus (blurred)')
  plt.suptitle('Frames at t-4')
  plt.show()

  # Temporal component
  plt.figure()
  n_nearest = 10
  for iprobe in range(probes.shape[0]):
    probe_plot = probes[iprobe, :]
    distances_probe = np.sum((tSNE_embedding - probe_plot)**2, 1)
    nearest_responses = np.argsort(distances_probe)
    retrieve_idx = nearest_responses[:n_nearest]
    retrieved_stim = stimulus_test[retrieve_idx, :, :, :]
    ttf_log = []
    for iret in range(n_nearest):
      xx = retrieved_stim[iret, :, :, :]
      ttf_log += [np.mean(np.mean(np.expand_dims(np.sign(xx[:, :, 4]), 2) * xx, 0), 0)]
    ttf_log = np.array(ttf_log)

    plt.subplot(3, 2, iprobe + 1)
    plt.plot(ttf_log.T, 'k', alpha=0.4)
    plt.plot(ttf_log.mean(0), linewidth=2)
    # plt.axhline(0)
    # plt.axis('off')

  plt.suptitle('Time course')
  plt.show()

  # Put each image at the embedded location
  fig = plt.figure()
  ax = plt.subplot(111)
  plt.plot([tSNE_embedding.min(0)[0], tSNE_embedding.max(0)[0]],
           [tSNE_embedding.min(0)[1], tSNE_embedding.max(0)[1]], alpha=0)
  for itarget in [1]: #range(stimulus_test.shape[0]):
    #s_blur = scipy.ndimage.filters.gaussian_filter(stimulus_test[itarget, :, :, 4], 1)
    s_blur = stimulus_test[itarget, :, :, 4]
    newax = plt.axes([tSNE_embedding[itarget, 0], tSNE_embedding[itarget, 1], 40, 40])
    #newax = fig.add_axes([tSNE_embedding[itarget, 0], tSNE_embedding[itarget, 1], 10, 10])
    newax.imshow(s_blur, cmap='gray', interpolation='nearest')

    #ax.figure.figimage(s_blur.T, tSNE_embedding[itarget, 0], tSNE_embedding[itarget, 1], alpha=0.5, zorder=1, cmap='gray', origin='lower')
    #newax.axis('off')
  plt.show()





  '''
  # Decoding: Predict stimulus
  batch_size = 300
  stim_history = 30
  sample_resp_retina = []
  for iretina in testing_datasets[::-1]:


    stimulus = stimuli[responses[iretina]['stimulus_key']]
    n_cells = responses[iretina]['responses'].shape[1]

    # make test responses
    t_min = 0
    t_max = 30000
    resp_test = responses[iretina]['responses'][t_min: t_max, :]
    random_times = np.random.randint(30, resp_test.shape[0], batch_size)
    resp_test_sample = resp_test[random_times, :]
    resp_test_sample = np.expand_dims(resp_test_sample, 2)
    # decode stimulus
    stim_decode = decode.predict_stimulus(resp_test_sample,
                                          sr_graph, responses[iretina])

    stimulus_target = np.zeros((batch_size, 80, 40, 30))
    stimulus_test = stimulus[t_min: t_max, :, :]
    for iitm, itm in enumerate(random_times):
      stimulus_target[iitm, :, :, :] = np.expand_dims(np.transpose(stimulus_test[itm:itm-30:-1, :, :], [1, 2, 0]), 0)

    # plot_decoding(stimulus_target, stim_decode, n_targets=10)
    sample_resp_retina += [{'stim_decoded': stimulus_target,
                            'stim_target': stimulus_target,
                            'response_test': resp_test_sample,
                            'iretina': iretina}]
    save_dict.update({'decode': sample_resp_retina})
    pickle.dump(save_dict, gfile.Open(save_analysis_filename, 'w'))
    print('Responses generated for retina %d' % iretina)
  '''



  '''
  # Encoding: Predict responses
  t_len = 300
  stim_history = 30
  sample_resp_retina = []
  for iretina in testing_datasets[::-1]:
    from IPython import embed; embed()

    stimulus = stimuli[responses[iretina]['stimulus_key']]
    n_cells = responses[iretina]['responses'].shape[1]
    theta_np_overall = np.zeros((0, n_cells))
    stim_batch_overall = np.zeros((0, stimulus.shape[1],
                                   stimulus.shape[2] , stim_history))

    for t_start in np.arange(1000, 2000, t_len):

      stim_batch = np.zeros((t_len, stimulus.shape[1],
                   stimulus.shape[2], stim_history))
      for isample, itime in enumerate(np.arange(t_start, t_start + t_len)):
        stim_batch = np.zeros((t_len, stimulus.shape[1],
                     stimulus.shape[2], stim_history))
        for isample, itime in enumerate(np.arange(t_start, t_start + t_len)):
          stim_batch[isample, :, :, :] = np.transpose(stimulus[itime: itime-stim_history:-1, :, :], [1, 2, 0])

        theta_np, r_s = encode.predict_responses(stim_batch, sr_graph, responses[iretina])

        theta_np_overall = np.append(theta_np_overall, theta_np, 0)
        stim_batch_overall = np.append(stim_batch_overall, stim_batch, 0)


    sample_resp_retina += [{'firing_rate': theta_np_overall,
                            'iretina': iretina,
                            }] # , 'theta_np_log': theta_np_log}]
    save_dict.update({'retina_samples': sample_resp_retina})
    pickle.dump(save_dict, gfile.Open(save_analysis_filename, 'w'))
    print('Responses generated for retina %d' % iretina)

  '''


