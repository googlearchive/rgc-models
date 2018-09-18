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
r"""Jointly embed stim-resp from multiple retina, conv embed for resp, time filter for stim."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import tensorflow as tf
import tensorflow.contrib.slim as slim
from absl import app
from absl import gfile
import numpy as np, h5py,numpy
import scipy.io as sio
from numpy.random import RandomState

# for plotting stuff
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import retina.response_model.python.metric_learning.end_to_end.utils as utils
import retina.response_model.python.metric_learning.end_to_end.convolutional_embed_resp as conv
import retina.response_model.python.metric_learning.end_to_end.data_util as data_util
from tensorflow.python.profiler.model_analyzer import PrintModelAnalysis

flags = tf.app.flags
flags.DEFINE_string('src_dir', '/home/bhaishahster/stim-resp4', 'Where is the datasets are')
flags.DEFINE_string('tmp_dir',
                    '/home/bhaishahster/'
                    'Downloads/stim-resp4',
                    'temporary folder on machine for better I/O')
flags.DEFINE_string('save_folder', '/home/bhaishahster/end_to_end4',
                    'where to store model and analysis')
flags.DEFINE_integer('max_iter', 2000000,
                    'Maximum number of iterations')
flags.DEFINE_integer('taskid', 0,
                    'Maximum number of iterations')

flags.DEFINE_integer('is_test', 0,
                    'If testing 1 , training 0')
flags.DEFINE_string('save_suffix',
                    '',
                    'suffix to save files')

flags.DEFINE_string('resp_layers',
                    '3, 128, 1, 3, 128, 1, 3, 128, 1, 3, 128, 2, 3, 128, 2, 3, 1, 1',
                    'suffix to save files')

flags.DEFINE_string('stim_layers',
                    '1, 5, 1, 3, 128, 1, 3, 128, 1, 3, 128, 1, 3, 128, 2, 3, 128, 2, 3, 1, 1',
                    'suffix to save files')

flags.DEFINE_bool('batch_norm',
                  True,
                  'Do we apply batch-norm regularization between layers?')

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=()):

  ## copy data locally
  dst = FLAGS.tmp_dir
  print('Starting Copy')
  if not gfile.IsDirectory(dst):
    gfile.MkDir(dst)

  files = gfile.ListDirectory(FLAGS.src_dir)
  for ifile in files:
    ffile = os.path.join(dst, ifile)
    if not gfile.Exists(ffile):
      gfile.Copy(os.path.join(FLAGS.src_dir, ifile), ffile)
      print('Copied %s' % os.path.join(FLAGS.src_dir, ifile))
    else:
      print('File exists %s' % ffile)

  print('File copied to destination')


  ## load data
  # load stimulus
  data = h5py.File(os.path.join(dst, 'stimulus.mat'))
  stimulus = np.array(data.get('stimulus')) - 0.5

  # load responses from multiple retina
  datasets_list = os.path.join(dst, 'datasets.txt')
  datasets = open(datasets_list, "r").read()
  training_datasets = [line for line in datasets.splitlines()]

  responses = []
  for idata in training_datasets:
    print(idata)
    data_file = os.path.join(dst, idata)
    data = sio.loadmat(data_file)
    responses += [data]
    print(np.max(data['centers'], 0))

  # generate additional features for responses
  num_cell_types = 2
  dimx = 80
  dimy = 40
  for iresp in responses:
    # remove cells which are outside 80x40 window.
    process_dataset(iresp, dimx, dimy, num_cell_types)

  ## generate graph -
  if FLAGS.is_test == 0:
    is_training = True
  if FLAGS.is_test == 1:
    is_training = True # False

  with tf.Session() as sess:

    ## Make graph
    # embed stimulus.
    time_window = 30
    stimx = stimulus.shape[1]
    stimy = stimulus.shape[2]
    stim_tf = tf.placeholder(tf.float32,
                             shape=[None, stimx,
                             stimy, time_window]) # batch x X x Y x time_window
    batch_norm = FLAGS.batch_norm
    stim_embed = embed_stimulus(FLAGS.stim_layers.split(','),
                                batch_norm, stim_tf, is_training,
                                reuse_variables=False)

    '''
    ttf_tf = tf.Variable(np.ones(time_window).astype(np.float32)/10, name='stim_ttf')
    filt = tf.expand_dims(tf.expand_dims(tf.expand_dims(ttf_tf, 0), 0), 3)
    stim_time_filt = tf.nn.conv2d(stim_tf, filt,
                                    strides=[1, 1, 1, 1], padding='SAME') # batch x X x Y x 1


    ilayer = 0
    stim_time_filt = slim.conv2d(stim_time_filt, 1, [3, 3],
                        stride=1,
                        scope='stim_layer_wt_%d' % ilayer,
                        reuse=False,
                        normalizer_fn=slim.batch_norm,
                        activation_fn=tf.nn.softplus,
                        normalizer_params={'is_training': is_training},
                        padding='SAME')
    '''


    # embed responses.
    num_cell_types = 2
    layers = FLAGS.resp_layers  # format: window x filters x stride .. NOTE: final filters=1, stride =1 throughout
    batch_norm = FLAGS.batch_norm
    time_window = 1
    anchor_model = conv.ConvolutionalProsthesisScore(sess, time_window=1,
                                                     layers=layers,
                                                     batch_norm=batch_norm,
                                                     is_training=is_training,
                                                     reuse_variables=False,
                                                     num_cell_types=2,
                                                     dimx=dimx, dimy=dimy)

    neg_model = conv.ConvolutionalProsthesisScore(sess, time_window=1,
                                                  layers=layers,
                                                  batch_norm=batch_norm,
                                                  is_training=is_training,
                                                  reuse_variables=True,
                                                  num_cell_types=2,
                                                  dimx=dimx, dimy=dimy)

    d_s_r_pos = tf.reduce_sum((stim_embed - anchor_model.responses_embed)**2, [1, 2, 3]) # batch
    d_pairwise_s_rneg = tf.reduce_sum((tf.expand_dims(stim_embed, 1) -
                                 tf.expand_dims(neg_model.responses_embed, 0))**2, [2, 3, 4]) # batch x batch_neg
    beta = 10
    # if FLAGS.save_suffix == 'lr=0.001':
    loss = tf.reduce_sum(beta * tf.reduce_logsumexp(tf.expand_dims(d_s_r_pos / beta, 1) -
                                                    d_pairwise_s_rneg / beta, 1), 0)
    # else :

    # loss = tf.reduce_sum(tf.nn.softplus(1 +  tf.expand_dims(d_s_r_pos, 1) - d_pairwise_s_rneg))
    accuracy_tf =  tf.reduce_mean(tf.sign(-tf.expand_dims(d_s_r_pos, 1) + d_pairwise_s_rneg))

    lr = 0.001
    train_op = tf.train.AdagradOptimizer(lr).minimize(loss)

    # set up training and testing data
    training_datasets_all = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
    testing_datasets = [0, 4, 8, 12]
    print('Testing datasets', testing_datasets)

    n_training_datasets_log = [1, 3, 5, 7, 9, 11, 12]

    if (np.floor(FLAGS.taskid / 4)).astype(np.int) < len(n_training_datasets_log):
      # do randomly sampled training data for 0<= FLAGS.taskid < 28
      prng = RandomState(23)
      n_training_datasets = n_training_datasets_log[(np.floor(FLAGS.taskid / 4)).astype(np.int)]
      for _ in range(10*FLAGS.taskid):
        print(prng.choice(training_datasets_all,
                          n_training_datasets, replace=False))
      training_datasets = prng.choice(training_datasets_all,
                                      n_training_datasets, replace=False)

      # training_datasets = [i for i in range(7) if i< 7-FLAGS.taskid] #[0, 1, 2, 3, 4, 5]


    else:
      # do 1 training data, chosen in order for FLAGS.taskid >= 28
      datasets_all = np.arange(16)
      training_datasets = [datasets_all[FLAGS.taskid % (4 * len(n_training_datasets_log))]]

    print('Task ID %d' % FLAGS.taskid)
    print('Training datasets', training_datasets)

    # Initialize stuff.
    file_name = ('end_to_end_stim_%s_resp_%s_beta_%d_taskid_%d'
                 '_training_%s_testing_%s_%s' % (FLAGS.stim_layers,
                                                 FLAGS.resp_layers, beta, FLAGS.taskid,
                                                 str(training_datasets)[1: -1],
                                                 str(testing_datasets)[1: -1],
                                                 FLAGS.save_suffix))
    saver_var, start_iter = initialize_model(FLAGS.save_folder, file_name, sess)

    # print model graph
    PrintModelAnalysis(tf.get_default_graph())

    # Add summary ops
    retina_number = tf.placeholder(tf.int16, name='input_retina');

    summary_ops = []
    for iret in np.arange(len(responses)):
      print(iret)
      r1 = tf.summary.scalar('loss_%d' % iret , loss)
      r2 = tf.summary.scalar('accuracy_%d' % iret , accuracy_tf)
      summary_ops += [tf.summary.merge([r1, r2])]

    # tf.summary.scalar('loss', loss)
    # tf.summary.scalar('accuracy', accuracy_tf)
    # summary_op = tf.summary.merge_all()

    # Setup summary writers
    summary_writers = []
    for loc in ['train', 'test']:
      summary_location = os.path.join(FLAGS.save_folder, file_name,
                                      'summary_' + loc )
      summary_writer = tf.summary.FileWriter(summary_location, sess.graph)
      summary_writers += [summary_writer]


    # start training
    batch_neg_train_sz = 100
    batch_train_sz = 100


    def batch(dataset_id):
      batch_train = get_batch(stimulus, responses[dataset_id]['responses'],
                              batch_size=batch_train_sz,
                              batch_neg_resp=batch_neg_train_sz,
                              stim_history=30, min_window=10)
      stim_batch, resp_batch, resp_batch_neg = batch_train
      feed_dict = {stim_tf: stim_batch,
                   anchor_model.responses_tf: np.expand_dims(resp_batch, 2),
                   neg_model.responses_tf: np.expand_dims(resp_batch_neg, 2),

                   anchor_model.map_cell_grid_tf: responses[dataset_id]['map_cell_grid'],
                   anchor_model.cell_types_tf: responses[dataset_id]['ctype_1hot'],
                   anchor_model.mean_fr_tf: responses[dataset_id]['mean_firing_rate'],

                   neg_model.map_cell_grid_tf: responses[dataset_id]['map_cell_grid'],
                   neg_model.cell_types_tf: responses[dataset_id]['ctype_1hot'],
                   neg_model.mean_fr_tf: responses[dataset_id]['mean_firing_rate'],
                   retina_number : dataset_id}

      return feed_dict

    def batch_few_cells(responses):
      batch_train = get_batch(stimulus, responses['responses'],
                              batch_size=batch_train_sz,
                              batch_neg_resp=batch_neg_train_sz,
                              stim_history=30, min_window=10)
      stim_batch, resp_batch, resp_batch_neg = batch_train
      feed_dict = {stim_tf: stim_batch,
                   anchor_model.responses_tf: np.expand_dims(resp_batch, 2),
                   neg_model.responses_tf: np.expand_dims(resp_batch_neg, 2),

                   anchor_model.map_cell_grid_tf: responses['map_cell_grid'],
                   anchor_model.cell_types_tf: responses['ctype_1hot'],
                   anchor_model.mean_fr_tf: responses['mean_firing_rate'],

                   neg_model.map_cell_grid_tf: responses['map_cell_grid'],
                   neg_model.cell_types_tf: responses['ctype_1hot'],
                   neg_model.mean_fr_tf: responses['mean_firing_rate'],
                   }

      return feed_dict

    if FLAGS.is_test == 1:
      print('Testing')
      save_dict = {}

      from IPython import embed; embed()
      ## Estimate one, fix others
      '''
      grad_resp = tf.gradients(d_s_r_pos, anchor_model.responses_tf)

      t_start = 1000
      t_len = 100
      stim_history = 30
      stim_batch = np.zeros((t_len, stimulus.shape[1],
                         stimulus.shape[2], stim_history))
      for isample, itime in enumerate(np.arange(t_start, t_start + t_len)):
        stim_batch[isample, :, :, :] = np.transpose(stimulus[itime: itime-stim_history:-1, :, :], [1, 2, 0])

      iretina = testing_datasets[0]
      resp_batch = np.expand_dims(np.random.rand(t_len, responses[iretina]['responses'].shape[1]), 2)

      step_sz = 0.01
      eps = 1e-2
      dist_prev = np.inf
      for iiter in range(10000):
        feed_dict = {stim_tf: stim_batch,
                     anchor_model.map_cell_grid_tf: responses[iretina]['map_cell_grid'],
                     anchor_model.cell_types_tf: responses[iretina]['ctype_1hot'],
                     anchor_model.mean_fr_tf: responses[iretina]['mean_firing_rate'],
                     anchor_model.responses_tf: resp_batch}
        dist_np, resp_grad_np = sess.run([d_s_r_pos, grad_resp], feed_dict=feed_dict)
        if np.sum(np.abs(dist_prev - dist_np)) < eps:
          break
        print(np.sum(dist_np), np.sum(np.abs(dist_prev - dist_np)))
        dist_prev = dist_np
        resp_batch = resp_batch - step_sz * resp_grad_np[0]

      resp_batch = resp_batch.squeeze()
      '''


      # from IPython import embed; embed()
      ## compute distances between s-r pairs for small number of cells

      test_retina = []
      for iretina in range(len(testing_datasets)):
        dataset_id = testing_datasets[iretina]

        num_cells_total = responses[dataset_id]['responses'].shape[1]
        dataset_center = responses[dataset_id]['centers'].mean(0)
        dataset_cell_distances = np.sqrt(np.sum((responses[dataset_id]['centers'] -
                                         dataset_center), 1))
        order_cells = np.argsort(dataset_cell_distances)

        test_sr_few_cells = {}
        for num_cells_prc in [5, 10, 20, 30, 50, 100]:
          num_cells = np.percentile(np.arange(num_cells_total),
                                    num_cells_prc).astype(np.int)

          choose_cells = order_cells[:num_cells]

          resposnes_few_cells = {'responses': responses[dataset_id]['responses'][:, choose_cells],
                                 'map_cell_grid': responses[dataset_id]['map_cell_grid'][:, :, choose_cells],
                                'ctype_1hot': responses[dataset_id]['ctype_1hot'][choose_cells, :],
                                'mean_firing_rate': responses[dataset_id]['mean_firing_rate'][choose_cells]}
          # get a batch
          d_pos_log = np.array([])
          d_neg_log = np.array([])
          for test_iter in range(1000):
            print(iretina, num_cells_prc, test_iter)
            feed_dict = batch_few_cells(resposnes_few_cells)
            d_pos, d_neg = sess.run([d_s_r_pos, d_pairwise_s_rneg], feed_dict=feed_dict)
            d_neg = np.diag(d_neg) # np.mean(d_neg, 1) #
            d_pos_log = np.append(d_pos_log, d_pos)
            d_neg_log = np.append(d_neg_log, d_neg)

          precision_log, recall_log, F1_log, FPR_log, TPR_log = ROC(d_pos_log, d_neg_log)

          print(np.sum(d_pos_log > d_neg_log))
          print(np.sum(d_pos_log < d_neg_log))
          test_sr= {'precision': precision_log, 'recall': recall_log,
                     'F1': F1_log, 'FPR': FPR_log, 'TPR': TPR_log,
                     'd_pos_log': d_pos_log, 'd_neg_log': d_neg_log,
                    'num_cells': num_cells}

          test_sr_few_cells.update({'num_cells_prc_%d' % num_cells_prc : test_sr})
        test_retina += [test_sr_few_cells]
      save_dict.update({'few_cell_analysis': test_retina})

      ## compute distances between s-r pairs - pos and neg.

      test_retina = []
      for iretina in range(len(testing_datasets)):
        # stim-resp log
        d_pos_log = np.array([])
        d_neg_log = np.array([])
        for test_iter in range(1000):
          print(test_iter)
          feed_dict = batch(testing_datasets[iretina])
          d_pos, d_neg = sess.run([d_s_r_pos, d_pairwise_s_rneg], feed_dict=feed_dict)
          d_neg = np.diag(d_neg) # np.mean(d_neg, 1) #
          d_pos_log = np.append(d_pos_log, d_pos)
          d_neg_log = np.append(d_neg_log, d_neg)

        precision_log, recall_log, F1_log, FPR_log, TPR_log = ROC(d_pos_log, d_neg_log)

        print(np.sum(d_pos_log > d_neg_log))
        print(np.sum(d_pos_log < d_neg_log))
        test_sr = {'precision': precision_log, 'recall': recall_log,
                     'F1': F1_log, 'FPR': FPR_log, 'TPR': TPR_log,
                   'd_pos_log': d_pos_log, 'd_neg_log': d_neg_log}

        test_retina += [test_sr]

      save_dict.update({'test_sr': test_retina})


      ## ROC curves of responses from repeats - dataset 1
      repeats_datafile = '/home/bhaishahster/metric_learning/datasets/2015-09-23-7.mat'
      repeats_data = sio.loadmat(gfile.Open(repeats_datafile, 'r'));
      repeats_data['cell_type'] = repeats_data['cell_type'].T

      # process repeats data
      process_dataset(repeats_data, dimx, dimy, num_cell_types)

      # analyse and store the result
      test_reps = analyse_response_repeats(repeats_data, anchor_model, neg_model, sess)
      save_dict.update({'test_reps_2015-09-23-7': test_reps})

      ## ROC curves of responses from repeats - dataset 2
      repeats_datafile = '/home/bhaishahster/metric_learning/examples_pc2005_08_03_0/data005_test.mat'
      repeats_data = sio.loadmat(gfile.Open(repeats_datafile, 'r'));
      process_dataset(repeats_data, dimx, dimy, num_cell_types)

      # analyse and store the result
      '''
      test_clustering = analyse_response_repeats_all_trials(repeats_data, anchor_model, neg_model, sess)
      save_dict.update({'test_reps_2005_08_03_0': test_clustering})
      '''
      #
      # get model params
      save_dict.update({'model_pars': sess.run(tf.trainable_variables())})


      save_analysis_filename = os.path.join(FLAGS.save_folder, file_name + '_analysis.pkl')
      pickle.dump(save_dict, gfile.Open(save_analysis_filename, 'w'))
      print(save_analysis_filename)
      return

    test_iiter = 0
    for iiter in range(start_iter, FLAGS.max_iter): # TODO(bhaishahster) :add FLAGS.max_iter

      # get a new batch
      # stim_tf, anchor_model.responses_tf, neg_model.responses_tf

      # training step
      train_dataset = training_datasets[iiter % len(training_datasets)]
      feed_dict_train = batch(train_dataset)
      _, loss_np_train = sess.run([train_op, loss], feed_dict=feed_dict_train)
      print(train_dataset, loss_np_train)

      # write summary
      if iiter % 10 == 0:
        # write train summary
        test_iiter = test_iiter + 1

        train_dataset = training_datasets[test_iiter % len(training_datasets)]
        feed_dict_train = batch(train_dataset)
        summary_train = sess.run(summary_ops[train_dataset], feed_dict=feed_dict_train)
        summary_writers[0].add_summary(summary_train, iiter)

        # write test summary
        test_dataset = testing_datasets[test_iiter % len(testing_datasets)]
        feed_dict_test = batch(test_dataset)
        l_test, summary_test = sess.run([loss, summary_ops[test_dataset]], feed_dict=feed_dict_test)
        summary_writers[1].add_summary(summary_test, iiter)
        print('Test retina: %d, loss: %.3f' % (test_dataset, l_test))

      # save model
      if iiter % 10 == 0:
        save_model(saver_var, FLAGS.save_folder, file_name, sess, iiter)



def ROC(distances_pos, distances_neg):
  """Compute ROC curve."""

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
    print(precision, recall)
    F1 = 2 * precision * recall / (precision + recall)

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    precision_log += [precision]
    recall_log += [recall]
    F1_log += [F1]
    TPR_log += [TPR]
    FPR_log += [FPR]

  return precision_log, recall_log, F1_log, FPR_log, TPR_log

def get_batch(stimulus, responses, batch_size=100, batch_neg_resp=100,
              stim_history=30, min_window=10):
  """Get a batch of training data."""

  stim = stimulus
  resp = responses

  t_max = np.minimum(stim.shape[0], resp.shape[0])
  stim_batch = np.zeros((batch_size, stim.shape[1],
                         stim.shape[2], stim_history))
  resp_batch = np.zeros((batch_size, resp.shape[1]))

  random_times = np.random.randint(stim_history, t_max - 1, batch_size)
  for isample in range(batch_size):
    itime = random_times[isample]
    stim_batch[isample, :, :, :] = np.transpose(stim[itime:
                                                     itime-stim_history:-1,
                                                     :, :],
                                                [1, 2, 0])
    resp_batch[isample, :] = resp[itime, :]

  # get negative responses.
  resp_batch_neg = np.zeros((batch_neg_resp, resp.shape[1]))
  for isample in range(batch_neg_resp):
    itime = np.random.randint(stim_history, t_max - 1, 1)
    while np.min(np.abs(itime - random_times)) < min_window:
      itime = np.random.randint(stim_history, t_max - 1, 1)
    resp_batch_neg[isample, :] = resp[itime, :]

  return stim_batch, resp_batch, resp_batch_neg

def give_cell_grid(centers, resolution, dimx=80, dimy=40):
  """Embeds each center on a discrete grid.

  Args:
    centers: center location of cells (n_cells x 2).
    resolution: Float specifying the resolution of grid.

  Returns:
    centers_grid : Discretized centers (n_cells x 2).
    grid_size : dimensions of the grid (2D integer tuple).
    map_cell_grid : mapping between cells to grid (grid_x x grid_y x n_cells)
  """

  n_cells = centers.shape[0]
  centers_grid = np.floor(centers -1 / resolution) # subtract 1 because matlab indexing starts from 1.
  # centers_grid -= np.min(centers_grid, 0)
  grid_size =[dimx, dimy]

  # map_cell_grid is location of each cell to grid point
  map_cell_grid = np.zeros((grid_size[0], grid_size[1], n_cells))
  for icell in range(n_cells):
    map_cell_grid[centers_grid[icell, 0], centers_grid[icell, 1], icell] = 1

  return centers_grid, grid_size, map_cell_grid

## Initialize and do saving, etc
def initialize_model(save_folder, file_name, sess):
  """Setup model variables and saving information.

  Args:
    save_folder (string) : Folder to store model.
                           Makes one if it does not exist.
    filename (string) : Prefix of model/checkpoint files.
    sess : Tensorflow session.
  """

  # Make folder.
  if not gfile.IsDirectory(save_folder):
    gfile.MkDir(save_folder)

  # Initialize variables.
  saver_var, start_iter = initialize_variables(sess, save_folder, file_name)
  return saver_var, start_iter

def initialize_variables(sess, save_folder, short_filename):
  """Initialize variables or restore from previous fits."""

  sess.run(tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer()))
  saver_var = tf.train.Saver(tf.global_variables(),
                             keep_checkpoint_every_n_hours=20)
  load_prev = False
  start_iter = 0
  try:
    # Restore previous fits if they are available
    # - useful when programs are preempted frequently on .
    latest_filename = short_filename + '_latest_fn'
    restore_file = tf.train.latest_checkpoint(save_folder,
                                              latest_filename)
    # Restore previous iteration count and start from there.
    start_iter = int(restore_file.split('/')[-1].split('-')[-1])
    saver_var.restore(sess, restore_file)  # restore variables
    load_prev = True
  except:
    tf.logging.info('No previous dataset')

  if load_prev:
    tf.logging.info('Previous results loaded from: ' + restore_file)
  else:
    tf.logging.info('Variables initialized')

  tf.logging.info('Loaded iteration: %d' % start_iter)

  return saver_var, start_iter

def save_model(saver_var, save_folder, file_name, sess, iiter):
  """Save model variables."""
  latest_filename = file_name + '_latest_fn'
  long_filename = os.path.join(save_folder, file_name)
  saver_var.save(sess, long_filename, global_step=iiter,
                      latest_filename=latest_filename)

def process_dataset(iresp, dimx, dimy, num_cell_types):
  """Process the dataset"""
  valid_cells0 = np.logical_and(iresp['centers'][:, 0]<=dimx, iresp['centers'][:, 1]<=dimy)
  valid_cells1 = np.logical_and(iresp['centers'][:, 0]>0, iresp['centers'][:, 1]>0)
  valid_cells = np.logical_and(valid_cells0, valid_cells1)

  iresp['centers'] = iresp['centers'][valid_cells, :]
  try:
    iresp['sta_params'] = iresp['sta_params'][valid_cells, :]
  except:
    print('No STA params')

  try:
    iresp['responses'] = iresp['responses'][:, valid_cells]
    mean_resp = np.mean(iresp['responses'], 0)
  except :
    print('No responses')

  try:
    iresp['repeats'] = iresp['repeats'][:, :, valid_cells]
    mean_resp = np.mean(np.mean(iresp['repeats'], 0), 0)
  except :
    print('No repeats')

  try:
    iresp['cellID_list'] = iresp['cellID_list'][:, valid_cells]
  except:
    print('No cell ID list')

  iresp['cell_type'] = iresp['cell_type'][:, valid_cells]
  print('Valid cells: %d/%d' % (np.sum(valid_cells), valid_cells.shape[0]))

  # compute mean firing rate for cells
  n_cells = np.squeeze(iresp['centers']).shape[0]


  # do embedding of centers on a grid
  # TODO(bhaishahster): compute centers from stim-resp here itself ?
  centers_grid, grid_size, map_cell_grid =  give_cell_grid(iresp['centers'],
                                                           resolution=1)

  # encode cell type as 1-hot vector
  ctype_1hot = np.zeros((n_cells, num_cell_types))
  for icell_type in np.arange(1, num_cell_types+1):
    ctype_1hot[:, icell_type-1] = np.double(iresp['cell_type'] == icell_type)

  iresp.update({'mean_firing_rate': mean_resp,
                'map_cell_grid': map_cell_grid,
                'ctype_1hot': ctype_1hot})

def embed_stimulus(layers, batch_norm, net, is_training, reuse_variables=False):
  n_layers = int(len(layers)/3)
  tf.logging.info('Number of layers: %d' % n_layers)
  # set normalization
  if batch_norm:
    normalizer_fn = slim.batch_norm
    tf.logging.info('Batch normalization')
  else:
    normalizer_fn = None

  activation_fn = tf.nn.softplus
  tf.logging.info('Logistic activation')

  for ilayer in range(n_layers):
    tf.logging.info('Building layer: %d, %d, %d'
                    % (int(layers[ilayer*3 + 1]), int(layers[ilayer*3]),
                       int(layers[ilayer*3 + 2])))
    net = slim.conv2d(net, int(layers[ilayer*3 + 1]),
                      int(layers[ilayer*3]),
                      stride=int(layers[ilayer*3 + 2]),
                      scope='stim_layer_wt_%d' % ilayer,
                      reuse=reuse_variables,
                      normalizer_fn=normalizer_fn,
                      activation_fn=activation_fn,
                      normalizer_params={'is_training': is_training},
                      )
  return net

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

if __name__ == '__main__':
  app.run(main)
