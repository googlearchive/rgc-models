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
"""This code is used for analysing jitter stimulus models.

Hopefully, finer resolution gives better subunit estimates.
"""

import sys
import os.path
import tensorflow as tf
from absl import app
from absl import flags
from absl import gfile
import cPickle as pickle
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import ndimage
import random


FLAGS = flags.FLAGS
# flags for data locations
flags.DEFINE_string('folder_name', 'experiment_jitter', 'folder where to store all the data')
flags.DEFINE_string('save_location',
                    '/home/bhaishahster/',
                    'where to store logs and outputs?');
flags.DEFINE_string('data_location',
                    '/home/foam/retina_data/Google datasets/2016-04-21-1/data006(2016-04-21-1_data006_data006)/',
                    'where to take data from?')

# flags for stochastic learning
flags.DEFINE_integer('batchsz', 100, 'batch size for training')
flags.DEFINE_integer('n_chunks',1793, 'number of data chunks') # should be 216
flags.DEFINE_integer('n_b_in_c', 1, 'number of batches in one chunk of data')
flags.DEFINE_integer('train_len', 216 - 21, 'how much training length to use?')
flags.DEFINE_float('step_sz', 200, 'step size for learning algorithm')

# random number generators initialized
# removes unneccessary data variabilities while comparing algorithms
flags.DEFINE_integer('np_randseed', 23, 'numpy RNG seed')
flags.DEFINE_integer('randseed', 65, 'python RNG seed')


# flags for model/loss specification
flags.DEFINE_string('model_id', 'relu_window_mother_sfm', 'which model to fit')
flags.DEFINE_string('loss', 'poisson', 'which loss to use?')

# model specific terms
# useful for convolution-like models
flags.DEFINE_integer('window', 16, 'size of window for each subunit in relu_window model')
flags.DEFINE_integer('stride', 16, 'stride for relu_window')
flags.DEFINE_integer('su_channels', 3, 'number of color channels each subunit should take input from')

# some models need regularization of parameters
flags.DEFINE_float('lam_w', 0.0001, 'sparsitiy regularization of w')
flags.DEFINE_float('lam_a', 0.0001, 'sparsitiy regularization of a')

# dataset specific
flags.DEFINE_float('n_cells',1, 'number of cells in the dataset')
FLAGS = flags.FLAGS

response = np.array([])
test_chunks = np.array([])
train_chunks = np.array([])
train_counter = np.array([])

def init_chunks():
  # initialize the chunks, called at the beginning of the code.
  global test_chunks
  global train_chunks
  global train_counter
  random_chunks = np.arange(FLAGS.n_chunks)+1
  test_chunks = random_chunks[0:20]
  train_chunks = random_chunks[22:]
  train_counter =0


def get_stim_resp(data_type='train'):
  # this function gets you the training and testing chunks!
  # permute chunks and decide the training and test chunks
  global train_chunks
  global test_chunks
  global train_counter
  global response

  def get_stimulus_batch(ichunk):
    stim_path = FLAGS.data_location + 'Stimulus/'
    stim_file = sio.loadmat(gfile.Open(stim_path+'stim_chunk_' + str(ichunk) + '.mat'))
    chunk_start = np.squeeze(stim_file['chunk_start'])
    chunk_end = np.squeeze(stim_file['chunk_end'])
    jump = stim_file['jump']
    stim_chunk = stim_file['stimulus_chunk']
    stim_chunk = np.transpose(stim_chunk, [3,0,1,2])
    return stim_chunk, chunk_start, chunk_end

  if(data_type=='test'):
    print('Loading test')
    chunk_ids = np.array(test_chunks).astype('int')
    
  if(data_type=='train'):
    print('Loading train')
    if(train_counter>=train_chunks.shape[0]):
      train_chunks = np.shuffle(train_chunks)
      train_counter=0
    chunk_ids = [np.squeeze(np.array(train_chunks[train_counter]).astype('int'))]
    train_counter =train_counter + 1
    
  stim_total = np.zeros((0,640,320,3))
  resp_total =np.zeros((0,FLAGS.n_cells))
  data_len_total = 0
    
  for ichunk in chunk_ids:
    print('Loading chunk:' + str(ichunk))
    # get chunks
    if(ichunk==chunk_ids[0]):
      print('first entry')
      # first entry into the chunk
      stim_chunk, chunk_start, chunk_end  = get_stimulus_batch(ichunk)
      resp_chunk = response[chunk_start+29:chunk_end+1,:]
    else:
      print('second entry')
      stim_chunk, chunk_start, chunk_end  = get_stimulus_batch(ichunk)
      stim_chunk = stim_chunk[30:,:,:,:]
      resp_chunk = response[chunk_start+30-1:chunk_end,:]

    data_len = resp_chunk.shape[0]
    print(chunk_start, chunk_end)
    print(np.shape(stim_chunk), np.shape(resp_chunk))

    # remove overlapping parts of chunks and then append them!
    stim_total = np.append(stim_total, stim_chunk, axis=0)
    resp_total = np.append(resp_total, resp_chunk, axis=0)
    data_len_total = data_len_total + data_len
  return stim_total, resp_total, data_len_total


def setup_dataset():
  # initialize paths, get dataset properties, etc
  path = FLAGS.data_location

  # load cell response
  response_path = path + 'response.mat'
  response_file = sio.loadmat(gfile.Open(response_path))
  resp_mat = response_file['binned_spikes']
  resp_file_cids = np.squeeze(response_file['cell_ids'])

  # load off parasol cell IDs
  cids_path = path + 'cell_ids/cell_ids_OFF parasol.mat'
  cids_file = sio.loadmat(gfile.Open(cids_path))
  cids_select = np.squeeze(cids_file['cids'])

  # find index of cells to choose from resp_mat
  resp_file_choose_idx = np.array([])
  for icell in np.array(cids_select):
    idx = np.where(resp_file_cids == icell)
    resp_file_choose_idx = np.append(resp_file_choose_idx, idx[0])
    

  # finally, get selected cells from resp_mat
  global response
  response = resp_mat[resp_file_choose_idx.astype('int'),:].T

  # load population time courses
  time_c_file_path = path + 'cell_ids/time_courses.mat'
  time_c_file = sio.loadmat(gfile.Open(time_c_file_path))
  tc_mat = time_c_file['time_courses']
  tm_cids = np.squeeze(time_c_file['cids'])

  # choose cells of interest
  tc_file_choose_idx = np.array([])
  for icell in np.array(cids_select):
    idx = np.where(tm_cids == icell)
    tc_file_choose_idx = np.append(tc_file_choose_idx, idx[0])
  tc_select = tc_mat[tc_file_choose_idx.astype('int'),:,:]
  tc_mean = np.squeeze(np.mean(tc_select,axis=0))
  n_cells = cids_select.shape[0]
  FLAGS.n_cells = n_cells

  #  'response', cell ids are 'cids_select' with 'n_cells' cells, 'tc_select' are timecourses, 'tc_mean' for mean time course
  return response, cids_select, n_cells, tc_select, tc_mean


def get_windows():
    # use FLAGS to get convolutional 'windows' for convolutional models.
    window = FLAGS.window
    n_channels = FLAGS.su_channels
    n_pix = ((2* window + 1) ** 2)*n_channels  # number of pixels in the window
    w_mask = np.zeros((2 * window + 1, 2 * window + 1, n_channels, n_pix))
    icnt = 0

    # make mask_tf: weight (dimx X dimy X npix) for convolutional layer,
    # where each layer is 1 for a particular pixel in window and 0 for others.
    # this is used for flattening the pixels in a window, so that different weights could be applied to each window
    for ichannel in range(n_channels):
        for ix in range(2 * window + 1):
            for iy in range(2 * window + 1):
                w_mask[ix, iy, ichannel, icnt] =1
                icnt = icnt + 1
            
    mask_tf = tf.constant(np.array(w_mask, dtype='float32'))
     # number of windows in x and y dimensions
    dimx = np.floor(1 + ((640 - (2 * window + 1))/FLAGS.stride)).astype('int')
    dimy = np.floor(1 + ((320 - (2 * window + 1))/FLAGS.stride)).astype('int')
    return mask_tf, dimx, dimy, n_pix


def main(argv):

  # initialize training and testing chunks
  init_chunks()

   # setup dataset
  _,cids_select, n_cells, tc_select, tc_mean = setup_dataset()

  # print parameters
  print('Save folder name: ' + str(FLAGS.folder_name) +
        '\nmodel:' + str(FLAGS.model_id) +
        '\nLoss:' + str(FLAGS.loss) +
        '\nbatch size' + str(FLAGS.batchsz) +
        '\nstep size' + str(FLAGS.step_sz) +
        '\ntraining length: ' + str(FLAGS.train_len) +
        '\nn_cells: '+str(n_cells))

  # filename for saving file
  short_filename = ('_loss='+
                    str(FLAGS.loss) + '_batch_sz='+ str(FLAGS.batchsz) +
                    '_step_sz'+ str(FLAGS.step_sz) +
                    '_tlen=' + str(FLAGS.train_len) + '_jitter')

  # setup model
  with tf.Session() as sess:
    # initialize stuff
    if FLAGS.loss == 'poisson':
      b_init = np.array(0.000001*np.ones(n_cells)) # a very small positive bias needed to avoid log(0) in poisson loss
    else:
      b_init =  np.log((tot_spks_chosen_cells)/(216000. - tot_spks_chosen_cells)) # log-odds, a good initialization for some

    # RGB time filter
    tm4D = np.zeros((30,1,3,3))
    for ichannel in range(3):
      tm4D[:,0,ichannel,ichannel] = tc_mean[:,ichannel]
    tc = tf.Variable((tm4D).astype('float32'),name = 'tc')

    d1=640
    d2=320
    colors=3

    # make data placeholders
    stim = tf.placeholder(tf.float32,shape=[None,d1,d2,colors],name='stim')
    resp = tf.placeholder(tf.float32,shape=[None,n_cells],name='resp')
    data_len = tf.placeholder(tf.float32,name='data_len')

    # time convolution
    # time course should be time,d1,color,color
    # original stimulus is (time, d1,d2,color). Permute it to (d2,time,d1,color) so that 1D convolution could be mimicked using conv_2d.
    stim_time_filtered = tf.transpose(tf.nn.conv2d(tf.transpose(stim,(2,0,1,3)),tc, strides=[1,1,1,1], padding='VALID'), (1,2,0,3))

    # learn almost convolutional model
    short_filename = ('model=' + str(FLAGS.model_id) + short_filename)
    mask_tf, dimx, dimy, n_pix = get_windows()
    w_del = tf.Variable(np.array( 0.05*np.random.randn(dimx, dimy, n_pix),dtype='float32'), name='w_del')
    w_mother = tf.Variable(np.array( np.ones((2 * FLAGS.window + 1, 2 * FLAGS.window + 1, FLAGS.su_channels, 1)),dtype='float32'), name='w_mother')
    a = tf.Variable(np.array(np.random.randn(dimx*dimy, n_cells),dtype='float32'), name='a')
    a_sfm = tf.transpose(tf.nn.softmax(tf.transpose(a)))
    b = tf.Variable(np.array(b_init,dtype='float32'), name='b')
    vars_fit = [w_mother, w_del, a] # which variables to fit
    if not FLAGS.loss == 'poisson':
      vars_fit = vars_fit + [b]

    # stimulus filtered with convolutional windows
    stim4D = stim_time_filtered#tf.expand_dims(tf.reshape(stim, (-1,40,80)), 3)
    stim_convolved = tf.reduce_sum(tf.nn.conv2d(stim4D, w_mother, strides=[1, FLAGS.stride, FLAGS.stride, 1], padding="VALID"),3)
    stim_masked = tf.nn.conv2d(stim4D, mask_tf, strides=[1, FLAGS.stride, FLAGS.stride, 1], padding="VALID" )
    stim_del = tf.reduce_sum(tf.mul(stim_masked, w_del), 3)

    # activation of different subunits
    su_act = tf.nn.relu(stim_del + stim_convolved)

    # get firing rate
    lam = tf.matmul(tf.reshape(su_act, [-1, dimx*dimy]), a_sfm) + b

    # regularization
    regularization = FLAGS.lam_w * tf.reduce_sum(tf.nn.l2_loss(w_del))

    # projection to satisfy hard variable constraints
    b_pos = tf.assign(b, (b + tf.abs(b))/2)
    def proj():
      if FLAGS.loss == 'poisson':
        sess.run(b_pos)


    if FLAGS.loss == 'poisson':
      loss_inter = (tf.reduce_sum(lam)/120. - tf.reduce_sum(resp*tf.log(lam))) / data_len

    loss = loss_inter + regularization # add regularization to get final loss function

    # training consists of calling training()
    # which performs a train step and project parameters to model specific constraints using proj()
    train_step = tf.train.AdagradOptimizer(FLAGS.step_sz).minimize(loss, var_list=vars_fit)
    def training(inp_dict):
      sess.run(train_step, feed_dict=inp_dict) # one step of gradient descent
      proj() # model specific projection operations

    # evaluate loss on given data.
    def get_loss(inp_dict):
      ls = sess.run(loss,feed_dict = inp_dict)
      return ls


    # saving details
    # make a folder with name derived from parameters of the algorithm - it saves checkpoint files and summaries used in tensorboard
    parent_folder = FLAGS.save_location + FLAGS.folder_name + '/'
    FLAGS.save_location = parent_folder + short_filename + '/'
    save_filename = FLAGS.save_location + short_filename


    # create summary writers
    # create histogram summary for all parameters which are learnt
    for ivar in vars_fit:
      tf.histogram_summary(ivar.name, ivar)
    # loss summary
    l_summary = tf.scalar_summary('loss',loss)
    # loss without regularization summary
    l_inter_summary = tf.scalar_summary('loss_inter',loss_inter)
    # Merge all the summary writer ops into one op (this way, calling one op stores all summaries)
    merged = tf.merge_all_summaries()
    # training and testing has separate summary writers
    train_writer = tf.train.SummaryWriter(FLAGS.save_location + 'train',
                                      sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.save_location + 'test')


    ## load previous results
    sess.run(tf.initialize_all_variables())
    saver_var = tf.train.Saver(tf.all_variables(), keep_checkpoint_every_n_hours=0.05)
    load_prev = False
    start_iter=0
    try:
      # restore previous fits if they are available - useful when programs are preempted frequently on.
      latest_filename = short_filename + '_latest_fn'
      restore_file = tf.train.latest_checkpoint(FLAGS.save_location, latest_filename)
      start_iter = int(restore_file.split('/')[-1].split('-')[-1]) # restore previous iteration count and start from there.
      saver_var.restore(sess, restore_file) # restore variables
      load_prev = True
      print('Previous dataset loaded')
    except:
      print('No previous dataset')

    # plot w_mother
    w_mot_eval = sess.run(w_mother)
    plt.figure()
    for idim in range(3):
      plt.subplot(1,3,idim+1)
      plt.imshow(np.squeeze(w_mot_eval[:,:,idim,0]),cmap='gray')
    plt.title('mother cell')
    plt.show()
    plt.draw()


if __name__ == '__main__':
  app.run()

