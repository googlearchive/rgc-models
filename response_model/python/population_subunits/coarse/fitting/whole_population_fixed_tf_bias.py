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
"""Population model, fixed bias.
"""
import math
import os.path
import subprocess
import sys

import tensorflow as tf
from absl import app
from absl import flags
from absl import gfile

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pylab
import matplotlib.pyplot as plt
from PIL import Image

import cPickle
import numpy as np,h5py
import scipy
import string
import scipy
import scipy.io as sio
from scipy import ndimage
import timeit
import random

FLAGS = flags.FLAGS
flags.DEFINE_float("lam_W", 0.0001,"sparsitiy regularization of W")
flags.DEFINE_float("lam_a", 0.0001,"sparsitiy regularization of a")
flags.DEFINE_integer("ratio_SU",2,"ratio of subunits/cells")
flags.DEFINE_float("su_grid_spacing",5.7,"grid spacing")
flags.DEFINE_integer("np_randseed",23,"numpy RNG seed")
flags.DEFINE_integer("randseed",65,"python RNG seed")
flags.DEFINE_float("bias_ratio",-2,"bias initialization compared to standard deviation")
flags.DEFINE_string("save_location",'/home/bhaishahster/Downloads/','where to store logs and outputs?');

flags.DEFINE_string("data_location",'/home/bhaishahster/Downloads/','where to take data from?')

def plot_Weights(W):
    #plt.figure()
    ns  = W.shape[1]
    for i in range(ns):
        WW=np.zeros(3200)
        WW =W[:,i]
        a=np.reshape(WW,[40,80])
        #ax=plt.subplot(1,4,i+1)
        #plt.imshow(a, cmap='gray',  interpolation='nearest')
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
    #plt.show()
    #plt.draw()

def hex_grid (gridX,d,n):
  x_log=np.array([])
  y_log=np.array([])
  for i in range(n):
    x_log = np.append(x_log ,(((i*d)%gridX) + (np.floor(i*d/gridX)%2)*d/2)) + np.random.randn(1)*0.01
    y_log = np.append(y_log,np.floor((i*d/gridX))*d/2) + np.random.randn(1)*0.01

  #plt.figure()
  #plt.plot(x_log,y_log,'.')
  #plt.show()
  #plt.draw()
  return x_log,y_log


def gaussSU(x_log,y_log,gridX=80,gridY=40):
  ns = x_log.shape[0]
  wts = np.zeros((3200,ns))
  for isu in range(ns):
    xx = np.zeros((gridY,gridX))

    if((np.round(y_log[isu]) >= gridY) | (np.round(y_log[isu]) < 0) | (np.round(x_log[isu])>=gridX) | (np.round(x_log[isu])<0)):
      continue

    xx[np.round(y_log[isu]),np.round(x_log[isu])]=1
    blurred_xx = ndimage.gaussian_filter(xx, sigma=2)
    #plt.imshow(blurred_xx)
    wts[:,isu] = np.ndarray.flatten(blurred_xx)
  return wts

def initialize_SU(nSU=107*2,gridX=80,gridY=40,spacing=5.7):
  spacing = FLAGS.su_grid_spacing
  x_log,y_log = hex_grid(gridX,spacing,nSU)
  wts = gaussSU(x_log,y_log)
  return wts

def main(argv):
  logfile = gfile.Open(FLAGS.save_location+'log_bias='+str(FLAGS.bias_ratio)+'_lam_W='+str(FLAGS.lam_W)+'_lam_a='+str(FLAGS.lam_a)+'.txt',"w")
  logfile.write('Starting new thread\n')
  logfile.flush()
  print('\nlog file written once')

  np.random.seed(FLAGS.np_randseed)
  random.seed(FLAGS.randseed)

  #plt.ion()
  ## Load data
  file=h5py.File(FLAGS.data_location+'Off_parasol.mat','r')
  logfile.write('\ndataset loaded')
  # Load Masked movie
  data = file.get('maskedMovdd')
  maskedMov=np.array(data)
  cells = file.get('cells')
  nCells = cells.shape[0]
  ttf_log = file.get('ttf_log')
  ttf_avg = file.get('ttf_avg')
  stimulus=maskedMov
  total_mask_log=file.get('totalMaskAccept_log');

  # Load spike Response of cells
  data = file.get('Y')
  biSpkResp_coll=np.array(data,dtype='float32')
  mask = np.array(np.ones(3200),dtype=bool)

  ##
  Nsub=FLAGS.ratio_SU*nCells
  StimDim = maskedMov.shape[1]

  # initialize subunits
  W_init = initialize_SU(nSU=Nsub)
  a_init = np.random.rand(Nsub,nCells)

  su_act = stimulus[10000:12000,:].dot(W_init)
  su_std = np.sqrt(np.diag(su_act.T.dot(su_act))/stimulus.shape[0])

  bias_init =FLAGS.bias_ratio*su_std
  print(bias_init)
  logfile.write('bias = '+ str(bias_init))
  logfile.write('\nSU initialized')

  with tf.Session() as sess:
    stim= tf.placeholder(tf.float32,shape=[None,StimDim],name="stim")
    resp = tf.placeholder(tf.float32,name="resp")
    data_len = tf.placeholder(tf.float32,name="data_len")

    #W = tf.Variable(tf.random_uniform([StimDim,Nsub]))
    W = tf.Variable(np.array(W_init,dtype='float32'))
    a = tf.Variable(np.array(a_init,dtype='float32'))
    bias = tf.Variable(np.array(bias_init,dtype='float32'))
    #a = tf.Variable(np.identity(Nsub,dtype='float32')*0.01)
    lam = tf.matmul(tf.nn.relu(tf.matmul(stim,W) + bias),tf.nn.relu(a)) + 0.0001 # collapse a dimension
    loss = (tf.reduce_sum(lam)/120. - tf.reduce_sum(resp*tf.log(lam)))/data_len  +  FLAGS.lam_W*tf.reduce_sum(tf.abs(W)) + FLAGS.lam_a*tf.reduce_sum(tf.abs(a))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss,var_list=[W,a,bias])

    sess.run(tf.initialize_all_variables())

    # Do the fitting
    batchSz = 100
    icnt=0

    fd_test = {stim :stimulus.astype('float32')[216000-10000:216000-1,:] , resp :biSpkResp_coll.astype('float32')[216000-10000:216000-1,:],data_len:10000}
    ls_train_log = np.array([])
    ls_test_log=np.array([])
    tms = np.random.permutation(np.arange(216000-1000))
    for istep in range(100000):
      time_start = timeit.timeit()
      fd_train = {stim :stimulus.astype('float32')[tms[icnt:icnt+batchSz],:] , resp : biSpkResp_coll.astype('float32')[tms[icnt:icnt+batchSz],:],data_len:batchSz}
      sess.run(train_step,feed_dict=fd_train)
      if istep%10==0:
        ls_train = sess.run(loss,feed_dict=fd_train)
        ls_test = sess.run(loss,feed_dict=fd_test)
        ls_train_log = np.append(ls_train_log,ls_train)
        ls_test_log = np.append(ls_test_log,ls_test)
        logfile.write('\nIterations: '+ str(istep)+' Training error: '+str(ls_train)+' Testing error: '+str(ls_test));
        logfile.flush()
        sio.savemat(FLAGS.save_location+'data_bias='+str(FLAGS.bias_ratio)+'_lam_W='+str(FLAGS.lam_W)+'_lam_a'+str(FLAGS.lam_a)+'_ratioSU'+str(FLAGS.ratio_SU)+'_grid_spacing_'+str(FLAGS.su_grid_spacing)+'.mat',{'bias_ratio':FLAGS.bias_ratio,'bias_init':bias_init,'bias':bias.eval(),'W':W.eval(),'a':a.eval(),'W_init':W_init,'a_init':a_init,'ls_train_log':ls_train_log,'ls_test_log':ls_test_log})

      icnt= icnt +batchSz
      if icnt >216000-10000:
        icnt =0
        tms = np.random.permutation(np.arange(216000-10000))

  logfile.close()
if __name__ == '__main__':
  app.run()

