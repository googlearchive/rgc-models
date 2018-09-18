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
"""Sparse model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags


# Import module
import scipy as sp
import numpy as np , h5py,numpy
import matplotlib.pyplot as plt
import matplotlib
import time
rng = np.random
import pickle
import copy
from tensorflow.python.platform import gfile
import os.path
FLAGS = flags.FLAGS


def get_neighbormat(mask_matrix, nbd=1):
  mask = np.ndarray.flatten(mask_matrix)>0

  x = np.repeat(np.expand_dims(np.arange(mask_matrix.shape[0]), 1), mask_matrix.shape[1], 1)
  y = np.repeat(np.expand_dims(np.arange(mask_matrix.shape[1]), 0), mask_matrix.shape[0], 0)
  x = np.ndarray.flatten(x)
  y = np.ndarray.flatten(y)

  idx = np.arange(len(mask))
  iidx = idx[mask]
  xx = np.expand_dims(x[mask], 1)
  yy = np.expand_dims(y[mask], 1)

  distance = (xx - xx.T)**2 + (yy - yy.T)**2
  neighbor_mat = np.double(distance <= nbd)

  return neighbor_mat


def Flat_clustering_sparse2(X, Y, Ns, tms_tr, tms_tst, batches=1, K=None, b=None,
                            lam_l1=1, eps=0.01, neighbor_mat=None,
                            stop_th=1e-9, max_iter=10000):

  # X is Txmask
  X_tr = X[tms_tr,:]
  Y_tr = Y[tms_tr]
  X_test = X[tms_tst,:]
  Y_test = Y[tms_tst]

  Tlen = Y_tr.shape[0]
  N1 = X_tr.shape[1]
  Sigma = numpy.dot(X_tr.transpose(),X_tr)/float(X_tr.shape[0])


  nBatch = batches;
  BatchSz = np.int(np.floor(Tlen/nBatch))
  icnt=0;

  if neighbor_mat is None:
      neighbor_mat = np.eye(N1)

  # initialize filters
  if K is None:
      K = 2*rng.rand(N1,Ns)-0.5
  if b is None:
      b = 2*rng.rand(Ns)-0.5

  lam_log=np.array([])
  lam_log_test = np.array([])
  grad_K_log=[]


  '''
  # better stopping criteria. If error does not improve for N repeats, then stop.
  continuous_reps = 4
  continous_eps = 1e-5
  lam_prev_repeat = np.inf
  iters_without_improvement = 0
  '''
  lam = np.inf
  for irepeat in range(np.int(max_iter/nBatch)):
    times = np.arange(Tlen)
    ibatch = 0

    # compute reweighted L1 weights
    wts = 1/(neighbor_mat.dot(np.abs(K)) + eps)

    # test data
    ftst = np.exp(numpy.dot(X_test,K)+b)
    fsumtst = ftst.sum(1)
    lam_test =  (numpy.sum(fsumtst)/120. - numpy.dot(Y_test.transpose(),numpy.log(fsumtst)))/float(Y_test.shape[0])
    lam_log_test=np.append(lam_log_test,lam_test)

    # train data
    lam_prev = lam
    f = np.exp(numpy.dot(X_tr,K)+b)
    fsum = f.sum(1)
    lam = (numpy.sum(fsum)/120. - numpy.dot(Y_tr.transpose(),numpy.log(fsum)))/float(Y_tr.shape[0])
    lam_log = np.append(lam_log, lam)
    # print('lam' , lam, 'lam_prev', lam_prev)
    if np.abs(lam - lam_prev) < stop_th:
      print('Stopping after %d iterations ' % irepeat)
      break

    # batch training
    NN=BatchSz/120.
    itime = times[np.arange(BatchSz) + ibatch*BatchSz]
    #print(itime)
    icnt=icnt+1
    Xi = X_tr[itime,:]
    Yi = Y_tr[itime]
    tms = np.int64(np.arange(BatchSz))
    t_sp = tms[Yi!=0]
    Y_tsp=Yi[t_sp]

    f = np.exp(numpy.dot(Xi,K)+b)
    fsum = f.sum(1)
    lam = (numpy.sum(fsum)/120. - numpy.dot(Yi.transpose(),numpy.log(fsum)))/float(Yi.shape[0])
    alpha = (f.transpose()/f.sum(1)).transpose()
    xx = (Y_tsp.transpose()*alpha[t_sp,:].transpose()).transpose()
    sta_f = Xi[t_sp,:].transpose().dot(xx)
    mean_ass_f = xx.sum(0)
    ''' below part is just to compute the gradient'''
    exponent = np.exp(np.diag(0.5*K.transpose().dot(Sigma.dot(K))) + b)
    gradK = (Sigma.dot(K)*exponent/120. - (1/float(Yi.shape[0]))*sta_f)
    grad_K_log.insert(len(grad_K_log),[(np.sum(gradK**2,0))])

    # update K
    K = numpy.linalg.solve(Sigma,sta_f)/mean_ass_f

    # Soft thresholding for K
    K = np.maximum(K - (wts*lam_l1), 0) - np.maximum(- K - (wts*lam_l1), 0)
    # update b
    b= numpy.log((1/NN)*mean_ass_f)-np.diag(0.5*K.transpose().dot(Sigma.dot(K)))


    ''''
        #print(irepeat, ibatch, lam_test, lam)
    if np.abs(lam_prev_repeat - lam) < continous_eps:
        iters_without_improvement = iters_without_improvement + 1
    else:
        iters_without_improvement = 0

    lam_prev_repeat = lam

    if iters_without_improvement == continuous_reps:
        print('Encountered %d repeats without a decrease in training loss.'
              '\n Quitting after %d passes over training data.' %(iters_without_improvement, irepeat))
        break;
    '''
  return K,b,alpha,lam_log,lam_log_test,grad_K_log
