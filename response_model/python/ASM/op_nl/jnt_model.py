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
"""Jointly fit subunits and output NL."""

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
from absl import gfile
import os.path

import tensorflow as tf

def Flat_clustering_jnt(X, Y, Ns, tms_tr, tms_tst, K=None, b=None, steps_max=10000, eps=1e-6):

    # X is Txmask
    X_tr = X[tms_tr,:]
    Y_tr = Y[tms_tr, :]
    X_test = X[tms_tst,:]
    Y_test = Y[tms_tst, :]

    Tlen = Y_tr.shape[0]
    times = np.arange(Tlen)
    N1 = X_tr.shape[1]
    n_cells = Y.shape[1]
    Sigma = numpy.dot(X_tr.transpose(),X_tr)/float(X_tr.shape[0])

    # initialize filters
    if K is None:
        K = 2*rng.rand(N1,Ns)-0.5
    if b is None:
        b = 2*rng.rand(Ns, n_cells)-0.5

    def compute_fr_loss(K, b, X_in, Y_in):
      '''
      K : # n_pix x #SU
      b : # SU x # cells
      X_in : T x # pixels
      Y_in : T x # cells

      '''
      f = np.exp(np.expand_dims(np.dot(X_in, K), 2) + b) # T x SU x Cells
      fsum = f.sum(1) # T x # cells
      loss = np.mean(fsum, 0) - np.mean(Y_in * np.log(fsum), 0) # cells
      return fsum, loss

    # Find subunits - no output NL
    lam_log = np.zeros((0, n_cells))
    lam_log_test = np.zeros((0, n_cells))
    lam = np.inf
    lam_test = np.inf
    fitting_phase = np.array([])
    for irepeat in range(np.int(steps_max)):

        # test data
        _, lam_test = compute_fr_loss(K, b, X_test, Y_test)
        lam_log_test = np.append(lam_log_test, np.expand_dims(lam_test, 0), 0)

        # train data
        lam_prev = np.copy(lam)
        _, lam = compute_fr_loss(K, b, X_test, Y_test)
        lam_log = np.append(lam_log, np.expand_dims(lam, 0), 0)

        #print(itime)
        K_new_list_nr = []
        K_new_list_dr = []
        mean_ass_f_list = []
        for icell in range(n_cells):
          tms = np.int64(np.arange(Tlen))
          t_sp = tms[Y_tr[:, icell] != 0]
          Y_tsp = Y_tr[t_sp, icell]

          f = np.exp(numpy.dot(X_tr, K) + b[:, icell])
          alpha = (f.transpose()/f.sum(1)).transpose()
          xx = (Y_tsp.transpose()*alpha[t_sp, :].T).T
          sta_f = X_tr[t_sp,:].transpose().dot(xx)
          mean_ass_f = xx.sum(0)

          K_new_list_nr += [numpy.linalg.solve(Sigma,sta_f)]
          K_new_list_dr += [mean_ass_f]
          mean_ass_f_list += [mean_ass_f]

        K_new_list_nr = np.array(K_new_list_nr)
        K_new_list_dr = np.array(K_new_list_dr)
        mean_ass_f_list = np.array(mean_ass_f_list).T # recompute ??

        K = np.mean(K_new_list_nr, 0) / np.mean(K_new_list_dr, 0)
        b = np.log((1/Tlen)*mean_ass_f_list)- np.expand_dims(np.diag(0.5*K.transpose().dot(Sigma.dot(K))), 1)

        # print(irepeat, lam, lam_prev)
        if np.sum(np.abs(lam_prev - lam)) < eps:
            #print('Subunits fitted, Train loss: %.7f, '
            #      'Test loss: %.7f after %d iterations' % (lam, lam_test, irepeat))
            break

    fitting_phase = np.append(fitting_phase, np.ones(lam_log.shape[0]))
    nl_params = np.repeat(np.expand_dims(np.array([1.0, 0.0]), 1), n_cells, 1)
    fit_params = [[np.copy(K), np.copy(b), nl_params ]]

    # fit NL + b + Kscale
    K, b, nl_params, loss_log, loss_log_test  = fit_scales(X_tr, Y_tr, X_test, Y_test,
                                                     Ns=Ns, K=K, b=b,
                                                     params=nl_params,
                                                     lr=0.001, eps=eps)

    lam_log = np.append(lam_log, np.array(loss_log), 0)
    lam_log_test = np.append(lam_log_test, np.array(loss_log_test), 0)
    fitting_phase = np.append(fitting_phase, 2 * np.ones(np.array(loss_log).shape[0]))
    fit_params += [[np.copy(K), np.copy(b), nl_params]]

    # Fit all params
    K, b, nl_params, loss_log, loss_log_test  = fit_all(X_tr, Y_tr, X_test, Y_test,
                                                     Ns=Ns, K=K, b=b, train_phase=3,
                                                     params=nl_params,
                                                     lr=0.001, eps=eps)
    lam_log = np.append(lam_log, np.array(loss_log), 0)
    lam_log_test = np.append(lam_log_test, np.array(loss_log_test), 0)
    fitting_phase = np.append(fitting_phase, 3 * np.ones(np.array(loss_log).shape[0]))
    fit_params += [[np.copy(K), np.copy(b), nl_params]]

    return K, b, alpha, lam_log, lam_log_test, fitting_phase, fit_params


def fit_all(X_tr, Y_tr, X_test, Y_test,
                   Ns=5, K=None, b=None, params=None, train_phase=2, lr=0.1, eps=1e-9):

    X = tf.placeholder(tf.float32)  # T x Nsub
    Y = tf.placeholder(tf.float32)  # T

    # initialize filters
    if K is None or b is None or params is None:
        raise "Not initialized"

    K_tf = tf.Variable(K.astype(np.float32))
    b_tf = tf.Variable(b.astype(np.float32))
    params_tf = tf.Variable(np.array(params).astype(np.float32))

    lam_int = tf.reduce_sum(tf.exp(tf.expand_dims(tf.matmul(X, K_tf), 2) + b_tf), 1) # T x # cells
    # lam = params_tf[0]*lam_int / (params_tf[1]*lam_int + 1)
    lam = tf.pow(lam_int, params_tf[0, :])/ (params_tf[1, :] * lam_int + 1) # T x # cells
    loss = tf.reduce_mean(lam, 0) - tf.reduce_mean(Y * tf.log(lam), 0)
    loss_all_cells = tf.reduce_sum(loss)

    if train_phase == 2:
        train_op = tf.train.AdamOptimizer(lr).minimize(loss_all_cells, var_list=[b_tf, params_tf])
    if train_phase == 3:
        train_op = tf.train.AdamOptimizer(lr).minimize(loss_all_cells, var_list=[K_tf, b_tf, params_tf])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        l_tr_log = []
        l_test_log = []
        l_tr_prev = np.inf
        for iiter in range(10000):
                l_tr, _ = sess.run([loss, train_op], feed_dict={X: X_tr, Y: Y_tr})
                l_test = sess.run(loss, feed_dict={X: X_test, Y: Y_test})

                l_tr_log += [l_tr]
                l_test_log += [l_test]

                #print(iiter, l_tr)
                if np.sum(np.abs(l_tr_prev - l_tr)) < eps:
                    # print('Nonlinearity fit after : %d iters, Train loss: %.7f' % (iiter, l_tr))
                    break
                l_tr_prev = l_tr

        return sess.run(K_tf), sess.run(b_tf), sess.run(params_tf), l_tr_log, l_test_log



def fit_scales(X_tr, Y_tr, X_test, Y_test,
                   Ns=5, K=None, b=None, params=None, lr=0.1, eps=1e-9):

    X = tf.placeholder(tf.float32)  # T x Nsub
    Y = tf.placeholder(tf.float32)  # T x n_cells

    # initialize filters
    if K is None or b is None or params is None:
        raise "Not initialized"

    K_tf_unscaled = tf.constant(K.astype(np.float32))
    K_scale = tf.Variable(np.ones((1, K.shape[1])).astype(np.float32))

    K_tf = tf.multiply(K_tf_unscaled, K_scale)
    b_tf = tf.Variable(b.astype(np.float32))
    params_tf = tf.Variable(np.array(params).astype(np.float32)) # 2 x # cells

    lam_int = tf.reduce_sum(tf.exp(tf.expand_dims(tf.matmul(X, K_tf), 2) + b_tf), 1) # T x # cells
    # lam = params_tf[0]*lam_int / (params_tf[1]*lam_int + 1)
    lam = tf.pow(lam_int, params_tf[0, :])/ (params_tf[1, :] * lam_int + 1) # T x # cells
    loss = tf.reduce_mean(lam, 0) - tf.reduce_mean(Y * tf.log(lam), 0)
    loss_all_cells = tf.reduce_sum(loss)

    train_op = tf.train.AdamOptimizer(lr).minimize(loss_all_cells, var_list=[K_scale, b_tf, params_tf])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        l_tr_log = []
        l_test_log = []
        l_tr_prev = np.inf
        for iiter in range(10000):
                l_tr, _ = sess.run([loss, train_op], feed_dict={X: X_tr, Y: Y_tr})
                l_test = sess.run(loss, feed_dict={X: X_test, Y: Y_test})

                l_tr_log += [l_tr]
                l_test_log += [l_test]
                # from IPython import embed; embed()
                # print(iiter, l_tr)
                if np.sum(np.abs(l_tr_prev - l_tr)) < eps:
                    # print('Nonlinearity fit after : %d iters, Train loss: %.7f' % (iiter, l_tr))
                    break
                l_tr_prev = l_tr
        return sess.run(K_tf), sess.run(b_tf), sess.run(params_tf), l_tr_log, l_test_log


def Flat_clustering(X, Y, Ns, tms_tr, tms_tst, batches=1, K=None, b=None, steps_max=10000, eps=1e-6):

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

    # initialize filters
    if K is None:
        K = 2*rng.rand(N1,Ns)-0.5
    if b is None:
        b = 2*rng.rand(Ns)-0.5

    # Find subunits - no output NL
    lam_log=np.array([])
    lam_log_test = np.array([])
    lam = np.inf
    lam_test = np.inf
    fitting_phase = np.array([])
    for irepeat in range(np.int(steps_max/nBatch)):
        #times=np.random.permutation(np.arange(Tlen))
        times = np.arange(Tlen)
        #print(irepeat)
        ibatch = 0

        # test data
        ftst = np.exp(numpy.dot(X_test,K)+b)
        fsumtst = ftst.sum(1)
        lam_test =  (numpy.sum(fsumtst) - numpy.dot(Y_test.transpose(),numpy.log(fsumtst)))/float(Y_test.shape[0])
        lam_log_test=np.append(lam_log_test,lam_test)

        # train data
        lam_prev = lam
        f = np.exp(numpy.dot(X_tr,K)+b)
        fsum = f.sum(1)
        lam = (numpy.sum(fsum) - numpy.dot(Y_tr.transpose(),numpy.log(fsum)))/float(Y_tr.shape[0])
        lam_log = np.append(lam_log,lam)

        # batch training
        NN=BatchSz
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
        lam = (numpy.sum(fsum) - numpy.dot(Yi.transpose(),numpy.log(fsum)))/float(Yi.shape[0])
        alpha = (f.transpose()/f.sum(1)).transpose()
        xx = (Y_tsp.transpose()*alpha[t_sp,:].transpose()).transpose()
        sta_f = Xi[t_sp,:].transpose().dot(xx)
        mean_ass_f = xx.sum(0)

        K = numpy.linalg.solve(Sigma,sta_f)/mean_ass_f
        b = numpy.log((1/NN)*mean_ass_f)-np.diag(0.5*K.transpose().dot(Sigma.dot(K)))

        #print(irepeat, ibatch, lam_test, lam)
        if np.abs(lam_prev - lam) < eps:
            print('Subunits fitted, Train loss: %.7f, '
                  'Test loss: %.7f after %d iterations' % (lam, lam_test, irepeat))
            break

    fitting_phase = np.append(fitting_phase, np.ones(lam_log.shape[0]))
    fit_params = [[np.copy(K), np.copy(b), [1.0, 0.0]]]


    return K, b, alpha, lam_log, lam_log_test, fitting_phase, fit_params



def Flat_clustering_jnt_pop(X, Y, Ns, tms_tr, tms_tst, batches=1,
                            K=None, b=None, eps=0.01, lam_l1=0.1,
                            steps_max=10000):

  # X is Txmask
  X_tr = X[tms_tr,:]
  Y_tr = Y[tms_tr, :]
  X_test = X[tms_tst,:]
  Y_test = Y[tms_tst, :]

  Tlen = Y_tr.shape[0]
  N1 = X_tr.shape[1]
  N_cell = Y.shape[1]
  Sigma = numpy.dot(X_tr.transpose(),X_tr)/float(X_tr.shape[0])

  nBatch = batches;
  BatchSz = np.int(np.floor(Tlen/nBatch))
  icnt=0;

  # initialize filters
  if K is None:
    #print('K initialized')
    K = 2*rng.rand(N1,Ns)-0.5

  if b is None:
    #print('b initialized')
    b = 2*rng.rand(Ns, N_cell)-0.5

  loss_log = []
  loss_log_test = []
  grad_K_log=[]
  loss = np.inf
  loss_test = np.inf
  for irepeat in range(np.int(steps_max/nBatch)):
    times = np.arange(Tlen)

    # test data
    ftst = np.exp(np.expand_dims(numpy.dot(X_test,K), 2) + np.expand_dims(b, 0)) # T x su x cell
    fsumtst = ftst.sum(1) # T x cell
    loss_test =  (numpy.sum(fsumtst, 0)/120. - numpy.sum(Y_test * numpy.log(fsumtst), 0))/float(Y_test.shape[0])
    loss_log_test += [[loss_test]]

    # train data
    loss_prev = loss
    f = np.exp(np.expand_dims(numpy.dot(X_tr,K), 2)+ np.expand_dims(b, 0)) # T x su x cell
    fsum = f.sum(1) # T x cell
    loss =  (numpy.sum(fsum, 0)/120. - numpy.sum(Y_tr * numpy.log(fsum), 0))/float(Y_tr.shape[0])
    loss_log += [[loss]]

    NN=BatchSz/120.
    icnt=icnt+1
    Xi = X_tr[times,:]
    K_new_list_nr = []
    K_new_list_dr = []
    mean_ass_f_list = []
    for icell in range(N_cell):
      Yi = Y_tr[times, icell]
      tms = np.int64(np.arange(BatchSz))
      t_sp = tms[Yi!=0]
      Y_tsp=Yi[t_sp]

      f = np.exp(Xi.dot(K) + b[:, icell])
      alpha = (f.transpose()/f.sum(1)).transpose()
      xx = (Y_tsp.transpose()*alpha[t_sp,:].transpose()).transpose()
      sta_f = Xi[t_sp,:].transpose().dot(xx)
      mean_ass_f = xx.sum(0)

      K_new_list_nr += [numpy.linalg.solve(Sigma,sta_f)]
      K_new_list_dr += [mean_ass_f]
      mean_ass_f_list += [ ]

    K_new_list_nr = np.array(K_new_list_nr)
    K_new_list_dr = np.array(K_new_list_dr)

    # update K
    K = np.mean(K_new_list_nr, 0) / np.mean(K_new_list_dr, 0)

    # recompute alpha
    mean_ass_f_list = []
    alpha_list = []
    for icell in range(N_cell):
      Yi = Y_tr[itime, icell]
      tms = np.int64(np.arange(BatchSz))
      t_sp = tms[Yi!=0]
      Y_tsp=Yi[t_sp]

      f = np.exp(Xi.dot(K) + b[:, icell])
      alpha = (f.transpose()/f.sum(1)).transpose()
      xx = (Y_tsp.transpose()*alpha[t_sp,:].transpose()).transpose()
      sta_f = Xi[t_sp,:].transpose().dot(xx)
      mean_ass_f = xx.sum(0)

      mean_ass_f_list += [mean_ass_f]
      alpha_list += [alpha]

    mean_ass_f_list = np.array(mean_ass_f_list).T

    b= (numpy.log((1/NN)*mean_ass_f_list) -
        np.expand_dims(np.diag(0.5*K.transpose().dot(Sigma.dot(K))), 1))

    #print(np.exp(b))

    if np.abs(np.sum(loss) - np.sum(loss_prev)) < eps_loss:
      print('Loss %.5f' % np.sum(loss))
      break

  fitting_phase = np.append(fitting_phase, np.ones(lam_log.shape[0]))
  fit_params = [[np.copy(K), np.copy(b), [1.0, 0.0]]]

  from IPython import embed; embed()
  # fit NL + b + Kscale
  K, b, params, l_log, l_log_test  = fit_scales(X_tr, Y_tr, X_test, Y_test,
                                                Ns=Ns, K=K, b=b, params=[1.0, 0.0],
                                                lr=0.001, eps=eps)

  loss_log = np.append(loss_log, l_log)
  loss_log_test = np.append(loss_log_test, l_log_test)
  fitting_phase = np.append(fitting_phase, 2 * np.ones(np.array(l_log).shape[0]))
  fit_params += [[np.copy(K), np.copy(b), params]]

  # Fit all params
  K, b, params, l_log, l_log_test  = fit_all(X_tr, Y_tr, X_test, Y_test,
                                             Ns=Ns, K=K, b=b,train_phase=3,
                                             params=params,
                                             lr=0.001, eps=eps)
  loss_log = np.append(loss_log, l_log)
  loss_log_test = np.append(loss_log_test, l_log_test)
  fitting_phase = np.append(fitting_phase, 3 * np.ones(np.array(l_log).shape[0]))
  fit_params += [[np.copy(K), np.copy(b), params]]

  return K, b, alpha_list, loss_log, loss_log_test, fitting_phase, fit_params
