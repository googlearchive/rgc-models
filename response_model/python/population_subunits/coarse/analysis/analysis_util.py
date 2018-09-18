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
"""Utils for analysis of fitted models."""

import sys
import os.path
import collections
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

from retina.response_model.python.population_subunits.coarse.fitting import data_utils
from retina.response_model.python.population_subunits.coarse.analysis import model_util


FLAGS = flags.FLAGS

def analyse_almost_convolutional_model(model, sess, data):

  # extract variables
  model_vars = model.variables
  a_np = sess.run(model_vars.a)
  w_mother_np = sess.run(model_vars.w_mother)
  w_del_np = sess.run(model_vars.w_del)
  bias_su_np = sess.run(model_vars.bias_su)
  bias_cell_np = sess.run(model_vars.bias_cell)
  try:
    scale_cell_np = sess.run(model_vars.scale_cell)
  except:
    pass
  stas = data.stas

  # extract softmax weights
  a_sfm = tf.transpose(tf.nn.softmax(tf.transpose(model_vars.a)))
  a_sfm_np = sess.run(a_sfm)

  from IPython.terminal.embed import InteractiveShellEmbed
  ipshell = InteractiveShellEmbed()
  ipshell()

  # plot all bias_cell_su
  plt.figure()
  for icell in range(100):
    print(icell)
    plt.subplot(10, 10, icell+1)
    plt.imshow(np.exp(np.reshape(a_sfm_np[:,icell], [model.params.dimx, model.params.dimy])), interpolation='nearest', cmap='gray')
    plt.xticks([])
    plt.yticks([])
  plt.show()



  # plot w_mother
  
  plt.figure()
  plot_weight(w_mother_np)

  # For a few cells, plot strongly connected subunits
  plt.figure()
  plot_su_for_cells(a_sfm_np, w_mother_np, w_del_np, data, model)
  plt.show()
  plt.draw()


  # plot spokes
  plt.figure()
  plot_su_gaussian_spokes(a_sfm_np, w_mother_np, w_del_np, model, stas, threshold=0.008)
  plt.show()
  plt.draw()

  from IPython.terminal.embed import InteractiveShellEmbed
  ipshell = InteractiveShellEmbed()
  ipshell()

  # correlation analysis


def analyse_almost_convolutional_softmax(model, sess, data):

  # start getting inputs
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

 # update bias only for almost_convolutional_mel_dropout_only_wdelta
  '''
  fdict = {model.stim: data.stimulus.astype(np.float32), model.resp: data.response.astype(np.float32)}
  for icell in range(model.params.n_cells):
    _, sel_cell_np = sess.run([model.probes.update_bias_only, model.probes.select_cells], feed_dict = fdict)
    tf.logging.info(sel_cell_np)
  '''
  # extract variables
  model_vars = model.variables
  w_mother_np = sess.run(model_vars.w_mother)
  w_del_np = sess.run(model_vars.w_del)
  bias_cell_su_np = sess.run(model_vars.bias_cell_su)
  stas = data.stas

  from IPython.terminal.embed import InteractiveShellEmbed
  ipshell = InteractiveShellEmbed()
  ipshell()

  # If stochastic convex approximation model
  bias_cell_su_np = np.expand_dims(np.transpose(bias_cell_su_np,[1,2,0]),0)

  #

  # plot all bias_cell_su
  plt.figure()
  for icell in range(100):
    print(icell)
    plt.subplot(10, 10, icell+1)
    plt.imshow(np.exp(bias_cell_su_np[0,:,:,icell]), interpolation='nearest', cmap='gray')
    plt.xticks([])
    plt.yticks([])
  plt.show()


  # plot all bias_cell_su
  plt.figure()
  for icell in range(12):
    print(icell)
    plt.subplot(4,3, icell+1)
    plt.plot(np.diff(np.sort(np.exp(np.ndarray.flatten(bias_cell_su_np[0,:,:,icell])))), 'b.')
    plt.xticks([])
    #plt.yticks([])
  plt.show()

  # plot firing prediction
  lam_norm_np, resp_np, su_act_sfm_np = sess.run([model.probes.lam_normalized, model.resp, model.probes.su_act_softmax])

  icell = 20
  plt.figure()
  plt.plot(lam_norm_np[:, icell], 'r')
  plt.hold(True)
  tms = np.arange(resp_np.shape[0])
  tms = tms[resp_np[:, icell] > 0]
  for itime in tms:
    plt.plot([itime, itime], [0, np.max(lam_norm_np[:, icell])], 'b')
  plt.show()


  # plot w_mother
  plot_weight(w_mother_np)

  # plot which subunits connected to which cells
  plot_su_wts(bias_cell_su_np)

  # For a few cells, plot strongly connected subunits
  # TODO(bhaishahster)
  plt.figure()
  plot_su_for_cells(np.exp(np.reshape(bias_cell_su_np,[-1, 107])), w_mother_np, w_del_np, data, model)
  plt.show()
  plt.draw()


  # plot spokes
  #b_c_s_flat = np.reshape(bias_cell_su_np, [-1, model.params.n_cells])
  #b_c_s_flat_sfm = np.exp(b_c_s_flat) / np.sum(np.exp(b_c_s_flat),0)
  b_c_s = np.exp(np.reshape(bias_cell_su_np, [-1, model.params.n_cells]))
  # after passing through exponential, threshold: 0.1
  # before passing through exp, threshold: 0.5
  #plot_su_gaussian_spokes(np.log(b_c_s), w_mother_np, w_del_np, model, stas, threshold=0.5)
  plot_su_gaussian_spokes(b_c_s, w_mother_np, w_del_np, model, stas, threshold=0.0005)

  # correlation analysis
  correlation_analysis(sess, data, model)

  # stop getting data
  coord.request_stop()

  # Wait for threads to finish.
  coord.join(threads)
  sess.close()


  # correlation analysis
def correlation_analysis(sess, data, model):

  from IPython.terminal.embed import InteractiveShellEmbed
  ipshell = InteractiveShellEmbed()
  ipshell()

  tlen = data.stimulus.shape[0]
  batch_sz = 5000
  n_batches = tlen/batch_sz
  spk_pred = np.zeros((tlen, model.params.n_cells))

  for ibatch in range(n_batches):
    print(ibatch)
    tms = np.arange(ibatch*batch_sz,(ibatch+1)*batch_sz)
    fdict = {model.stim: data.stimulus.astype(np.float32)[tms,:], model.resp: data.response.astype(np.float32)[tms,:]}
    lam = sess.run(model.probes.lam, feed_dict=fdict)
    spk_pred[tms, :] = np.random.poisson(lam)

  plt.plot(np.sum(data.response,0), np.sum(spk_pred, 0), '.')
  plt.show()

  def plot_xcorr(data, spk_pred, icell, jcell):
    xx1 = np.correlate(data.response[:, icell], data.response[:, jcell], mode='same');
    xx2 = np.correlate(spk_pred[:, icell], spk_pred[:, jcell], mode='same');
    tt = np.arange(-50,50)*1000/120
    plt.plot(tt, xx1[108000-50:108000+50], 'k');
    plt.hold(True)
    plt.plot(tt, xx2[108000-50:108000+50], 'r');
    plt.legend(['Recorded', 'Predicted'])
    plt.xlabel('time (ms)')



  icell = 97
  jcell = 94

  # auto correlation
  plt.subplot(1,3,1)
  plot_xcorr(data, spk_pred, icell, icell)

  plt.subplot(1,3,2)
  plot_xcorr(data, spk_pred, jcell, jcell)

  plt.subplot(1,3,3)
  plot_xcorr(data, spk_pred, icell, jcell)

  plt.suptitle('cells : %d, %d' %(icell, jcell))


  # plot recorded and predicted correlation at a particular time delay
  true_corr_log = np.array([])
  pred_corr_log = np.array([])
  delay=2
  for icell in range(model.params.n_cells):
    print(icell)
    for jcell in range(model.params.n_cells):
      if icell == jcell :
        continue

      true_corr = np.corrcoef(data.response[:-delay, icell], data.response[delay:, jcell])[0][1];
      pred_corr = np.corrcoef(spk_pred[:-delay, icell], spk_pred[delay:, jcell])[0][1];
      true_corr_log = np.append(true_corr_log, true_corr)
      pred_corr_log = np.append(pred_corr_log, pred_corr)

  plt.plot(true_corr_log, pred_corr_log, '.');
  plt.hold(True)
  plt.plot([-0.01, 0.14], [-0.01, 0.14], 'g');
  plt.xlim([-0.01, 0.14])
  plt.ylim([-0.01, 0.14])
  plt.xlabel('Recorded correlation');
  plt.ylabel('Predicted correlation');

  # shared subunits v/s 

def plot_su_wts(bias_cell_su_np):
  cell_list = [23, 52, 60, 10, 73, 100]
  ncell_plt = len(cell_list)
  for iicell, icell in enumerate(cell_list):
    fig = plt.subplot(ncell_plt, 2, iicell*2 + 1)
    plt.imshow(np.exp(bias_cell_su_np[0, :, :, icell]),interpolation='nearest', cmap='gray');
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title('exp(bcs)')

    fig = plt.subplot(ncell_plt, 2, iicell*2 + 2)
    plt.imshow((bias_cell_su_np[0, :, :, icell]),interpolation='nearest', cmap='gray');
    plt.title('bcs')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

  plt.show()

def get_total_weights(w_mother_np, w_del_np, model):
  dimx = model.params.dimx
  dimy = model.params.dimy
  window = model.params.window

  wts = np.array(0 * np.random.randn(dimx, dimy, (2*window +1)**2))
  for idimx in np.arange(dimx):
    print(idimx)
    for idimy in np.arange(dimy):
      wts[idimx, idimy, :] = (np.ndarray.flatten(w_mother_np) +
                              w_del_np[idimx, idimy, :])

  return wts


def plot_weight(weight):
  plt.imshow(np.squeeze(weight), interpolation='nearest', cmap='gray')
  plt.show()
  plt.draw()

def plot_su_strong(b_eval, wts, threshold=0.008):
    # b_eval should be 1 dimensional

    xx = np.sort(np.ndarray.flatten(b_eval));
    idx = np.min(np.where(np.diff(xx)>threshold)) # before : 0.01
    a_thr = (xx[idx] + xx[idx+1])/2

    b_eval_square = np.reshape(b_eval, [wts.shape[0], wts.shape[1]])
    r, c = np.where(b_eval_square>a_thr)

    window = FLAGS.window
    plt.figure()
    for isu in range(len(r)):
      idimx = r[isu]
      idimy = c[isu]
      wt_plot = np.reshape(wts[idimx, idimy, :], (2*window+1,2*window+1))
      ww = np.zeros((40,80))
      ww[idimx*FLAGS.stride: idimx*FLAGS.stride + (2*window+1),
         idimy*FLAGS.stride:
         idimy*FLAGS.stride + (2*window+1)] =  wt_plot

      plt.subplot(1, len(r), isu+1)
      plt.imshow(ww, cmap='gray', interpolation='nearest')
      plt.title(str(b_eval_square[idimx, idimy]))
      plt.xticks([])
      plt.yticks([])


def plot_su_gaussian_spokes(b_eval, w_mother_np, w_del_np, model, stas, threshold=0.008):
  # plot subunits contour and spokes from cells to spokes

  # get useful parameters
  wts = get_total_weights(w_mother_np, w_del_np, model)
  dimx = model.params.dimx
  dimy = model.params.dimy
  window = model.params.window

  # plot contour over fitted subunits
  ww_sum = np.zeros((40, 80))
  ifig = 0

  # get cell centers
  ncells = b_eval.shape[1]
  cell_centers = [[]]*ncells

  for icell in np.arange(ncells):
    try:
      '''
      cell_centers[icell],_ = plot_wts_gaussian_fits(np.reshape(stas[:, icell],
                                                                (40,80)),
                                                     colors='r', levels = 0.8 ,
                                                     alpha_fill=0.3,
                                                     fill_color='k',
                                                     toplot=False)
      '''
      cell_centers[icell] = get_centers(np.reshape(stas[:, icell],(40,80)) )
      print('cell %d done' % icell)
    except:
      print('Failed fitting of STA')
    plt.hold(True)

  # from IPython.terminal.embed import InteractiveShellEmbed
  # ipshell = InteractiveShellEmbed()
  # ipshell()

  # choose colors for cells by approximately solving 3-graph coloring problem.
  cell_cell_distance = np.zeros((ncells, ncells))
  for icell in np.arange(ncells):
    print(icell)
    for jcell in np.arange(ncells):
      cell_cell_distance[icell, jcell] = np.linalg.norm(np.array(cell_centers[icell])-np.array(cell_centers[jcell]))
  A = np.exp(-cell_cell_distance/150)
  w,v = np.linalg.eig(A)
  col_idx = np.argmax(v[:,-5:], axis=1)

  # plot subunits for all the cells, and make su-cell spokes
  su_cell_cnt = np.zeros((dimx, dimy));
  su_centersx = np.zeros((dimx, dimy));
  su_centersy = np.zeros((dimx, dimy));
  cell_num_su = np.array([])
  plt.ion()
  fig = plt.figure()
  ax = plt.subplot(111)
  ax.set_axis_bgcolor((0.9, 0.9, 0.9))

  dist_log=np.array([])
  r = lambda: np.double(random.randint(0,255))/255
  cols = np.array([[0,0,0.8],[0.8,0,0],[0,0.8,0], [0.8,0,0.8], [0.4, 0, 0.8]])
  for icell_cnt, icell in enumerate(np.arange(b_eval.shape[1])):
    #if icell==30 or icell==56: # has low number of spikes = (5000 or 8000)
    #  continue;

    #if icell==43: # not well spike sorted - has bad STA in vision
    #  continue;

    #if icell==105: # high # spikes, STA ok, but connects to a far away su - remove it!
    #  continue;

    icnt = -1
    new_col = cols[col_idx[icell]] # np.array([r(), r(), r()])
    # select thereshold based on breakpoint in values
    xx = np.sort(np.ndarray.flatten(b_eval[:,icell]));
    try :
      #idx = np.min(np.where(np.diff(xx)>threshold)[0][0]) # before : 0.01
      idx = len(xx)-6  # plot 4 SU on each cell
      a_thr = (xx[idx] + xx[idx+1])/2
    except:
      a_thr = np.inf  # no big jump in connection strength detected


    #print('cell: %d, threshold %.3f' % (icell, a_thr))
    #a_thr = 0.1#np.percentile(np.abs(b_eval[:, icell]), 99.5)
    isu_cnt = 0
    for idimx in np.arange(dimx):
      for idimy in np.arange(dimy):
        icnt = icnt + 1

        if(np.abs(b_eval[icnt,icell]) > a_thr):
          isu_cnt += 1
          print('plotting cell: %d, su (%d, %d), weight: %.3f, su_#: %d'% (icell, idimx, idimy, b_eval[icnt,icell], isu_cnt))

          ww = np.zeros((40,80))
          ww[idimx*FLAGS.stride: idimx*FLAGS.stride + (2*window+1),
             idimy*FLAGS.stride:
             idimy*FLAGS.stride + (2*window+1)] =  b_eval[icnt, icell] * (
             np.reshape(wts[idimx, idimy, :], (2*window+1,2*window+1)))
          wts_sq = np.reshape(wts[idimx, idimy, :], (2*window+1,2*window+1))
          try:
            center,_ = plot_wts_gaussian_fits(wts_sq, shiftx=idimx*FLAGS.stride,
                                              shifty=idimy*FLAGS.stride,
                                              colors='k',
                                              levels = 0.5,
                                              toplot=False,
                                              alpha_fill=0,
                                              fill_color='k')

            dx = np.array([cell_centers[icell][0], center[0]])
            dy = np.array([cell_centers[icell][1], center[1]])
            dist_log = np.append(dist_log, np.sqrt(np.sum(dx**2) + np.sum(dy**2)))

            plt.plot(dx,dy, linewidth=2.5, color=new_col)
            #plt.plot(cell_centers[icell][0],cell_centers[icell][1], markersize=10, color=new_col)
            su_cell_cnt[idimx, idimy] +=1
            su_centersx[idimx, idimy] = center[0]
            su_centersy[idimx, idimy] = center[1]
            #plt.hold(True)
            plt.text(cell_centers[icell][0], cell_centers[icell][1], str(icell), fontsize=10, color='b')
            #plt.text(center[0], center[1], str([idimx,idimy,icnt]), fontsize=10, color='k')
            #plt.hold(True)
          except:
            print('Failed fitting of subunit')
          #plt.title('cell %d, %d, wt: %0.3f' %(idimx, idimy, b_eval[icnt, icell]))
          plt.hold(True)

    cell_num_su = np.append(cell_num_su, isu_cnt) # count the number of subunits for each cell.

  # plot dots at centers of connected subunits
  toplot = True
  for idimx in np.arange(dimx):
    for idimy in np.arange(dimy):
      wts_sq = np.reshape(wts[idimx, idimy, :], (2*window+1,2*window+1))
      if su_cell_cnt[idimx, idimy]>0:
        col='k'
        msz = 6
        fill_col='k'
        if su_cell_cnt[idimx, idimy]>1:
          col = 'k'
          msz=6
          fill_col='b'
        try :
          center,_ = plot_wts_gaussian_fits(wts_sq, shiftx=idimx*FLAGS.stride,
                                              shifty=idimy*FLAGS.stride,
                                              colors='k',
                                              levels = 0.5,
                                              toplot=toplot,
                                              alpha_fill=0.4,
                                              fill_color=fill_col)
        except:
          pass
        plt.plot(su_centersx[idimx,idimy], su_centersy[idimx,idimy], '.', markersize=msz, color=col)

  plt.axis('off')
  ax.set_yticks([])
  ax.set_xticks([])
  plt.axis('Image')

  #plt.savefig('coarse_subunits.pdf', facecolor=fig.get_facecolor(), transparent=True)
  plt.show()
  plt.draw()

  # plot the histogram for number of subunits across cells.
  #plt.ion()
  plt.hist(cell_num_su)
  plt.title('# subunits accross cells')
  plt.show()
  plt.draw()


def get_centers(zobs, shiftx=0, shifty=0,):
  """Return the center of mass of zobs"""

  # upsample the subunit image - gives better fits that way
  scale=10
  zobs = np.repeat(np.repeat(zobs, scale, 1), scale, 0)
  shiftx *= scale
  shifty *= scale

  dimx, dimy= zobs.shape
  zobs = np.ndarray.flatten(zobs)
  x = np.repeat(np.expand_dims(np.arange(dimx), 1),dimy, 1);
  y = np.repeat(np.expand_dims(np.arange(dimy), 0),dimx, 0);
  x = np.ndarray.flatten(x)
  y = np.ndarray.flatten(y)
  i = zobs.argmax()
  return y[i], x[i]



def plot_wts_gaussian_fits(zobs, shiftx=0, shifty=0, colors='r', levels=1, fill_color='w', alpha_fill=0.0, toplot=True):

  #define model function and pass independant variables x and y as a list
  def gauss2d((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

  # upsample the subunit image - gives better fits that way
  scale=10
  zobs = np.repeat(np.repeat(zobs,scale,1),scale,0)
  shiftx *= scale
  shifty *= scale

  dimx, dimy= zobs.shape
  zobs = np.ndarray.flatten(zobs)
  x = np.repeat(np.expand_dims(np.arange(dimx), 1),dimy, 1);
  y = np.repeat(np.expand_dims(np.arange(dimy), 0),dimx, 0);
  x = np.ndarray.flatten(x)
  y = np.ndarray.flatten(y)
  xy = [np.ndarray.flatten(x),np.ndarray.flatten(y)]
  import scipy.optimize as opt
  i = zobs.argmax()
  guess = [1, x[i], y[i], 1, 1, 1]
  pred_params, uncert_cov = opt.curve_fit(gauss2d, xy, zobs, p0=guess)

  x = np.repeat(np.expand_dims(np.arange(0,dimx,1), 1),dimy, 1);
  y =np.repeat(np.expand_dims(np.arange(0, dimy,1), 0),dimx, 0);
  xy = [np.ndarray.flatten(x),np.ndarray.flatten(y)]
  zpred = gauss2d(xy, *pred_params)
  max_val = gauss2d([pred_params[1], pred_params[2]], *pred_params)

  zpred_bool = np.double(zpred < max_val*np.exp(-(levels**2)/2)**2)

  if toplot:
    #plt.contour(np.reshape(zpred ,(50,50)),1)
    plt.contour(y+shifty, x+shiftx, np.reshape(zpred_bool,(dimx, dimy)) ,1,linewidth=3,alpha=0.6, colors=colors);
    plt.hold(True)
    zpred_bool[zpred_bool==1] = np.nan
    plt.contourf(y+shifty,x+shiftx, np.reshape(zpred_bool,(dimx, dimy)) ,1, linewidth=0,
                 colors=(fill_color,'w'), alpha=alpha_fill)
    plt.hold(True)

  center = [pred_params[2]+shifty, pred_params[1]+shiftx]
  sigmas = [pred_params[4], pred_params[3]]
  return center, sigmas


def plot_su_for_cells(b_eval, w_mother_np, w_del_np, data, model):
    # plot strong subunits, true STA and STA from fitted subunits for each cell.

    wts = get_total_weights(w_mother_np, w_del_np, model)
    dimx = model.params.dimx
    dimy = model.params.dimy
    window = model.params.window
    cells = np.arange(np.sum(data.cells_choose))
    total_mask = data.cell_mask
    stas = data.stas

    n_cells = cells.shape[0]
    window = FLAGS.window
    #plt.hist(np.ndarray.flatten(b_eval))
    #plt.show()
    #plt.draw()

    #from IPython.terminal.embed import InteractiveShellEmbed
    #ipshell = InteractiveShellEmbed()
    #ipshell()


    #plt.ion()
    plt.figure()
    cells_plt = [23, 52, 60, 10, 73, 100]#45, 61, 105, 0] # use 'cells' to have all cells in plot
    n_plt_cells = len(cells_plt)
    for icell_cnt, icell in enumerate(cells_plt):
      mask2D = np.reshape(total_mask[icell,: ], [40, 80])
      nz_idx = np.nonzero(mask2D)
      np.shape(nz_idx)
      print(nz_idx)
      ylim = np.array([np.min(nz_idx[0])-1, np.max(nz_idx[0])+1])
      xlim = np.array([np.min(nz_idx[1])-1, np.max(nz_idx[1])+1])


      xx = np.sort(np.ndarray.flatten(b_eval[:,icell]));
      #idx = np.min(np.where(np.diff(xx)>0.01))
      #a_thr = (xx[idx] + xx[idx+1])/2
      n_plots_max = 6
      xx_sort = np.sort(xx) # default order is ascend
      a_thr = xx_sort[-(n_plots_max)]

      n_plots = np.sum(np.abs(b_eval[:, icell]) > a_thr)
      nx = np.ceil(np.sqrt(n_plots)).astype('int')
      ny = np.ceil(np.sqrt(n_plots)).astype('int')
      ifig=0
      ww_sum = np.zeros((40,80))

      su_idx = np.where(b_eval[:, icell]>a_thr)[0]
      su_wts = b_eval[su_idx,icell]
      su_order = np.argsort(su_wts)
      su_idx = su_idx[su_order]
      su_idx = su_idx[-1::-1]

      iidimx, iidimy = np.unravel_index(su_idx, [dimx, dimy])
      iidimx = np.squeeze(iidimx)
      iidimy = np.squeeze(iidimy)

      '''
      icnt=-1
      for idx in range(dimy):
        for jdx in range(dimx):
          icnt = icnt+1
          if(icnt==su_idx[0]):
            print(idx, jdx, icnt)
      '''
      for isu in range(iidimx.shape[0]):
        idimx = iidimx[isu]
        idimy = iidimy[isu]
        icnt = su_idx[isu]

        #n_plots_max = 6 # np.max(np.sum(b_eval[:,icell] > a_thr,0))+2
        if (np.abs(b_eval[icnt,icell]) > a_thr): # strongly connected subunit
          ifig = ifig + 1
          fig = plt.subplot(n_plt_cells, n_plots_max+2, icell_cnt*(n_plots_max+2) + ifig + 2)
          print(n_plt_cells, n_plots_max+2, icell_cnt*(n_plots_max+2) + ifig + 2)

          ww = np.zeros((40,80))
          ww[idimx*FLAGS.stride: idimx*FLAGS.stride + (2*window+1),
             idimy*FLAGS.stride:
             idimy*FLAGS.stride + (2*window+1)] = (b_eval[icnt, icell] * (
             np.reshape(wts[idimx, idimy, :], (2*window+1,2*window+1))))
          plt.imshow(ww, interpolation='nearest', cmap='gray')
          plt.ylim(ylim)
          plt.xlim(xlim)
          plt.title(str(b_eval[icnt,icell]), fontsize=10)
          fig.axes.get_xaxis().set_visible(False)
          fig.axes.get_yaxis().set_visible(False)
          ww_sum = ww_sum + ww

      # plot STA from the fitted subunits of the model by
      # just adding them together
      fig = plt.subplot(n_plt_cells, n_plots_max+2, icell_cnt*(n_plots_max+2) + 2)
      plt.imshow(ww_sum, interpolation='nearest', cmap='gray')
      plt.ylim(ylim)
      plt.xlim(xlim)
      fig.axes.get_xaxis().set_visible(False)
      fig.axes.get_yaxis().set_visible(False)
      plt.title('STA from model',fontsize=10)

      # plot true STA from WN
      fig = plt.subplot(n_plt_cells, n_plots_max+2, icell_cnt*(n_plots_max+2) + 1)
      plt.imshow(np.reshape(stas[:, icell], [40, 80]), interpolation='nearest', cmap='gray')
      plt.ylim(ylim)
      plt.xlim(xlim)
      fig.axes.get_xaxis().set_visible(False)
      fig.axes.get_yaxis().set_visible(False)
      plt.title('True STA'+ ' thr: ' + str(a_thr),fontsize=10)

    #plt.ioff()
