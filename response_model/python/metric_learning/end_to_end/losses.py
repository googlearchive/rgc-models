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
r"""Define losses for learning embeddings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def log_sum_exp(anchor_embed, pos_embed, neg_embed, beta):
  r"""Log-sum-exp loss to learn the embedding.

  Use loss = \sum_i log(1 + \sum_j(exp(di+ - dij-))), where
  di = L2 distance between embedded ith anchor and positive.
  dij = L2 distance between embedded ith anchor and jth negative.

  Args :
    anchor_embed : embedding of anchors.
    pos_embed : embedding of positives.
    neg_embed : embedding of negatives.
    beta : temperature parameter.

  Returns :
    loss : the relaxed loss for learning the embedding.
    accuracy_tf : accuracy over the bact Pr(d(anchor, pos)< d(anchor, neg))
    d_pos : distance between anchor and positive.
    d_pairwise_neg : distance between all pairs of anchor and negative.
  """

  d_pos = tf.reduce_sum((anchor_embed - pos_embed)**2, [1, 2, 3])  # batch
  d_pairwise_neg = tf.reduce_sum((tf.expand_dims(anchor_embed, 1) -
                                  tf.expand_dims(neg_embed, 0))**2, [2, 3, 4])
  # batch x batch_neg

  difference = (tf.expand_dims(d_pos/beta, 1) -
                d_pairwise_neg/beta)  # postives x negatives

  # Option 1(unused) : log-sum-exp loss
  # log(\sum_j(exp(d+ - dj-)))
  # loss = tf.reduce_sum(beta * tf.reduce_logsumexp(difference, 1), 0)

  # Option 2
  # log(1 + \sum_j(exp(d+ - dj-)))
  difference_padded = tf.pad(difference, [[0, 0], [0, 1]])
  loss = tf.reduce_sum(beta * tf.reduce_logsumexp(difference_padded, 1), 0)

  accuracy_tf = tf.reduce_mean(tf.sign(-tf.expand_dims(d_pos, 1) +
                                       d_pairwise_neg))

  return loss, accuracy_tf, d_pos, d_pairwise_neg


def log_sum_exp_margin(anchor_embed, pos_embed, neg_embed, beta, margin_pairs):
  r"""Log-sum-exp loss to learn the embedding.

  Use loss = \sum_i log(1 + \sum_j(exp(di+ - dij- + D(ri+, rj)))), where
  di = L2 distance between ith embedded anchor and positive.
  dij = L2 distance between ith embedded anchor and jth negative.
  D(ri+, rj) = Hamming between positive and negative response.

  Args :
    anchor_embed : embedding of anchors.
    pos_embed : embedding of positives.
    neg_embed : embedding of negatives.
    beta : temperature parameter.
    margin_pairs : margins for positive-negative response pairs.

  Returns :
    loss : the relaxed loss for learning the embedding.
    accuracy_tf : accuracy over the bact Pr(d(anchor, pos)< d(anchor, neg))
    d_pos : distance between anchor and positive.
    d_pairwise_neg : distance between all pairs of anchor and negative.
  """

  d_pos = tf.reduce_sum((anchor_embed - pos_embed)**2, [1, 2, 3])  # batch
  d_pairwise_neg = tf.reduce_sum((tf.expand_dims(anchor_embed, 1) -
                                  tf.expand_dims(neg_embed, 0))**2, [2, 3, 4])
  # batch x batch_neg

  difference = (tf.expand_dims(d_pos/beta, 1) -
                d_pairwise_neg/beta + margin_pairs/beta)  # postives x negatives

  # Option 1(unused) : log-sum-exp loss
  # log(\sum_j(exp(d+ - dj-)))
  # loss = tf.reduce_sum(beta * tf.reduce_logsumexp(difference, 1), 0)

  # Option 2
  # log(1 + \sum_j(exp(d+ - dj-)))
  difference_padded = tf.pad(difference, [[0, 0], [0, 1]])
  loss = tf.reduce_sum(beta * tf.reduce_logsumexp(difference_padded, 1), 0)

  accuracy_tf = tf.reduce_mean(tf.sign(-tf.expand_dims(d_pos, 1) +
                                       d_pairwise_neg))

  return loss, accuracy_tf, d_pos, d_pairwise_neg


def log_sum_exp_inner_product(anchor_embed, pos_embed, neg_embed, beta):
  r"""Log-sum-exp loss to learn the embedding.

  Use loss = \sum_i log(1 + \sum_j(exp(di+ - dij-))), where
  di = inner product between ith anchor and positive.
  dij = inner product between ith anchor and jth negative.

  Args :
    anchor_embed : embedding of anchors.
    pos_embed : embedding of positives.
    neg_embed : embedding of negatives.
    beta : temperature parameter.

  Returns :
    loss : the relaxed loss for learning the embedding.
    accuracy_tf : accuracy over the bact Pr(d(anchor, pos)< d(anchor, neg))
    d_pos : distance between anchor and positive.
    d_pairwise_neg : distance between all pairs of anchor and negative.
  """

  d_pos = - tf.reduce_sum((anchor_embed * pos_embed), [1, 2, 3])  # batch
  d_pairwise_neg = - tf.reduce_sum((tf.expand_dims(anchor_embed, 1) *
                                    tf.expand_dims(neg_embed, 0)), [2, 3, 4])
  # batch x batch_neg

  difference = (tf.expand_dims(d_pos/beta, 1) -
                d_pairwise_neg/beta)  # postives x negatives

  # Option 1(unused) : log-sum-exp loss
  # log(\sum_j(exp(d+ - dj-)))
  # loss = tf.reduce_sum(beta * tf.reduce_logsumexp(difference, 1), 0)

  # Option 2
  # log(1 + \sum_j(exp(d+ - dj-)))
  difference_padded = tf.pad(difference, [[0, 0], [0, 1]])
  loss = tf.reduce_sum(beta * tf.reduce_logsumexp(difference_padded, 1), 0)

  accuracy_tf = tf.reduce_mean(tf.sign(-tf.expand_dims(d_pos, 1) +
                                       d_pairwise_neg))

  return loss, accuracy_tf, d_pos, d_pairwise_neg


def log_sum_exp_kl_divergence(anchor_embed_mean, pos_embed_mean, neg_embed_mean,
                              anchor_embed_var, pos_embed_var, neg_embed_var,
                              beta):
  r"""Log-sum-exp loss to learn the embedding.

  Use loss = \sum_i log(1 + \sum_j(exp(di+ - dij-))), where
  di = symmetric KL diveregence between gaussian
          embedded ith anchor and positive.
  dij = symmetric KL diveregence between gaussian
          embedded ith anchor and jth negative.

  Args :
    anchor_embed_mean : mean of gaussian embedding of anchor.
    pos_embed_mean : mean of gaussian embedding of pos.
    neg_embed_mean : mean of gaussian embedding of neg.
    anchor_embed_var : variance of gaussian embedding of anchor.
    pos_embed_var : variance of gaussian embedding of pos.
    neg_embed_var : variance of gaussian embedding of neg.
    beta : temperature parameter.

  Returns :
    loss : the relaxed loss for learning the embedding.
    accuracy_tf : accuracy over the bact Pr(d(anchor, pos)< d(anchor, neg))
    d_pos : distance between anchor and positive.
    d_pairwise_neg : distance between all pairs of anchor and negative.
  """

  # compute symmetric KL divergence between anchor and positive
  mean_term_ap = ((anchor_embed_mean - pos_embed_mean)**2) / 2
  var_term_ap = ((anchor_embed_var**2 + pos_embed_var**2) /
                 (2 * anchor_embed_var * pos_embed_var))
  d_pos = tf.reduce_sum(mean_term_ap * var_term_ap - 0.5, [1, 2, 3])  # batch

  # compute symmetric KL divergence between anchor and negative
  mean_term_an = (tf.expand_dims(anchor_embed_mean, 1) -
                  tf.expand_dims(neg_embed_mean, 0)) ** 2 / 2
  anchor_embed_var_expand = tf.expand_dims(anchor_embed_var, 1)
  neg_embed_var_expand = tf.expand_dims(neg_embed_var, 1)
  var_term_an = ((anchor_embed_var_expand**2 + neg_embed_var_expand**2) /
                 (2 * anchor_embed_var_expand * neg_embed_var_expand))
  d_pairwise_neg = tf.reduce_sum(mean_term_an * var_term_an - 0.5,
                                 [2, 3, 4])  # batch x batch_neg

  difference = (tf.expand_dims(d_pos/beta, 1) -
                d_pairwise_neg/beta)  # postives x negatives

  # Option 1(unused) : log-sum-exp loss
  # log(\sum_j(exp(d+ - dj-)))
  # loss = tf.reduce_sum(beta * tf.reduce_logsumexp(difference, 1), 0)

  # Option 2
  # log(1 + \sum_j(exp(d+ - dj-)))
  difference_padded = tf.pad(difference, [[0, 0], [0, 1]])
  loss = tf.reduce_sum(beta * tf.reduce_logsumexp(difference_padded, 1), 0)

  accuracy_tf = tf.reduce_mean(tf.sign(-tf.expand_dims(d_pos, 1) +
                                       d_pairwise_neg))

  return loss, accuracy_tf, d_pos, d_pairwise_neg

def log_sum_exp_kernel(anchor_embed, pos_embed, neg_embed, beta, kernel):
  r"""Log-sum-exp loss to learn the embedding.

  Use loss = \sum_i log(1 + \sum_j(exp(di+ - dij-))), where
  di = L2 distance between embedded ith anchor and positive.
  dij = L2 distance between embedded ith anchor and jth negative.

  Args :
    anchor_embed : embedding of anchors.
    pos_embed : embedding of positives.
    neg_embed : embedding of negatives.
    beta : temperature parameter.
    kernel : 2d square kernel for loss

  Returns :
    loss : the relaxed loss for learning the embedding.
    accuracy_tf : accuracy over the bact Pr(d(anchor, pos)< d(anchor, neg))
    d_pos : distance between anchor and positive.
    d_pairwise_neg : distance between all pairs of anchor and negative.
  """

  # d_pos = tf.reduce_sum((anchor_embed - pos_embed)**2, [1, 2, 3])  # batch x dimx x dimy x 1
  kernel_4d = tf.expand_dims(tf.expand_dims(kernel, 2), 3)
  diff_pos = (anchor_embed - pos_embed) ** 2

  distances_pos_kernel = tf.nn.conv2d(diff_pos, kernel_4d,
                                   strides=[1, 1, 1, 1], padding='VALID')

  diff_pairwise_neg = (tf.expand_dims(anchor_embed, 1) -
                                  tf.expand_dims(neg_embed, 0))**2 # batch x batch_neg x dimx x dimy x 1
  pairwise_shape = tf.shape(diff_pairwise_neg)
  d_pairwise_neg_4d = tf.reshape(diff_pairwise_neg, [-1, pairwise_shape[2], pairwise_shape[3], pairwise_shape[4]])

  distances_neg_kernel_4d = tf.nn.conv2d(d_pairwise_neg_4d, kernel_4d,
                                      strides=[1, 1, 1, 1], padding='VALID')
  pairwise_shape_convolved = tf.shape(distances_neg_kernel_4d)
  distances_neg_kernel = tf.reshape(distances_neg_kernel_4d,
                                    [pairwise_shape[0], pairwise_shape[1],
                                     pairwise_shape_convolved[1],
                                     pairwise_shape_convolved[2],
                                     pairwise_shape_convolved[3]])

  difference = (tf.expand_dims(distances_pos_kernel/beta, 1) -
                distances_neg_kernel/beta)  # postives x negatives x dimx x dimy x 1

  # Option 1(unused) : log-sum-exp loss
  # log(\sum_j(exp(d+ - dj-)))
  # loss = tf.reduce_sum(beta * tf.reduce_logsumexp(difference, 1), 0)

  # Option 2
  # log(1 + \sum_j(exp(d+ - dj-)))
  difference_padded = tf.pad(difference, [[0, 0], [0, 1], [0, 0] ,[0, 0], [0, 0]])
  loss_example = tf.reduce_sum(beta * tf.reduce_logsumexp(difference_padded, 1), [1, 2, 3])
  loss = tf.reduce_sum(loss_example, 0)

  #
  d_pos = tf.reduce_sum(diff_pos, [1, 2, 3])
  d_pairwise_neg = tf.reduce_sum(diff_pairwise_neg, [2, 3, 4])
  accuracy_tf = tf.reduce_mean(tf.sign(-tf.expand_dims(d_pos, 1) +
                                       d_pairwise_neg))

  return loss, accuracy_tf, d_pos, d_pairwise_neg
