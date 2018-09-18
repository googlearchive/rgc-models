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
"""Book-keeping operations (initialize, save/reload models, summary)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


FLAGS = tf.app.flags.FLAGS


def get_filename(training_datasets, testing_datasets, beta, sr_model):
  """Get filename for saving results."""

  if sr_model == 'convolutional_embedding':
    model_str = ''  # For past consistency.
  else:
    model_str = '_%s' % sr_model

  file_name = ('end_to_end%s_stim_%s_resp_%s_beta_%d_taskid_%d'
               '_training_%s_testing_%s_%s' % (model_str,
                                               FLAGS.stim_layers,
                                               FLAGS.resp_layers, beta,
                                               FLAGS.taskid,
                                               str(training_datasets)[1: -1],
                                               str(testing_datasets)[1: -1],
                                               FLAGS.save_suffix))
  if FLAGS.subsample_cells_train:
    file_name += '_subsample_cells'

  if sr_model == 'convolutional_encoder_2' or sr_model == 'convolutional_encoder_using_retina_id':
    file_name += ('reg_%.3f_%.3f_%.3f_%.3f' % (FLAGS.scale_regularization,
                                               FLAGS.scale_triplet,
                                               FLAGS.scale_encode,
                                               FLAGS.scale_decode))

  if FLAGS.use_EIs:
    file_name += ('_bn_ei_%s_ei_layers_%s_scales_%.3f_%.3f_%.3f' %
                  (FLAGS.batch_norm_ei, FLAGS.ei_layers,
                   FLAGS.scale_encode_from_ei,
                   FLAGS.scale_regularization_from_ei,
                   FLAGS.scale_match_embeddding))

  return file_name


def initialize_model(save_folder, file_name, sess):
  """Setup model variables and saving information.

  Args :
    save_folder (string) : Folder to store model.
                           Makes one if it does not exist.
    file_name (string) : Prefix of model/checkpoint files.
    sess : Tensorflow session.

  Returns :
    saver_var : TF op for saving parameters.
    start_iter (int): loaded iteration to start from.

  """

  # Make folder.
  if not gfile.IsDirectory(save_folder):
    gfile.MkDir(save_folder)

  # Initialize variables.
  saver_var, start_iter = initialize_variables(sess, save_folder, file_name)
  return saver_var, start_iter


def initialize_variables(sess, save_folder, short_filename):
  """Initialize variables de-novo or restore from previous fits."""

  sess.run(tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer()))
  saver_var = tf.train.Saver(tf.global_variables(),
                             keep_checkpoint_every_n_hours=5)
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
  except ValueError:
    tf.logging.info('No previous dataset - cant load file')
  except AttributeError:
    tf.logging.info('No previous dataset - cant find start_iter')

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


def write_tensorboard(sess, stim_test_embed, labels, label_images=None,
                      embedding_name='stim_embed',
                      log_dir='/home/bhaishahster/tb',
                      model_name='model'):
  """Save embedding to visualize in tensorboard."""

  ## Tensorboard embedding visualization
  # from tf.contrib.tensorboard.plugins import projector
  projector = tf.contrib.tensorboard.plugins.projector

  # Create randomly initialized embedding weights which will be trained.
  embedding_var = tf.Variable(stim_test_embed.astype(np.float32),
                              name=embedding_name)

  config = projector.ProjectorConfig()

  # You can add multiple embeddings. Here we add only one.
  embedding = config.embeddings.add()
  embedding.tensor_name = embedding_var.name

  # Link this tensor to its metadata file (e.g. labels).
  embedding.metadata_path = os.path.join(log_dir,
                                         'metadata.tsv')
  # Optionally, give label images.
  if label_images is not None:
    embedding.sprite.image_path = os.path.join(log_dir,
                                               'sprite_images_%s.png' %
                                               model_name)
    embedding.sprite.single_image_dim.extend([label_images.shape[1],
                                              label_images.shape[2]])

  # Use the same LOG_DIR where you stored your checkpoint.
  summary_writer = tf.summary.FileWriter(log_dir)

  # The next line writes a projector_config.pbtxt in the LOG_DIR.
  # TensorBoard will read this file during startup.
  projector.visualize_embeddings(summary_writer, config)
  sess.run(tf.variables_initializer(var_list=[embedding_var]))
  saver = tf.train.Saver(var_list=[embedding_var])
  saver.save(sess, os.path.join(log_dir, '%s.ckpt' % model_name), 0)

  # Create metadata
  with open(embedding.metadata_path, 'w') as f:
    print(embedding.metadata_path)
    f.write('Index\tLabel\n')
    for index, label in enumerate(labels):
      f.write('%d\t%d\n' % (index, label))

  if label_images is not None:
    sprite_image = create_sprite_image(label_images)
    plt.imsave(embedding.sprite.image_path, sprite_image, cmap='gray')


def create_sprite_image(images):
  """Returns a sprite image consisting of images passed as argument."""

  if isinstance(images, list):
    images = np.array(images)  # Images should be count x width x height
  img_h = images.shape[1]
  img_w = images.shape[2]
  n_plots = int(np.ceil(np.sqrt(images.shape[0])))

  spriteimage = np.ones((img_h * n_plots, img_w * n_plots))
  for i in range(n_plots):
    for j in range(n_plots):
      this_filter = i * n_plots + j
      if this_filter < images.shape[0]:
        this_img = images[this_filter]
        spriteimage[i * img_h:(i + 1) * img_h,
                    j * img_w:(j + 1) * img_w] = this_img

  return spriteimage
