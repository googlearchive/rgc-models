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
"""Fit subunits for coarse resolution data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags


# Import module
import numpy as np
rng = np.random

FLAGS = flags.FLAGS


def main(argv):

  # Coarse data
  with open('tasks_coarse_aug.txt', 'wb') as f:
    # individual cells
    for icell in range(107):
      for nsub in  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                    14, 15, 16, 17, 18]:
        for proj_type in ['lnl1']:
          for lam_proj in [0.0]:
            # for ipart in range(1):
            partitions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            f.write('%s;%d;%s;%.3f;%s;%s\n' % ([icell], nsub, proj_type,
                                               lam_proj, partitions, [icell]))

  # population, with common mask
  with open('tasks_coarse_pop_common_mask_aug.txt', 'wb') as f:
    mask_cells = [39, 42, 44, 45]
    for cells in [[39, 42, 44, 45], [39], [42], [44], [45]]:
      for nsub in  np.arange(len(cells), 15 * len(cells)): # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for proj_type in ['lnl1']:
          for lam_proj in np.arange(0, 3, 0.2):
              f.write('%s;%d;%s;%.3f;%s;%s\n' % (cells, nsub, proj_type, lam_proj, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], mask_cells))

  with open('tasks_coarse_pop_common_mask_aug_2.txt', 'wb') as f:
    mask_cells = [39, 42, 44, 45]
    for cells in [[39, 42, 44, 45], [39], [42], [44], [45]]:
      for nsub in  np.arange(len(cells), 9 * len(cells)): # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for proj_type in ['lnl1']:
          for lam_proj in np.arange(0.1, 3, 0.1):
            f.write('%s;%d;%s;%.3f;%s;%s\n' % (cells, nsub, proj_type, lam_proj, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], mask_cells))

  # Null tasks
  with open('tasks_null_2015_10_29_2_aug.txt', 'wb') as f:
    for cells in range(107):
      for nsub in np.arange(1, 12):
        for proj_type in ['lnl1']:
          for lam_proj in [0.0]:
            partitions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            f.write('%s;%d;%s;%.3f;%s;%s\n' % ([cells], nsub, proj_type, lam_proj, partitions, [cells]))

  with open('tasks_null_2015_11_09_1_aug.txt', 'wb') as f:
    for cells in range(99):
      for nsub in np.arange(1, 12):
        for proj_type in ['lnl1']:
          for lam_proj in [0.0]:
            partitions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            f.write('%s;%d;%s;%.3f;%s;%s\n' % ([cells], nsub, proj_type, lam_proj, partitions, [cells]))

  with open('tasks_null_2015_11_09_8_aug.txt', 'wb') as f:
    for cells in range(199):
      for nsub in np.arange(1, 11):  # TODO(bhaishahster): Run again for the 12 th subunit!
        for proj_type in ['lnl1']:
          for lam_proj in [0.0]:
            partitions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            f.write('%s;%d;%s;%.3f;%s;%s\n' % ([cells], nsub, proj_type, lam_proj, partitions, [cells]))


  with open('tasks_null_2015_11_09_8_aug_part2.txt', 'wb') as f:
    for cells in range(199):
      for nsub in [12]:  # TODO(bhaishahster): Run again for the 12 th subunit!
        for proj_type in ['lnl1']:
          for lam_proj in [0.0]:
            partitions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            f.write('%s;%d;%s;%.3f;%s;%s\n' % ([cells], nsub, proj_type, lam_proj, partitions, [cells]))


  with open('tasks_null_2015_11_09_8_aug_part3.txt', 'wb') as f:
    for cells in range(199):
      for nsub in [11]:  # TODO(bhaishahster): Run again for the 12 th subunit!
        for proj_type in ['lnl1']:
          for lam_proj in [0.0]:
            partitions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            f.write('%s;%d;%s;%.3f;%s;%s\n' % ([cells], nsub, proj_type, lam_proj, partitions, [cells]))

  with open('tasks_nsem_3_datasets_aug.txt', 'wb') as f:
    for cells in range(244):
      for nsub in np.arange(1, 9):
        for proj_type in ['lnl1']:
          for lam_proj in [0]: #np.arange(0, 0.4, 0.1):
              f.write('%s;%d;%s;%.3f\n' % ([cells], nsub, proj_type, lam_proj))

  with open('tasks_nsem_3_datasets_aug_2.txt', 'wb') as f:
    for cells in range(244):
      for nsub in np.arange(9, 15):
        for proj_type in ['lnl1']:
          for lam_proj in [0]: #np.arange(0, 0.4, 0.1):
              f.write('%s;%d;%s;%.3f\n' % ([cells], nsub, proj_type, lam_proj))
  '''
  # Coarse data
  with open('tasks_coarse_2.txt', 'wb') as f:
    # individual cells
    for icell in range(107):
      for nsub in  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for proj_type in ['lnl1']:
          for lam_proj in [0.0]:
            for ipart in range(5):
              f.write('%s;%d;%s;%.3f;%d\n' % ([icell], nsub, proj_type, lam_proj, ipart))

    # population
    for cells in [[39, 42, 44, 45], [39, 42], [44, 45]]:
      for nsub in  np.arange(1, 11) * len(cells): # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for proj_type in ['lnl1']:
          for lam_proj in np.arange(0, 4, 0.5):
            for ipart in range(5):
              f.write('%s;%d;%s;%.3f;%d\n' % (cells, nsub, proj_type, lam_proj, ipart))

    for cells in [range(107)]:
      for nsub in  np.arange(107, 107*10, 107): # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for proj_type in ['lnl1']:
          for lam_proj in np.arange(0, 7, 0.5):
            for ipart in range(5):
              f.write('%s;%d;%s;%.3f;%d\n' % (cells, nsub, proj_type, lam_proj, ipart))


  # Fine resolution
  # Few cells, more exhaustive
  with open('tasks_fine.txt', 'wb') as f:
    for cells in [1, 28, 38, 43, 30, 24]: #range(52):
      for nsub in np.arange(1, 20):
        for proj_type in ['lnl1', 'l1']:
          for lam_proj in np.arange(0, 2, 0.1):
            for ipart in range(2):
              f.write('%s;%d;%s;%.3f;%d\n' % ([cells], nsub, proj_type, lam_proj, ipart))


  # Few cells, more exhaustive (2)
  with open('tasks_fine_3.txt', 'wb') as f:
    for cells in [1, 28, 38, 43, 30, 24]: #range(52):
      for nsub in np.arange(1, 12):
        for proj_type in ['lnl1', 'l1']:
          for lam_proj in np.arange(0.05, 2, 0.1):
            for ipart in range(2):
              f.write('%s;%d;%s;%.3f;%d\n' % ([cells], nsub, proj_type, lam_proj, ipart))
  

  # Fine resolution all cells, less exhaustive
  with open('tasks_fine_2.txt', 'wb') as f:
    for cells in range(52):
      for nsub in np.arange(1, 18):
        for proj_type in ['lnl1']:
          for lam_proj in np.arange(0, 2, 0.2):
            for ipart in range(1):
              f.write('%s;%d;%s;%.3f;%d\n' % ([cells], nsub, proj_type, lam_proj, ipart))


  # Fine resolution - population
  # Few cells, more exhaustive
  with open('tasks_fine_population.txt', 'wb') as f:
    for cells in [[0, 1, 2, 3]]: #range(52):
      for nsub in np.arange(4, 15 * 4):
        for proj_type in ['lnl1']:
          for lam_proj in np.arange(0, 3, 0.1):
            for ipart in range(2):
              f.write('%s;%d;%s;%.3f;%d\n' % (cells, nsub, proj_type, lam_proj, ipart))


  # population
  with open('tasks_coarse_pop.txt', 'wb') as f:
    for cells in [[39, 42, 44, 45], [39, 42], [44, 45]]: #, [39], [42], [44], [45]]:
      for nsub in  np.arange(len(cells), 11 * len(cells)): # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for proj_type in ['lnl1']:
          for lam_proj in np.arange(0, 4, 0.2):
            for ipart in range(5):
              f.write('%s;%d;%s;%.3f;%d\n' % (cells, nsub, proj_type, lam_proj, ipart))

  # population - 2
  with open('tasks_coarse_pop_2.txt', 'wb') as f:
    for cells in [[39, 42, 44, 45], [39, 42], [44, 45], [39], [42], [44], [45]]:
      for nsub in  np.arange(len(cells), 11 * len(cells)): # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for proj_type in ['lnl1']:
          for lam_proj in np.arange(0, 1.5, 0.1):
            for ipart in range(5):
              f.write('%s;%d;%s;%.3f;%d\n' % (cells, nsub, proj_type, lam_proj, ipart))

  # population, with common mask
  with open('tasks_coarse_pop_common_mask.txt', 'wb') as f:
    mask_cells = [39, 42, 44, 45]
    for cells in [[39, 42, 44, 45], [39, 42], [44, 45], [39], [42], [44], [45]]:
      for nsub in  np.arange(len(cells), 11 * len(cells)): # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for proj_type in ['lnl1']:
          for lam_proj in np.arange(0, 3, 0.2):
            for ipart in range(5):
              f.write('%s;%d;%s;%.3f;%d;%s\n' % (cells, nsub, proj_type, lam_proj, ipart, mask_cells))

  # population, with common mask, bigger
  with open('tasks_coarse_pop_common_mask_2.txt', 'wb') as f:
    mask_cells = [39, 42, 44, 45, 40, 46, 48]
    for cells in [[39, 42, 44, 45, 40, 46, 48], [39], [42], [44], [45], [40], [46], [48]]:
      for nsub in  np.arange(len(cells), 11 * len(cells)): # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for proj_type in ['lnl1']:
          for lam_proj in np.arange(0, 2, 0.2):
            for ipart in range(5):
              f.write('%s;%d;%s;%.3f;%d;%s\n' % (cells, nsub, proj_type, lam_proj, ipart, mask_cells))

  # population, with common mask, second group
  with open('tasks_coarse_pop_common_mask_3.txt', 'wb') as f:
    mask_cells = [77, 83, 78, 76, 74, 4, 86]
    for cells in [[77, 83, 78, 76, 74, 4, 86], [77], [83], [78], [76], [74], [4], [86]]:
      for nsub in  np.arange(len(cells), 11 * len(cells)): # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for proj_type in ['lnl1']:
          for lam_proj in np.arange(0, 2, 0.2):
            for ipart in range(5):
              f.write('%s;%d;%s;%.3f;%d;%s\n' % (cells, nsub, proj_type, lam_proj, ipart, mask_cells))


  # population of 4 cells, with common mask, fourth group
  with open('tasks_coarse_pop_common_mask_4.txt', 'wb') as f:
    mask_cells = [77, 83, 78, 76]
    for cells in [[77, 83, 78, 76], [77], [83], [78], [76]]:
      for nsub in  np.arange(len(cells), 11 * len(cells)):
        for proj_type in ['lnl1']:
          for lam_proj in np.arange(0, 2, 0.2):
            for ipart in range(5):
              f.write('%s;%d;%s;%.3f;%d;%s\n' % (cells, nsub, proj_type, lam_proj, ipart, mask_cells))


  # population of 4 cells, with common mask, fifth group
  with open('tasks_coarse_pop_common_mask_5.txt', 'wb') as f:
    mask_cells = [95, 90, 3, 6]
    for cells in [[95, 90, 3, 6], [95], [90], [3], [6]]:
      for nsub in  np.arange(len(cells), 11 * len(cells)):
        for proj_type in ['lnl1']:
          for lam_proj in np.arange(0, 2, 0.2):
            for ipart in range(5):
              f.write('%s;%d;%s;%.3f;%d;%s\n' % (cells, nsub, proj_type, lam_proj, ipart, mask_cells))



  # Compare convolutional and subunit model.
  # Vary convolutional network architecture and subunit model.
  # Vary amount of training data.
  with open('tasks_compare_conv.txt', 'wb') as f:
    for cells in range(52):
      for frac_train in [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.8]:

        model = 'su'
        for nsub in np.arange(1, 11):
          # LNL1
          for proj_type in ['lnl1']:
            for lam_proj in [0.0]:
              f.write('%s;%s;%d;%s;%.3f;%.4f\n' % ([cells], model, nsub, proj_type, lam_proj, frac_train))

        model = 'conv'
        for dim_filters in [1,2,3,4,5,6]:
          for strides in [1]:
            for num_filters in [1]:
              f.write('%s;%s;%d;%d;%d;%.4f\n' % ([cells], model, dim_filters, strides, num_filters, frac_train))


  with open('tasks_less_data_su.txt', 'wb') as f:
    for cells in range(20): # out of 107
      for frac_train in [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.8]:
        model = 'su'
        for nsub in np.arange(1, 11):
          # LNL1
          for proj_type in ['lnl1']:
            for lam_proj in np.arange(0, 1.0, 0.2):
              f.write('%s;%s;%d;%s;%.3f;%.4f\n' % ([cells], model, nsub, proj_type, lam_proj, frac_train))


  with open('tasks_less_data_su_2.txt', 'wb') as f:
    for cells in range(20): # out of 107
      for frac_train in [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.8]:
        model = 'su'
        for nsub in np.arange(1, 11):
          # LNL1
          for proj_type in ['lnl1']:
            for lam_proj in np.arange(1.2, 2.2, 0.2):
              f.write('%s;%s;%d;%s;%.3f;%.4f\n' % ([cells], model, nsub, proj_type, lam_proj, frac_train))


  with open('tasks_less_data_su_l1.txt', 'wb') as f:
    for cells in range(20): # out of 107
      for frac_train in [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.8]:
        model = 'su'
        for nsub in np.arange(1, 11):
          # LNL1
          for proj_type in ['l1']:
            for lam_proj in np.arange(0, 1.0, 0.2):
              f.write('%s;%s;%d;%s;%.3f;%.4f\n' % ([cells], model, nsub, proj_type, lam_proj, frac_train))

  with open('tasks_less_data_su_797.txt', 'wb') as f:
    for cells in [10]: # out of 107
      for frac_train in [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.8]:
        model = 'su'
        for nsub in np.arange(1, 11):
          # L1
          for proj_type in ['l1']:
            for lam_proj in np.arange(0, 0.5, 0.05):
              f.write('%s;%s;%d;%s;%.3f;%.4f\n' % ([cells], model, nsub, proj_type, lam_proj, frac_train))

          # LNL1
          for proj_type in ['lnl1']:
            for lam_proj in np.arange(0, 1.6, 0.05):
              f.write('%s;%s;%d;%s;%.3f;%.4f\n' % ([cells], model, nsub, proj_type, lam_proj, frac_train))



  with open('tasks_less_data_su_few_cells_2.txt', 'wb') as f:
    for cells in range(40): #[92,  40,  73,  63,  47,  11,  51,  76]:
      for frac_train in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]: # old - > [0.01, 0.03, 0.05, 0.1, 0.25, 0.8]:
        model = 'su'
        for nsub in [4]: #np.arange(1, 11):
          # L1
          for proj_type in ['l1']:
            for lam_proj in np.arange(0, 1.0, 0.05):
              f.write('%s;%s;%d;%s;%.3f;%.4f\n' % ([cells], model, nsub, proj_type, lam_proj, frac_train))

          # LNL1
          for proj_type in ['lnl1']:
            for lam_proj in np.arange(0, 1.0, 0.05):
              f.write('%s;%s;%d;%s;%.3f;%.4f\n' % ([cells], model, nsub, proj_type, lam_proj, frac_train))



  with open('tasks_compare_conv_deep.txt', 'wb') as f:
    for cells in range(52):
      for frac_train in [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.8]:

        model = 'su'
        for nsub in np.arange(1, 11):
          # LNL1
          for proj_type in ['lnl1']:
            for lam_proj in [0.0]:
              f.write('%s;%s;%d;%s;%.3f;%.4f\n' % ([cells], model, nsub, proj_type, lam_proj, frac_train))

        model = 'conv'
        for num_layers in [1, 3, 6]:
          for dim_filters in [1, 3, 6]:
            for strides in [1]:
              for num_filters in [1]:

                layer_str = ''
                for ilayer in range(num_layers):
                  layer_str += '%d,%d,%d,' % (dim_filters, num_filters, strides)

                f.write('%s;%s;%s;%.4f\n' % ([cells], model, layer_str[:-1], frac_train))

  # NSEM tasks
  with open('tasks_nsem_2015-11-09-3.txt', 'wb') as f:
    for cells in range(107):
      for nsub in np.arange(1, 11):
        for proj_type in ['lnl1']:
          for lam_proj in [0.0]:
            for ipart in range(5):
              f.write('%s;%d;%s;%.3f;%d\n' % ([cells], nsub, proj_type, lam_proj, ipart))

  with open('tasks_nsem_2015-11-09-3_2.txt', 'wb') as f:
    for cells in range(107):
      for nsub in np.arange(1, 11):
        for proj_type in ['lnl1']:
          for lam_proj in np.arange(0, 1, 0.2):
            for ipart in range(1):
              f.write('%s;%d;%s;%.3f;%d\n' % ([cells], nsub, proj_type, lam_proj, ipart))




  with open('tasks_nsem_2012-08-09-3.txt', 'wb') as f:
    for cells in range(26):
      for nsub in np.arange(1, 11):
        for proj_type in ['lnl1']:
          for lam_proj in np.arange(0, 1, 0.1):
              f.write('%s;%d;%s;%.3f\n' % ([cells], nsub, proj_type, lam_proj))


  with open('tasks_nsem_3_datasets.txt', 'wb') as f:
    for cells in range(244):
      for nsub in np.arange(1, 11):
        for proj_type in ['lnl1']:
          for lam_proj in np.arange(0, 0.7, 0.2):
              f.write('%s;%d;%s;%.3f\n' % ([cells], nsub, proj_type, lam_proj))

  with open('tasks_nsem_3_datasets_2.txt', 'wb') as f:
    for cells in range(244):
      for nsub in np.arange(1, 11):
        for proj_type in ['lnl1']:
          for lam_proj in np.arange(0.1, 0.9, 0.2):
              f.write('%s;%d;%s;%.3f\n' % ([cells], nsub, proj_type, lam_proj))

  with open('tasks_nsem_3_datasets_3.txt', 'wb') as f:
    for cells in range(244):
      for nsub in np.arange(1, 11):
        for proj_type in ['lnl1']:
          for lam_proj in np.arange(0.05, 0.4, 0.1):
              f.write('%s;%d;%s;%.3f\n' % ([cells], nsub, proj_type, lam_proj))


  # Fit WN (3 stages) on coarse data
  with open('tasks_coarse_2015_11_09_1.txt', 'wb') as f:
    # individual cells
    for icell in range(99):
      for nsub in  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for proj_type in ['lnl1']:
          for lam_proj in [0.0]:
            for ipart in range(5):
              f.write('%s;%d;%s;%.3f;%d\n' % ([icell], nsub, proj_type, lam_proj, ipart))

  with open('tasks_coarse_2015_11_09_8.txt', 'wb') as f:
    # individual cells
    for icell in range(199):
      for nsub in  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for proj_type in ['lnl1']:
          for lam_proj in [0.0]:
            for ipart in range(5):
              f.write('%s;%d;%s;%.3f;%d\n' % ([icell], nsub, proj_type, lam_proj, ipart))



  '''

if __name__ == '__main__':
  app.run(main)
