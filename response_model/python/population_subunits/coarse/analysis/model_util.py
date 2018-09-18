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
"""Create models for computing  population firing rates for a given stimulus.

Contains approximately convolutional model - where each subunit is summation of
a common 'mother' filter and subunit specific modifications.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from retina.response_model.python.population_subunits.coarse.analysis import almost_convolutional
from retina.response_model.python.population_subunits.coarse.analysis import almost_convolutional_experimental
from retina.response_model.python.population_subunits.coarse.analysis import almost_convolutional_softmax
from retina.response_model.python.population_subunits.coarse.analysis import almost_convolutional_exponential_dropout
from retina.response_model.python.population_subunits.coarse.analysis import almost_convolutional_exp_dropout_scaling
from retina.response_model.python.population_subunits.coarse.analysis import almost_convolutional_exp_dropout_only_wdelta
from retina.response_model.python.population_subunits.coarse.analysis import almost_convolutional_exp_one_cell_only_wdelta
from retina.response_model.python.population_subunits.coarse.analysis import almost_convolutional_mel_dropout_only_wdelta
from retina.response_model.python.population_subunits.coarse.analysis import almost_convolutional_experimental_wdel_only

def setup_response_model(model_id, *model_build_params):
  """Based on model_id, build the appropriate model graph.

  Args:
    model_id : string identifying which model to build
    *model_build_params : *args for model, passed onto the specific model class

  Returns:
    returns model class with training ops, and other ops to view and analyse it.

  Raises:
    NameError : if the model_id does not match any existing model.
  """

  # Get filename and make folder.
  # Build model.
  if model_id == 'almost_convolutional':
    model_coll = almost_convolutional.AlmostConvolutionalModel(
        *model_build_params)
  elif model_id == 'almost_convolutional_softmax':
    model_coll = almost_convolutional_softmax.AlmostConvolutionalSoftmax(
        *model_build_params)
  elif model_id == 'almost_convolutional_experimental':
    model_coll = almost_convolutional_experimental.AlmostConvolutionalExperimental(
        *model_build_params)
  elif model_id == 'almost_convolutional_exponential_dropout':
    model_coll = almost_convolutional_exponential_dropout.AlmostConvolutionalExponentialDropout(
        *model_build_params)
  elif model_id == 'almost_convolutional_exp_dropout_scaling':
    model_coll = almost_convolutional_exp_dropout_scaling.AlmostConvolutionalExpDropoutScale(
        *model_build_params)
  elif model_id == 'almost_convolutional_exp_dropout_only_wdelta':
    model_coll = almost_convolutional_exp_dropout_only_wdelta.AlmostConvolutionalExpDropoutOnlyWdelta(
        *model_build_params)
  elif model_id == 'almost_convolutional_exp_one_cell_only_wdelta':
    model_coll = almost_convolutional_exp_one_cell_only_wdelta.AlmostConvolutionalExpOneCellOnlyWdelta(
        *model_build_params)
  elif model_id == 'almost_convolutional_mel_dropout_only_wdelta':
    model_coll = almost_convolutional_mel_dropout_only_wdelta.AlmostConvolutionalMELDropoutOnlyWdelta(
        *model_build_params)
  elif model_id == 'almost_convolutional_experimental_wdel_only':
    model_coll = almost_convolutional_experimental_wdel_only.AlmostConvolutionalExperimentalWdelOnly(
        *model_build_params)
  else:
    raise NameError('Model not identified')

  return model_coll

