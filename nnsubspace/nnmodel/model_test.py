from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nnsubspace.nndataset.dataset import Dataset
from nnsubspace.nnmodel.model import NNModel

dataset_ = Dataset(dataset='mnist')
model_ = NNModel(dataset='imagenet', model_id='0')