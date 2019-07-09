#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nnsubspace.nndataset.dataset import Dataset
from nnsubspace.nnmodel.model import NNModel

dataset = Dataset(dataset_name='mnist')
model = NNModel(dataset_name='imagenet', model_id='0')