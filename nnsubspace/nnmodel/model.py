#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import load_model
from keras import backend as K


class NNModel:
    def __init__(self, dataset_name, model_id=0):
        self.model_file = []
        self.model = []

        self.model_file_dict = {}
        self.model_file_dict_mnist = {}
        self.model_file_dict_cifar10 = {}
        self.model_file_dict_cifar100 = {}
        self.model_file_dict_imagenet = {}

        self.tf_session = K.get_session()
        self.set_model(dataset_name, model_id)

    def set_model(self, dataset_name, model_id):
        self.model_file_dict_mnist = {
            '0': 'models/model_mnist_cnn_softplus.h5',
            '1': 'model1'
        }
        self.model_file_dict_cifar10 = {
            '0': 'models/cifar10_ResNet20v2_model.189.h5',
            '1': 'model1'
        }
        self.model_file_dict_cifar100 = {
            '0': 'models/cifar100_ResNet20v2_model.090.h5',
            '1': 'model1'
        }
        self.model_file_dict_imagenet = {
            '0': 'models/model_resnet50_imagenet.h5',
            '1': 'model1'
        }

        self.model_file_dict = {
            'mnist': self.model_file_dict_mnist,
            'cifar10': self.model_file_dict_cifar10,
            'cifar100': self.model_file_dict_cifar100,
            'imagenet': self.model_file_dict_imagenet
        }

        try:
            self.model_file = self.model_file_dict[dataset_name][model_id]
        except:
            print('Error: dataset {} and model_id {} is not avaliable'.format(
                dataset_name, model_id))

        self.model = load_model(self.model_file)
