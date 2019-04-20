from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras.datasets import mnist, cifar10, cifar100
from keras import backend as K
from keras.utils import to_categorical


class Dataset:
    def __init__(self, dataset=None):
        self.dataset = dataset

        self.num_classes = 0
        self.img_rows = 0
        self.img_cols = 0
        self.img_channels = 0
        self.input_shape = []

        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

        self.x_train_mean = 0

        if dataset:
            self.load_data(dataset)

    def decode_predictions(self, y):
        if self.dataset == 'mnist':
            class_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            index = y.argmax()
            print('label: {}, score: {:.2f}'.format(class_list[index], y.max()))

        if self.dataset == 'cifar10':
            class_list = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            index = y.argmax()
            print('label: {}, score: {:.2f}'.format(class_list[index], y.max()))

        if self.dataset == 'cifar100':
            class_list = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                          'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                          'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                          'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                          'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                          'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                          'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                          'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                          'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                          'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                          'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                          'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                          'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                          'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                          'worm']
            index = y.argmax()
            print('label: {}, score: {:.2f}'.format(class_list[index], y.max()))

    def revert_input(self, x):
        if self.dataset in ['mnist']:
            return np.int8(x)

        if self.dataset in ['cifar10', 'cifar100']:
            return np.int8((x + self.x_train_mean) * 255)

    def load_data_mnist(self):
        self.num_classes = 10
        self.img_rows = 28
        self.img_cols = 28
        self.img_channels = 1

        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data(path='mnist.npz')
        if K.image_data_format() == 'channels_first':
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_channels, self.img_rows, self.img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.img_channels, self.img_rows, self.img_cols)
            self.input_shape = (self.img_channels, self.img_rows, self.img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, self.img_channels)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.img_rows, self.img_cols, self.img_channels)
            self.input_shape = (self.img_rows, self.img_cols, self.img_channels)

        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255

        self.x_train_mean = np.zeros_like(self.x_train[0])

        print('x_train shape:', self.x_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        self.y_train = to_categorical(self.y_train, self.num_classes)
        self.y_test = to_categorical(self.y_test, self.num_classes)

    def load_data_cifar10(self):
        self.num_classes = 10
        self.img_rows = 32
        self.img_cols = 32
        self.img_channels = 3

        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        # Input image dimensions.
        self.input_shape = self.x_train.shape[1:]

        # Normalize data.
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255

        self.x_train_mean = np.mean(self.x_train, axis=0)
        self.x_train -= self.x_train_mean
        self.x_test -= self.x_train_mean

        print('x_train shape:', self.x_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        self.y_train = to_categorical(self.y_train, self.num_classes)
        self.y_test = to_categorical(self.y_test, self.num_classes)

    def load_data_cifar100(self):
        self.num_classes = 100
        self.img_rows = 32
        self.img_cols = 32
        self.img_channels = 3

        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar100.load_data()

        # Input image dimensions.
        self.input_shape = self.x_train.shape[1:]

        # Normalize data.
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255

        self.x_train_mean = np.mean(self.x_train, axis=0)
        self.x_train -= self.x_train_mean
        self.x_test -= self.x_train_mean

        print('x_train shape:', self.x_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        self.y_train = to_categorical(self.y_train, self.num_classes)
        self.y_test = to_categorical(self.y_test, self.num_classes)

    def load_data_imagenet(self):
        self.num_classes = 1000
        self.img_rows = 224
        self.img_cols = 224
        self.img_channels = 3

        (self.x_train, self.y_train), (self.x_test, self.y_test) = imagenet_load_data()

    def load_data(self, dataset):
        self.dataset = dataset

        if dataset == 'mnist':
            self.load_data_mnist()

        if dataset == 'cifar10':
            self.load_data_cifar10()

        if dataset == 'cifar100':
            self.load_data_cifar100()

        if dataset == 'imagenet':
            self.load_data_imagenet()
