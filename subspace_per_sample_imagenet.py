from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

import matplotlib.pyplot as plt
import numpy as np
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image

import nnsubspace.visual.subspaceplot as subspaceplot
from nnsubspace.nnmodel.model import NNModel
from nnsubspace.nnsubspace.subspace import NNSubspace


def load_imagenet(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)/255
    return x


dataset_name = 'imagenet'

x_train_mean = np.zeros((224, 224, 3))
x_train_mean[:, :, 0] = 103.939
x_train_mean[:, :, 1] = 116.779
x_train_mean[:, :, 2] = 123.68

x_train_mean = x_train_mean / 255

model_ = NNModel(dataset=dataset_name, model_id='0')

# read imagenet_datalist
with open("output.csv", 'r', newline='') as resultFile:
    wr = csv.reader(resultFile)
    imagenet_datalist = list(wr)

num_gradient_mc = 400

# loop over images
for i_sample in [7526]: #1205, 1410, 227, 7526
    img_path = imagenet_datalist[i_sample][0]
    label_true = imagenet_datalist[i_sample][1]
    label_pred = imagenet_datalist[i_sample][2]
    score = np.float32(imagenet_datalist[i_sample][3])

    if score < 1:
        print('#{}'.format(i_sample), label_true, label_pred, score)

        x = load_imagenet(img_path)

        AS = NNSubspace(model=model_.model, x=x, x_train_mean=x_train_mean, dataset_name=dataset_name)

        AS.sampling_setup(num_gradient_mc=num_gradient_mc,
                          num_rs_mc=num_gradient_mc * 4,
                          seed=7,
                          bool_clip=True,
                          sigma=50 / 255,
                          num_eigenvalue=20)
        AS.run()

        subspaceplot.imshow(np.squeeze(x + x_train_mean), figsize=(3, 3))
        plt.figure(figsize=(3, 3))
        ax = plt.axes([0, 0, 1, 1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        eigenvector = np.copy(AS.eigenvector[:, 0])
        eigenvector_norm = np.abs(eigenvector.reshape(224, 224, 3)) / np.abs(eigenvector).max()
        eigenvector_clip = eigenvector_norm > 0.03
        print(np.sum( (eigenvector * eigenvector_clip.flatten()) ** 2))
        ax.imshow(eigenvector_clip * np.squeeze(x + x_train_mean))
        plt.autoscale(tight=True)
        plt.show()
