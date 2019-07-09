# nnsubspace
This repo contains the code associated with the paper

Ji, Weiqi, Zhuyin Ren, and Chung K. Law. "Uncertainty Propagation in Deep Neural Network Using Active Subspace." arXiv preprint arXiv:1903.03989 (2019).

It contains the code / model / datasets for the MNIST datasets using LeNet which presented in the first version of the paper, and the ResNet on cifar10/cifar100/ImageNet. The later ones implies the same procedure as the MNIST datasets, and they will be updated in the paper soon. The models for MNIST/cifar10/cifar100 are included in the repo while the one for ImageNet is the same as the pretrained ResNet50 model provided in Keras.

Example codes can be found in subspace_per_sample.py for MNIST/cifar10/cifar100, and subspace_per_sample_imagenet.py for ImageNet. The datasets of MNIST, cifar10 and cifar100 can be readily accessed via keras functions. The ImageNet dataset can be downloaded from the official site and you can start from the validation datasets.

Feel free to contact me at jiweiqi10@gmail.com (Weiqi_Ji) if you have any question, suggestion, comment and future collabration.
