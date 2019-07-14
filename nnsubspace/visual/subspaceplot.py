from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


def imshow(x, figsize=None):
    if figsize is None:
        figsize = (5, 5)

    plt.figure(figsize=figsize)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(x)
    plt.autoscale(tight=True)
    plt.show()


def eigenplot(w, num_eigenvalue, figsize=None):
    if figsize is None:
        figsize = (5, 5)

    plt.figure(figsize=figsize)
    plt.semilogy(range(num_eigenvalue), w[0:num_eigenvalue]**2, '-o')
    plt.xlabel('index')
    plt.ylabel('eigenvalue')
    plt.show()


def eigenvectorplot(eigenvector, figsize=None):
    if figsize is None:
        figsize = (5, 5)

    plt.figure(figsize=figsize)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    eigenvector = np.abs(eigenvector) / np.max(np.abs(eigenvector))
    ax.imshow(eigenvector)
    plt.autoscale(tight=True)
    plt.show()


def summaryplot(xv, y, poly1d, figsize=None):
    if figsize is None:
        figsize = (5, 5)

    plt.figure(figsize=figsize)
    plt.plot(xv, y, 'o')
    plt.plot(np.sort(xv), poly1d(np.sort(xv)), '-')
    plt.xlabel(r'$w_1^T\xi$')
    plt.ylabel('output')
    plt.show()