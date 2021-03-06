from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from nnsubspace.nnsubspace.subspace import NNSubspace
from nnsubspace.nndataset.dataset import Dataset
from nnsubspace.nnmodel.model import NNModel
import nnsubspace.visual.subspaceplot as subspaceplot

dataset_name = 'mnist'

dataset = Dataset(dataset_name=dataset_name)
model = NNModel(dataset_name=dataset_name, model_id='0')

# Score trained model on test data.
scores = model.model.evaluate(dataset.x_test, dataset.y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

for i_sample, x in enumerate(dataset.x_test[0:5000]):
    x = np.expand_dims(x, axis=0)
    y = model.model.predict(x)
    if y.max() < 0.5:
        print('sample {}'.format(i_sample))
        dataset.decode_predictions(y)
        dataset.decode_predictions(dataset_.y_test[i_sample])

        subspaceplot.imshow(np.squeeze(x + dataset.x_train_mean),
                            figsize=(2, 2))

        AS = NNSubspace(model=model.model,
                        x=x,
                        x_train_mean=dataset.x_train_mean)

        AS.sampling_setup(num_gradient_mc=500,
                          num_rs_mc=50000,
                          seed=7,
                          bool_clip=True,
                          sigma=5 / 255,
                          num_eigenvalue=20)
        AS.run()
