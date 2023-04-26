import numpy as np
from matplotlib import pyplot as plt

mi_array = np.load('apr26_t0824.npy')

epochs = list(range(len(mi_array)))
plt.scatter(mi_array[:, 0], mi_array[:, 1], c=epochs, label='Mutual Information L1')
plt.xlabel('I(X,T)')
plt.ylabel('I(Y,T)')
plt.colorbar()
plt.show()