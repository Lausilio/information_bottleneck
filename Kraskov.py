from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy.special


def kraskov_entropy(T, k=2):
    n_samples, d = T.shape

    # Find k+1 nearest neighbors in the T space
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(T)
    distances, _ = nbrs.kneighbors(T)

    # Calculate r_i as the distance to the k-th nearest neighbor
    r = distances[:, k]
    a = 0
    for i in r:
        a += np.log(i + 1e-16)
    a = a*(d/n_samples)
    
    # Compute the entropy using Kraskov's formula
    entropy = (a + d/2*np.log(np.pi) - scipy.special.gammaln(d/2 + 1) + scipy.special.psi(n_samples) - scipy.special.psi(k))
    
    return entropy