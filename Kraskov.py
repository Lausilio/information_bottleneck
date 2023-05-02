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
    
    # Compute the entropy using Kraskov's formula
    entropy = ( (np.sum(np.log(r + 1e-16))/np.log(2))*(d/n_samples) + d/2*np.log(np.pi)/np.log(2) - scipy.special.gammaln(d/2 + 1)/np.log(2) + scipy.special.psi(n_samples) - scipy.special.psi(k) )
    
    return entropy
