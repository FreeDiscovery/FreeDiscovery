# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array


def _merge_clusters(X):
    """
    Compute a union of all clusters

    Used to determine which cluster_id a document should belong to if at least one of it's
    lexicons suggest that it's a duplicate

    Approximate time complexity O(n_samples*n_features)

    Parameters
    ----------
     X: array (n_samples, n_features)
    """
    X = check_array(X, ensure_2d=True)

    n_samples, n_features = X.shape

    y = np.zeros(n_samples, dtype=X.dtype)

    out = {}

    for (i_idx, X_row) in enumerate(X):
        for j_idx, X_el in enumerate(X_row):
            if (j_idx, X_el) in out:
                res = out[(j_idx, X_el)]
                break
        else:
            res = X_row[0] # use the 1st columnt index for this cluster id

        for (j_idx, X_el) in enumerate(X_row):
            out[(j_idx, X_el)] = res

        y[i_idx] = res
    return y


class IMatchDuplicates(BaseEstimator):
    """
    Find near duplicates using the randomized I-match backend

    This class aims to expose a scikit-learn compatible API.


    Parameters
    ----------
     - n_rand_lexicons: int, default=1
        number of random lexicons used for duplicate detection
        If equal to 1 no lexicon randomization is used which is equivalent
        to the original I-Match implementation by Chowdhury & Grossman (2002)
     - rand_lexicon_ratio: float, defualt=0.7
        ratio of the vocabulary used in random lexicons

    References
    ----------
     - Kołcz & Chowdhury (2008) - Lexicon randomization for
       near-duplicate detection with I-Match
     - Chowdhury et al. (2002) - 

    """
    def __init__(self, n_rand_lexicons=1, rand_lexicon_ratio=0.7):
        self._fit_X = None
        self.n_rand_lexicons = n_rand_lexicons
        self.rand_lexicon_ratio = rand_lexicon_ratio
        if n_rand_lexicons < 1:
            raise ValueError
        if not ( 0 < rand_lexicon_ratio < 1 ):
            raise ValueError


    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : array_like or sparse (CSR) matrix, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        self : object
            Returns self.
        """
        from sklearn.utils.murmurhash import murmurhash3_bytes_u32
        self._fit_X = X = check_array(X, accept_sparse='csr')

        n_samples, n_features = X.shape

        slice_list = [slice(None)] # no lexicon randomization

        if self.n_rand_lexicons > 1: # use lexicon randomization
            for _ in range(self.n_rand_lexicons - 1):
                # make a random choice of features that will be used
                slice_list.append(np.random.choice(n_features,
                                    int(self.rand_lexicon_ratio*n_features),
                                    replace=False))

        ihash_all = []
        for islice in slice_list:
            X_sl = X[:, islice]
            ihash = []
            for irow in range(n_samples):
                # compute features hash, don't use a random seed
                hash_res = murmurhash3_bytes_u32(X_sl[irow].indices.tobytes(),
                                                 seed=0)
                ihash.append(hash_res)
            ihash_all.append(ihash)

        self.hash_ = np.array(ihash_all).T
        self.hash_is_dup_ =  np.array([ counts > 0 for _, counts in 
                                           [np.unique(col, return_counts=True)
                                               for col in self.hash_.T]]).T
        self.labels_ = _merge_clusters(self.hash_)
