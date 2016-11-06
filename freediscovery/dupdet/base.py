# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array


class SimhashDuplicates(BaseEstimator):
    """
    Find near duplicates using the simhash-py backend

    This class aims to expose a scikit-learn compatible API.
    """
    def __init__(self):
        self._fit_X = None
        self._fit_shash_dict = {}

    def fit(self, X, y=None):  # TODO parameter 'y' is unused
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
        from simhash import compute  # TODO resolve reference
        self._fit_X = X = check_array(X, accept_sparse='csr')

        n_features = X.shape[1]

        def _scale_hash_64bit(indices, n_features):
            return indices.astype('uint64')*((2**64-1)//n_features)

        shash = []
        for idx in range(X.shape[0]):
            mhash = _scale_hash_64bit(X[idx].indices, n_features)
            shash.append(compute(mhash))
        self._fit_shash = np.asarray(shash, dtype='uint64')
        self._fit_shash_dict = {val: key for key, val in enumerate(self._fit_shash)}

    def get_index_by_hash(self, shash):
        """ Get document index by hash

        Parameters
        ----------
        shash: uint64
           a simhash value

        Returns
        -------
        index: int
           a document index
        """
        return self._fit_shash_dict[shash]

    def query(self, distance=2, blocks='auto'):
        """ Find all the nearests neighbours for the dataset

        Parameters
        ----------
        distance : int, default=2
            Maximum number of differnet bits in the simhash
        blocks : int or 'auto', default='auto'
                number of blocks into which the simhash is split
                when searching for duplicates,
                see  https://github.com/seomoz/simhash-py

        Returns
        -------
        simhash : array
            the simhash value for all documents in the collection
        cluster_id : array
            the exact duplicates (documents with the same simhash)
            are grouped by in cluster_id
        dup_pairs : list
            list of tuples for the near-duplicates
        """
        from simhash import find_all  # TODO resolve reference

        if distance >= 64:
            raise ValueError('Wrong input parameter for distance = {}'.format(distance)
                            + 'Must be less than 64!')

        _, cluster_id = np.unique(self._fit_shash, return_inverse=True)

        if blocks == 'auto':
            blocks = min(distance*2, 64)
        matches = find_all(self._fit_shash, blocks, distance)
        matches = np.array(matches)
        return self._fit_shash, cluster_id, matches


