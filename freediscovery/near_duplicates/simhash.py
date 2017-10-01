# Authors: Roman Yurchak
#
# License: BSD 3 clause

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array


class SimhashNearDuplicates(BaseEstimator):
    """Near duplicates detection using the simhash algorithm.

    A classical near-duplicates detection involves comparing all pairs of
    samples in the collection. For a collection of size ``N``, this is
    typically an ``O(N^2)`` operation. Simhash algorithm allows to
    retrieve near duplicates with a significantly better computational
    scaling.

    .. Note:: this estimator requires
              the `simhash-py <https://github.com/seomoz/simhash-py>`_Python package
              to be installed.

    Parameters
    ----------
    hash_func : str or function, default='murmurhash3_int_u32'
        the hashing function used to hash documents.
        Possibles values are "murmurhash3_int_u32" or a custom function.
    hash_func_nbytes : int, default=64
        expected size of the hash produced by hash_func

    References
    ----------
    .. [Charikar2002]  `Charikar, M. S. (2002, May).
                        Similarity estimation techniques from rounding
                        algorithms.
                        In Proceedings of the thiry-fourth annual ACM symposium
                        on Theory of computing (pp. 380-388). ACM.`
    """
    def __init__(self, hash_func='murmurhash3_int_u32', hash_func_nbytes=32):
        self._fit_X = None
        self._fit_shash_dict = {}
        if isinstance(hash_func, str):
            if hash_func == 'murmurhash3_int_u32':
                from sklearn.utils.murmurhash import murmurhash3_int_u32
                hash_func = murmurhash3_int_u32
                hash_func_nbytes = 32
            else:
                raise ValueError
        elif not hasattr(hash_func, '__call__'):
            raise ValueError

        self.hash_func = hash_func
        if hash_func_nbytes not in [32, 64]:
            raise ValueError('Hashing function other than 64bit '
                             'or 32bit are not supported!')

        self.hash_func_nbytes = hash_func_nbytes

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : {array, sparse matrix}, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        self : object
            Returns self.
        """
        from simhash import compute
        self._fit_X = X = check_array(X, accept_sparse='csr')

        n_features = X.shape[1]

        def _scale_hash_32_64bit(indices):
            return indices*((2**64-1)//2**32-1)

        hash_func = self.hash_func

        hashing_table = np.array(
                [hash_func(el, 0) for el in range(n_features)],
                dtype='uint64')

        shash = []
        for idx in range(X.shape[0]):
            # get hashes of indices
            mhash = hashing_table[X[idx].indices]
            if self.hash_func_nbytes == 32:
                mhash = _scale_hash_32_64bit(mhash)
            shash.append(compute(mhash))
        _fit_shash = np.asarray(shash, dtype='uint64')
        self._fit_shash = _fit_shash
        self._fit_shash_dict = {val: key
                                for key, val in enumerate(self._fit_shash)}

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
        from simhash import find_all

        if distance >= 64:
            raise ValueError(('Wrong input parameter for distance = {} '
                              'Must be less than 64!')
                             .format(distance))

        _, cluster_id = np.unique(self._fit_shash, return_inverse=True)

        if blocks == 'auto':
            blocks = min(distance*2, 64)
        matches = find_all(self._fit_shash, blocks, distance)
        matches = np.array(matches, dtype='uint64')
        return self._fit_shash, cluster_id, matches
