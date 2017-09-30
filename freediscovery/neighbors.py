# Authors: Roman Yurchak
#
# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors, NearestCentroid
from sklearn.utils.validation import check_array
from freediscovery.base import RankerMixin

# a subclass of the NearestCentroid from scikit-learn that also
# includes the distance to the nearest centroid


class NearestCentroidRanker(NearestCentroid):

    def decision_function(self, X):
        """Compute the distances to the nearest centroid for
        an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        C : array, shape = [n_samples]
        """
        from sklearn.metrics.pairwise import pairwise_distances
        from sklearn.utils.validation import check_array, check_is_fitted

        check_is_fitted(self, 'centroids_')

        X = check_array(X, accept_sparse='csr')

        return pairwise_distances(X, self.centroids_,
                                  metric=self.metric).min(axis=1)


def _chunk_kneighbors(func, X, batch_size=5000, **args):
    """ Chunk kneighbors computations to reduce RAM requirements

    Parameters
    ----------
    func : function
      the function to run
    X : ndarray
      the array func is applied to
    batch_size : int
      batch size

    Returns
    -------
    dist : array
       distance array
    ind : array of indices
    """
    n_samples = X.shape[0]
    ind_arr = []
    dist_arr = []
    # don't enter the last loop if n_sampes is a multiple of batch_size
    for k in range(n_samples//batch_size + int(n_samples % batch_size != 0)):
        mslice = slice(k*batch_size, min((k+1)*batch_size, n_samples))
        X_sl = X[mslice, :]
        dist_k, ind_k = func(X_sl, **args)
        ind_arr.append(ind_k)
        dist_arr.append(dist_k)
    return (np.concatenate(dist_arr, axis=0), np.concatenate(ind_arr, axis=0))


class NearestNeighborRanker(BaseEstimator, RankerMixin):
    """A nearest neighbor ranker.

    The distance is returned in terms of cosine_similarity

    Parameters
    ----------
    radius : float, optional (default = 1.0)
        Range of parameter space to use by default for :meth:`radius_neighbors`
        queries.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDtree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    method : str, def
        If "unsupervised" only distances to the positive samples are used
        in the ranking. If "supervised" both the distance to the positive
        and negative documents are used for ranking (i.e. if a document
        is slightly further away from a positive document than from a negative
        one, it will be considered negative with a very low score)

    """

    def __init__(self, radius=1.0, algorithm='brute', leaf_size=30, n_jobs=1):

        self.algorithm = algorithm
        self.radus = radius
        self.n_jobs = n_jobs
        self.leaf_size = leaf_size

    def fit(self, X, y):
        """Fit the model using X as training data
        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training data, shape [n_samples, n_features],

        """
        X = check_array(X, accept_sparse='csr')
        y = np.asarray(y, dtype='int')
        y_unique = np.unique(y)

        index = np.arange(len(y), dtype='int')

        if len(y_unique) == 0:
            raise ValueError('The training set must have at least '
                             'one document category!')

        # define nearest neighbors search objects for each category
        self._mod = [NearestNeighbors(n_neighbors=1,
                                      leaf_size=self.leaf_size,
                                      algorithm=self.algorithm,
                                      n_jobs=self.n_jobs,
                                      # euclidean metric by default
                                      metric='cosine',
                                      ) for el in range(len(y_unique))]

        index_mapping = []
        for imod, y_val in enumerate(y_unique):
            mask = (y == y_val)
            index_mapping.append(index[mask])
            self._mod[imod].fit(X[mask])

        self.index_mapping = index_mapping

    def kneighbors(self, X=None, batch_size=5000):
        """Finds the K-neighbors of a point.
        Returns indices of and distances to the neighbors of each point.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            the input array
        batch_size : int
            the batch size
        Returns
        -------
        S_cos : array [n_samples, n_categories]
            the cosine similarity to closest point in each category
        indices : array [n_samples, n_categories]
            Indices of the nearest points in the population matrix.
        --------
        """
        X = check_array(X, accept_sparse='csr')

        n_classes = len(self._mod)

        S_res = np.zeros((X.shape[0], n_classes), dtype='float')
        nn_idx_res = np.zeros((X.shape[0], n_classes), dtype='int')

        for imod in range(n_classes):
            D_i, nn_idx_i_loc = _chunk_kneighbors(self._mod[imod].kneighbors,
                                                  X,
                                                  batch_size=batch_size)

            # only NearestNeighbor-1 (only one column in the kneighbors output)
            # convert from eucledian distance in L2 norm space to cosine
            # similarity
            # S_cos = seuclidean_dist2cosine_sim(D_i[:,0])
            S_res[:, imod] = 1 - D_i[:, 0]
            # map local index within index_mapping to global index
            nn_idx_res[:, imod] = self.index_mapping[imod][nn_idx_i_loc[:, 0]]

        return S_res, nn_idx_res
