# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.externals import joblib
from sklearn.preprocessing import normalize
from sklearn.utils import check_array, as_float_array, check_random_state
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, svd_flip
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.decomposition import TruncatedSVD

from .text import FeatureVectorizer
from .base import _BaseWrapper
from .categorization import NearestNeighborRanker
from .utils import setup_model
from .exceptions import (WrongParameter, NotImplementedFD)


def _touch(filename):
    open(filename, 'ab').close()


class _LSIWrapper(_BaseWrapper):
    """Document categorization using Latent Semantic Indexing (LSI)

    Parameters
    ----------
    cache_dir : str
       folder where the model will be saved
    parent_id : str
       dataset id
    mid : str
       LSI model id (the dataset id will be inferred)
    verbose : bool, optional
       print progress messages
    """

    _wrapper_type = "lsi"

    def __init__(self, cache_dir='/tmp/', parent_id=None, mid=None, verbose=False):

        super(_LSIWrapper, self).__init__(cache_dir=cache_dir,
                                          parent_id=parent_id, mid=mid)


    def fit_transform(self, n_components=150, n_iter=5):
        """
        Perform the SVD decomposition
        
        Parameters
        ----------
        n_components : int
           number of selected singular values (number of LSI dimensions)
        n_iter : int
           number of iterations for the stochastic SVD algorithm

        Returns
        -------
        mid : str
           model id
        lsi : _BaseWrapper
           the TruncatedSVD object
        exp_var : float
           the explained variance of the SVD decomposition
        """

        parent_id = self.pipeline.mid

        dsid_dir = self.fe.dsid_dir
        if not os.path.exists(dsid_dir):
            raise IOError

        pars = {'parent_id': parent_id, 'n_components': n_components}

        mid_dir_base = os.path.join(dsid_dir, self._wrapper_type)
    
        if not os.path.exists(mid_dir_base):
            os.mkdir(mid_dir_base)
        mid, mid_dir = setup_model(mid_dir_base)


        ds = self.pipeline.data
        svd = _TruncatedSVD_LSI(n_components=n_components,
                                n_iter=n_iter #, algorithm='arpack'
                               )
        lsi = svd
        lsi.fit(ds)

        ds_p = lsi.transform_lsi_norm(ds)

        joblib.dump(pars, os.path.join(mid_dir, 'pars'), compress=9)
        joblib.dump(lsi, os.path.join(mid_dir, 'model'))
        joblib.dump(ds_p, os.path.join(mid_dir, 'data'))

        exp_var = lsi.explained_variance_ratio_.sum()
        self.mid = mid

        return lsi, exp_var


# The below class is identical to TruncatedSVD,
# https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/decomposition/truncated_svd.py#L25
# the only reason is the we need to save the Sigma matrix when performing this transform!
# This will not longer be necessary with sklearn v0.19

class _TruncatedSVD_LSI(TruncatedSVD):
    """
    A patch of `sklearn.decomposition.TruncatedSVD` to include whitening (`scikit-learn/scikit-learn#7832)`
    """

    def transform_lsi(self, X):
        """ LSI transform, normalized by the inverse of the eigen values"""
        X = check_array(X, accept_sparse='csr')
        return safe_sparse_dot(X, self.components_.T).dot(np.diag(1./self.Sigma))

    def transform_lsi_norm(self, X):
        Y = self.transform_lsi(X)
        normalize(Y, copy=False)
        return Y

    def fit_transform(self, X, y=None):
        """ Fit LSI model to X and perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.

        Returns
        -------

        X_new : array, shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        X = as_float_array(X, copy=False)
        random_state = check_random_state(self.random_state)

        # If sparse and not csr or csc, convert to csr
        if sp.issparse(X) and X.getformat() not in ["csr", "csc"]:
            X = X.tocsr()

        if self.algorithm == "arpack":
            U, Sigma, VT = svds(X, k=self.n_components, tol=self.tol)
            # svds doesn't abide by scipy.linalg.svd/randomized_svd
            # conventions, so reverse its outputs.
            Sigma = Sigma[::-1]
            U, VT = svd_flip(U[:, ::-1], VT[::-1])

        elif self.algorithm == "randomized":
            k = self.n_components
            n_features = X.shape[1]
            if k >= n_features:
                raise ValueError("n_components must be < n_features;"
                                 " got %d >= %d" % (k, n_features))
            U, Sigma, VT = randomized_svd(X, self.n_components,
                                          n_iter=self.n_iter,
                                          random_state=random_state)
        else:
            raise ValueError("unknown algorithm %r" % self.algorithm)

        self.components_ = VT
        self.Sigma = Sigma[:self.n_components]

        # Calculate explained variance & explained variance ratio
        X_transformed = np.dot(U, np.diag(Sigma))
        self.explained_variance_ = exp_var = np.var(X_transformed, axis=0)
        if sp.issparse(X):
            _, full_var = mean_variance_axis(X, axis=0)
            full_var = full_var.sum()
        else:
            full_var = np.var(X, axis=0).sum()
        self.explained_variance_ratio_ = exp_var / full_var
        return X_transformed
