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


class LSI(_BaseWrapper):
    """Document categorization using Latent Semantic Indexing (LSI)

    Parameters
    ----------
    cache_dir : str
       folder where the model will be saved
    dsid : str
       dataset id
    mid : str
       LSI model id (the dataset id will be inferred)
    verbose : bool, optional
       print progress messages
    """

    _DIRREF = "lsi"

    def __init__(self, cache_dir='/tmp/', dsid=None, mid=None, verbose=False):
        if dsid is None and mid is not None:
            self.dsid = dsid =  self.get_dsid(cache_dir, mid)
            self.mid = mid
        elif dsid is not None:
            self.dsid  = dsid
            self.mid = None
        elif dsid is None and mid is None:
            raise ValueError

        self.data_dir = None
        self.verbose = verbose

        self.fe = FeatureVectorizer(cache_dir=cache_dir, dsid=dsid)

        self.model_dir = os.path.join(self.fe.cache_dir, dsid, self._DIRREF)

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        if self.mid is not None:
            pars = self._load_pars(self.mid)
        else:
            pars = None
        self._pars = pars

    def transform(self, n_components, n_iter=5):
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
        lsi : BaseEstimator
           the TruncatedSVD object
        exp_var : float
           the explained variance of the SVD decomposition
        """

        dsid = self.dsid

        dsid_dir = self.fe.dsid_dir
        if not os.path.exists(dsid_dir):
            raise IOError

        pars = {'dsid': dsid, 'n_components': n_components}

        mid_dir_base = os.path.join(dsid_dir, "lsi")
    
        if not os.path.exists(mid_dir_base):
            os.mkdir(mid_dir_base)
        mid, mid_dir = setup_model(mid_dir_base)

        joblib.dump(pars, os.path.join(mid_dir, 'pars'), compress=9)
        ds = joblib.load(os.path.join(dsid_dir, 'features'))

        svd = _TruncatedSVD_LSI(n_components=n_components,
                                n_iter=n_iter #, algorithm='arpack'
                               )
        lsi = svd
        lsi.fit(ds)

        joblib.dump(lsi, os.path.join(mid_dir, 'lsi_decomposition'))

        exp_var = lsi.explained_variance_ratio_.sum()
        self.mid = mid

        return lsi, exp_var

    def predict(self, index, y, method='nearest-neighbor-1', chunk_size=100):
        """
        Predict the document relevance using a previously trained LSI model

        Parameters
        ----------
        index : array-like, shape (n_samples)
           document indices of the training set
        y : array-like, shape (n_samples)
           target binary class relative to index
        method : str, optional, default='nearest-neighbor-1'
           if `method=="nearest-neighbor-1"` the cosine distance to the closest relevant/non relevant document is used as classification score,
           otherwise if `method=="nearest-centroid"` the centroid of relevant documents is used as the query vector.

        """
        if method in ['nearest-centroid', 'nearest-neighbor-1']:
            pass
        elif method in ['nearest-neighbor-diff', 'nearest-neighbor-combine', 'stacking']:
            raise WrongParameter('method = {} is implemented but is not production ready and was disabled for v0.5 release'.format(method))
        else:
            raise NotImplementedFD()

        index = np.asarray(index, dtype='int')
        y = np.asarray(y, dtype='int')

        _, ds = self.fe.load(self.dsid)  #, mmap_mode='r')

        lsi = joblib.load(os.path.join(self.model_dir, self.mid, 'lsi_decomposition'))

        d_p = lsi.transform_lsi_norm(ds)

        if method.startswith('nearest-neighbor'):
            cmod = NearestNeighborRanker(n_jobs=-1) # euclidean metric by default

            cmod.fit(d_p[index], y)
            D, _, md = cmod.kneighbors(d_p)
            y_test = D[:]
            y_train = D[index]

        elif method == 'nearest-centroid':
            from sklearn.neighbors import NearestCentroid
            cmod  = NearestCentroid()
            cmod.fit(d_p[index], y)
            y_test = cmod.predict(d_p)
            # other optional return values
            md = {}
        else:
            raise NotImplementedFD('method={} not supported!'.format(method))

        y_train = y_test[index]

        return (lsi, y_train, y_test, md)

    def list_models(self):
        lsi_path = os.path.join(self.fe.dsid_dir, 'lsi')
        out = []
        if not os.path.exists(lsi_path):
            return out
        for mid in os.listdir(lsi_path):
            try:
                pars = self._load_pars(mid)
                out.append(pars)
            except:
                raise
        return out

    def _load_pars(self, mid):
        """ Load LSI parameters from disk"""
        lsi_path = os.path.join(self.model_dir, mid)
        if not os.path.exists(lsi_path):
            raise ValueError('Model id {} not found in the cache {}!'.format(mid, lsi_path))
        pars = joblib.load(os.path.join(lsi_path, 'pars'))
        pars['id'] = mid
        return pars

    def load(self, mid):
        """ Load the computed features from cache specified by mid """
        if self.fe.cache_dir is None:
            raise ValueError('cache_dir is None: cannot load from cache!')
        mid_dir = self.get_path(mid)
        model = joblib.load(os.path.join(mid_dir, 'lsi_decomposition'))
        return model


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
