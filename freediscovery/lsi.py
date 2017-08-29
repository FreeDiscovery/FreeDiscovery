# -*- coding: utf-8 -*-
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.utils import check_array, as_float_array, check_random_state
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, svd_flip
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.decomposition import TruncatedSVD


def _compute_lsi_dimensionality(n_components, n_samples, n_features,
                                alpha=0.33):
    """ Reduce the number of LSI components for small datasets """
    n_components_samples = min(n_components, alpha*n_samples)
    msg = []
    if n_components_samples < n_components:
        msg.append(('The ingested dataset has only {} documents while {} LSI '
                    'components were requested; '
                    'decreasing the number of LSI components: ')
                   .format(n_samples, n_components))

    n_components_feature = int(min(n_components, alpha*n_features))
    if n_components_feature < n_components:
        msg.append(('The vocabulary in the ingested dataset has '
                    'only {} words (or n-grams) while {} LSI '
                    'components were requested; '
                    'decreasing the number of LSI components: ')
                   .format(n_features, n_components))
    n_components_opt = int(min(n_components_samples, n_components_feature))
    n_components_opt = max(5, n_components_opt)
    if n_components_opt < n_components:
        msg.append('Decreasing n_components from {} to {}'
                   .format(n_components, n_components_opt))
    if msg:
        msg = '\n'.join(msg)
        warnings.warn(msg)
    return n_components_opt


# The below class is identical to TruncatedSVD,
# the only reason is the we need to save the Sigma matrix when
# performing this transform!
# This will not longer be necessary with sklearn v0.19

class _TruncatedSVD_LSI(TruncatedSVD):
    """
    A patch of `sklearn.decomposition.TruncatedSVD` to include whitening
    (`scikit-learn/scikit-learn#7832)`
    """

    def transform_lsi(self, X):
        """ LSI transform, normalized by the inverse of the eigen values"""
        X = check_array(X, accept_sparse='csr')
        return safe_sparse_dot(X, self.components_.T).dot(
                 np.diag(1./self.Sigma))

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
