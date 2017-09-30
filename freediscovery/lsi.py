# Authors: Roman Yurchak
#
# License: BSD 3 clause

import warnings

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot


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


class _TruncatedSVD_LSI(TruncatedSVD):
    """
    A patch of `sklearn.decomposition.TruncatedSVD` to include whitening
    (`scikit-learn/scikit-learn#7832)`
    """

    def transform_lsi(self, X):
        """ LSI transform, normalized by the inverse of the eigen values"""
        X = check_array(X, accept_sparse='csr')
        return safe_sparse_dot(X, self.components_.T).dot(
                    np.diag(1./self.singular_values_[:self.n_components]))

    def transform_lsi_norm(self, X):
        Y = self.transform_lsi(X)
        normalize(Y, copy=False)
        return Y
