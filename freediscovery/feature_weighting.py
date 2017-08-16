import warnings

import scipy.sparse as sp
import numpy as np

from sklearn.utils.validation import check_array
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X.
    (copied from scikit-learn)
    """
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(sp.csc_matrix(X, copy=False).indptr)


def _document_length(X):
    return X.sum(axis=1)


def _validate_smart_notation(scheme):

    if not isinstance(scheme, str) or len(scheme) != 3:
        raise ValueError('Expected a 3 character long string for scheme, '
                         'got {}'.format(scheme))

    scheme_t, scheme_d, scheme_n = scheme
    if scheme_t not in 'nlabL':
        raise ValueError(('Term frequency weighting {}'
                          'not supported, must be one of nlabL')
                         .format(scheme_t))
    if scheme_d not in 'ntp':
        raise ValueError(('Document frequency weighting {}'
                          'not supported, must be one of ntp')
                         .format(scheme_d))
    if scheme_n not in 'ncbu':
        raise ValueError(('Document normalization {}'
                          'not supported, must be one of ncbu')
                         .format(scheme_n))
    if scheme_n not in 'nc':
        raise NotImplementedError(
                   ('Document normalization {}'
                    'is not yet implemented, must be one of nt')
                   .format(scheme_n))
    return scheme_t, scheme_d, scheme_n

class FeatureWeightingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, weighting='nnc'):
        """Apply document term weighting and normalization on the extracted
        text features

        weighting : str
          the SMART notation for document, term weighting and normalization.
          In the form [nlabL][ntp][ncb] , see
          https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System
        """
        _validate_smart_notation(weighting)
        self.weighting = weighting
        self._df = None
        self._dl = None

    def fit(self, X, y=None):
        """Learn the document lenght and document frequency vector
        (if necessary).

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        X = check_array(X, ['csr', 'csc', 'coo'])
        self._dl = _document_length(X)
        scheme_t, scheme_d, scheme_n = _validate_smart_notation(self.weighting)
        if scheme_d in 'tp':
            self._df = _document_frequency(X)
        self._n_features = X.shape[1]
        return self

    def transform(self, X, y=None):
        """Apply document term weighting and normalization on text features

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.
        """
        X = check_array(X, ['csr', 'csc', 'coo'])
        check_is_fitted(self, '_dl', 'vector is not fitted')
        if X.shape[1] != self._n_features:
            raise ValueError(('Model fitted with n_features={} '
                              'but X.shape={}').format(self._n_features, X.shape))

        return feature_weighting(X, self.weighting, self._df)


def feature_weighting(tf, weighting, df=None):
    """
    Weight a vector space model following the SMART notation.


    Parameters
    ----------

    df : sparse csr array
      the term frequency matrix (n_documents, n_features)

    weighting : str
      the SMART notation for document term weighting and normalization.
      In the form [nlabL][ntp][ncb] , see
      https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System

    df : dense ndarray (optional)
      precomputed inverse document frequency matrix (n_samples,).
      If not provided, it will be recomputed if necessary.

    Returns
    -------

    X : sparse csr array
      the weighted term frequency matrix

    References
    ----------

    1. Manning, Christopher D.; Raghavan, Prabhakar; Schütze, Hinrich (2008),
       `"Document and query weighting schemes"
       <https://nlp.stanford.edu/IR-book/html/htmledition/document-and-query-weighting-schemes-1.html>`_
    """

    tf = check_array(tf, ['csr', 'csc', 'coo'])
    if df is not None:
        df = check_array(df, ensure_2d=False)

    n_samples, n_features = tf.shape

    scheme_t, scheme_d, scheme_n = _validate_smart_notation(weighting)

    X = tf

    if scheme_t == 'n':
        pass
    elif scheme_t == 'l':
        X.data = 1 + np.log(tf.data)
    elif scheme_t == 'a':
        max_tf = 1. / np.squeeze(tf.max(axis=1).A)
        _max_tf_diag = sp.spdiags(max_tf, diags=0, m=n_samples,
                                  n=n_samples, format='csr')
        X = 0.5 * _max_tf_diag.dot(tf)
        X.data += 0.5

    elif scheme_t == 'b':
        X.data = tf.data.astype('bool').astype('int')
    elif scheme_t == 'L':
        mean_tf = 1. / (1 + np.log(np.squeeze(tf.mean(axis=1).A)))
        _mean_tf_diag = sp.spdiags(mean_tf, diags=0, m=n_samples,
                                   n=n_samples, format='csr')

        X.data = (1 + np.log(tf.data))
        X = _mean_tf_diag.dot(X)
    else:
        raise ValueError

    if scheme_d == 'n':
        pass
    elif scheme_d in 'tp':
        if df is None:
            df = _document_frequency(tf)
        if scheme_d == 't':
            idf = np.log(float(n_samples) / df) + 1.0
        elif scheme_d == 'p':
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                                        message="divide by zero encountered in log",
                                        category=RuntimeWarning)
                idf = np.fmax(0, np.log((float(n_samples) - df)/df))
        _idf_diag = sp.spdiags(idf, diags=0, m=n_features,
                               n=n_features, format='csr')
        X = X * _idf_diag
    else:
        raise ValueError

    if scheme_n == 'n':
        pass
    elif scheme_n == 'c':
        X = normalize(X, norm="l2", copy=False)
    else:
        raise ValueError
    return X
