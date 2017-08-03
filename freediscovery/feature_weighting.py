import warnings

import scipy.sparse as sp
import numpy as np

from sklearn.utils.validation import check_array
from sklearn.preprocessing import normalize


def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X.
    (copied from scikit-learn)
    """
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(sp.csc_matrix(X, copy=False).indptr)


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


def smart_feature_weighting(tf, scheme, idf=None):
    """
    Weight a vector space model following the SMART notation.


    Parameters
    ----------

    df : sparse csr array
      the term frequency matrix (n_documents, n_features)

    scheme : str
      the SMART notation for document, term weighting and normalization.
      In the form [nlabL][ntp][ncb] , see
      https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System

    idf : sparse csr array (optional)
      precomputed inverse document frequency matrix (n_documents, n_features).
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
    if idf is not None:
        idf = check_array(idf, ['csr', 'csc', 'coo'], ensure_2d=False)

    n_samples, n_features = tf.shape

    scheme_t, scheme_d, scheme_n = _validate_smart_notation(scheme)

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
        if idf is None:
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
