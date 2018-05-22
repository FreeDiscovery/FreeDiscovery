# -*- coding: utf-8 -*-
#
# Authors: Roman Yurchak
#
# License: BSD 3 clause

import warnings

import scipy.sparse as sp
import numpy as np

from sklearn.utils.validation import check_array
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.sparsefuncs_fast import csr_row_norms
from sklearn.exceptions import DataConversionWarning

from .utils import _mean_csr_nonzero_axis1


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

    if not isinstance(scheme, str) or len(scheme) not in [3, 4]:
        raise ValueError('Expected a 3 or 4 character long string for scheme, '
                         'got {}'.format(scheme))

    if len(scheme) == 3:
        scheme_t, scheme_d, scheme_n = scheme
    elif len(scheme) == 4:
        scheme_t, scheme_d, scheme_n1, scheme_n2 = scheme
        scheme_n = ''.join([scheme_n1, scheme_n2])

    if scheme_t not in 'nlabLd':
        raise ValueError(('Term frequency weighting {} '
                          'not supported, must be one of nlabL')
                         .format(scheme_t))
    if scheme_d not in 'ntspd':
        raise ValueError(('Document frequency weighting {} '
                          'not supported, must be one of ntp')
                         .format(scheme_d))
    if scheme_n not in ['n', 'c', 'l', 'u', 'cp', 'lp', 'up', 'b']:
        raise ValueError(('Document normalization {} '
                          'not supported, must be of the form [nclub][p]?')
                         .format(scheme_n))
    if scheme_n not in ['n', 'c', 'l', 'u', 'cp', 'lp', 'up', ]:
        raise NotImplementedError(
                   ('Document normalization {}'
                    'is not yet implemented, must be of the form [nclu][p]?')
                   .format(scheme_n))
    return scheme_t, scheme_d, scheme_n


class SmartTfidfTransformer(BaseEstimator, TransformerMixin):
    """TF-IDF weighting and normalization with the SMART IR notation

    This class is similar to
    :class:`sklearn.feature_extraction.text.TfidfTransformer` but supports
    a larger number of TF-IDF weighting and normalization schemes.
    It should be fitted on the document-term matrix computed by
    :class:`sklearn.feature_extraction.text.CountVectorizer`.

    The TF-IDF transform consists of three subsequent operations, determined
    by the ``weighting`` parameter,

    1. Term frequency weighing:

       natural (``n``), log (``l``), augmented
       (``a``),  boolean (``b``), log average (``L``)

    2. Document frequency weighting:

       none (``n``), idf (``t``), smoothed
       idf (``s``),  probabilistic (``p``), smoothed probabilistic (``d``)

    3. Document normalization:

       none (``n``), cosine (``c``), length (``l``),
       unique (``u``).

    Following the SMART IR notation, the ``weighting`` parameter is written
    as the concatenation of thee characters describing each processing step.
    In addition the pivoted normalization can be enabled with a fourth
    character ``p``.


    See the :ref:`tfidf_section` documentation section for more details.


    Parameters
    ----------
    weighting : str, default='nsc'
      the SMART notation for document, term weighting and normalization.
      In the form ``[nlabL][ntspd][ncb][p]``.
    norm_alpha : float, default=0.75
      the α parameter in the pivoted normalization. This parameter is only
      used when ``weighting='???p'``.
    norm_pivot : float, default=None
      the pivot value used for the normalization. If not provided
      it is computed as the mean of the ``norm(tf*idf)``. This parameter is
      only used when ``weighting='???p'``.
    compute_df : bool, default=False
      compute the document frequency (``df_`` attribute) even when it's not
      explicitly required by the weighting scheme.
    copy : boolean, default=True
      Whether to copy the input array and operate on the copy or perform
      in-place operations in fit and transform.


    See also
    --------
    :class:`sklearn.feature_extraction.text.TfidfTransformer`


    References
    ----------
    .. [Manning2008] C.D. Manning, P. Raghavan, H. Schütze,
       `"Document and query weighting schemes"
       <https://nlp.stanford.edu/IR-book/html/htmledition/document-and-query-weighting-schemes-1.html>`_ ,
       2008
    .. [Singhal1996] A. Singhal, C. Buckley, and M. Mitra.
       `"Pivoted document length normalization."
       <https://ecommons.cornell.edu/bitstream/handle/1813/7217/95-1560.pdf?sequence=1>`_ , 1996
    """ # noqa
    def __init__(self, weighting='nsc', norm_alpha=0.75, norm_pivot=None,
                 compute_df=False, copy=True):
        _validate_smart_notation(weighting)
        self.weighting = weighting
        self.norm_alpha = norm_alpha
        self.norm_pivot = norm_pivot
        self.compute_df = compute_df
        self.copy = copy

    def fit(self, X, y=None):
        """Learn the document lenght and document frequency vector
        (if necessary).

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        X = check_array(X, ['csr'], copy=self.copy)
        scheme_t, scheme_d, scheme_n = _validate_smart_notation(self.weighting)
        self.dl_ = _document_length(X)
        if scheme_d in 'stp' or self.compute_df:
            self.df_ = _document_frequency(X)
        else:
            self.df_ = None
        if sp.isspmatrix_csr(X):
            self.du_ = np.diff(X.indptr)
        else:
            self.du_ = X.shape[-1] - (X == 0).sum(axis=1)
        self._n_features = X.shape[1]

        if self.df_ is not None:
            df_n_samples = len(self.dl_)
        else:
            df_n_samples = None

        if scheme_n.endswith('p') and self.norm_pivot is None:
            # Need to compute the pivot if it's not provided
            _, self.norm_pivot = _smart_tfidf(X, self.weighting, self.df_,
                                              df_n_samples,
                                              norm_alpha=self.norm_alpha,
                                              norm_pivot=self.norm_pivot,
                                              return_pivot=True)

        return self

    def fit_transform(self, X, y=None):
        """Apply document term weighting and normalization on text features

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        X = check_array(X, ['csr'], copy=self.copy)

        scheme_t, scheme_d, scheme_n = _validate_smart_notation(self.weighting)
        self.dl_ = _document_length(X)
        if scheme_d in 'stpd' or self.compute_df:
            self.df_ = _document_frequency(X)
        else:
            self.df_ = None
        if sp.isspmatrix_csr(X):
            self.du_ = np.diff(X.indptr)
        else:
            self.du_ = X.shape[-1] - (X == 0).sum(axis=1)
        self._n_features = X.shape[1]

        if self.df_ is not None:
            df_n_samples = len(self.dl_)
        else:
            df_n_samples = None

        if self.df_ is not None:
            df_n_samples = len(self.dl_)
        else:
            df_n_samples = None

        X, self.norm_pivot = _smart_tfidf(X, self.weighting, self.df_,
                                          df_n_samples,
                                          norm_alpha=self.norm_alpha,
                                          norm_pivot=self.norm_pivot,
                                          return_pivot=True)
        return X

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
        X = check_array(X, ['csr'], copy=self.copy)
        check_is_fitted(self, 'dl_', 'vector is not fitted')
        if X.shape[1] != self._n_features:
            raise ValueError(('Model fitted with n_features={} '
                              'but X.shape={}')
                             .format(self._n_features, X.shape))

        if self.df_ is not None:
            df_n_samples = len(self.dl_)
        else:
            df_n_samples = None

        return _smart_tfidf(X, self.weighting, self.df_,
                            df_n_samples,
                            norm_alpha=self.norm_alpha,
                            norm_pivot=self.norm_pivot)


def _smart_tfidf(tf, weighting, df=None, df_n_samples=None, norm_alpha=0.75,
                 norm_pivot=None, return_pivot=False):
    """
    Apply TF-IDF feature weighting using the SMART notation.


    Parameters
    ----------
    df : sparse csr array
      the term frequency matrix (n_documents, n_features)

    weighting : str, default='nnc'
      the SMART notation for document term weighting and normalization.
      In the form [nlabL][ntspd][nclu][p] , see
      https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System

    df : array, shape=[n_features], optional
      precomputed inverse document frequency matrix (n_features,).
      If not provided, it will be recomputed if necessary. Both df and
      df_n must be provided at the same time.

    df_n_samples : float, default=None
      when using a inverse document frequency matrix, the number of
      documents that were used to compute the df. Both df and df_n
      must be provided at the same time.

    norm_alpha : float, default=0.75
      the alpha parameter in the pivoted normalization. Only used when
      weighting='???p'.

    norm_pivot : float, default=None
      the pivot value used for the normalization. If not provided, and
      weighting='???p', it is computed as the mean of the norm(tf*idf).

    return_pivot : bool, default=False
      return the computed norm_pivot

    Returns
    -------

    X : sparse csr array
      the weighted term frequency matrix

    norm_pivot : flot
      return the norm pivot (only when return_pivot=True)

    References
    ----------
    .. [Manning2008] C.D. Manning, P. Raghavan, H. Schütze,
       `"Document and query weighting schemes"
       <https://nlp.stanford.edu/IR-book/html/htmledition/document-and-query-weighting-schemes-1.html>`_ ,
       2008
    .. [Singhal1996] A. Singhal, C. Buckley, and M. Mitra.
       `"Pivoted document length normalization."
       <https://ecommons.cornell.edu/bitstream/handle/1813/7217/95-1560.pdf?sequence=1>`_ , 1996
    """  # noqa

    tf = check_array(tf, ['csr'])
    if (df is None) != (df_n_samples is None):
        raise ValueError(('df={} and df_n_samples={}, while both should be '
                          'either provided or not provided')
                         .format(df is None, df_n_samples))
    if df is not None:
        df = check_array(df, ensure_2d=False)
        if df.shape[0] != tf.shape[-1]:
            raise ValueError(('df array provided with n_features={} ,'
                              'while in the tf array n_features={}')
                             .format(df.shape[0], tf.shape[1]))

    if not 0 <= norm_alpha <= 1:
        raise ValueError('norm_alpha={} not in [0, 1]'.format(norm_alpha))

    n_samples, n_features = tf.shape
    if df_n_samples is None:
        df_n_samples = n_samples

    scheme_t, scheme_d, scheme_n = _validate_smart_notation(weighting)

    X = tf

    # term weighting
    if scheme_t == 'n':
        pass
    elif scheme_t == 'l':
        X.data = 1 + np.log(tf.data)
    elif scheme_t == 'd':
        X.data = 1 + np.log(1 + np.log(tf.data))
    elif scheme_t == 'a':
        max_tf = np.squeeze(tf.max(axis=1).A)
        # if max_tf is zero, the tf are going to be all zero anyway
        # so we set it to 1 in order to prevent overflows
        max_tf[max_tf == 0] = 1
        _max_tf_diag = sp.spdiags(1. / max_tf, diags=0, m=n_samples,
                                  n=n_samples, format='csr')
        X = 0.5 * _max_tf_diag.dot(tf)
        X.data += 0.5

    elif scheme_t == 'b':
        X.data = tf.data.astype('bool').astype('int')
    elif scheme_t == 'L':
        mean_tf = _mean_csr_nonzero_axis1(tf)
        # if mean_tf is zero, the tf are going to be all zero anyway
        # so we set it to 1 in order to prevent overflows
        mean_tf[mean_tf == 0] = 1.0
        mean_tf = (1 + np.log(mean_tf))
        _mean_tf_diag = sp.spdiags(1./mean_tf, diags=0, m=n_samples,
                                   n=n_samples, format='csr')

        X.data = (1 + np.log(tf.data))
        X = _mean_tf_diag.dot(X)
    else:
        raise ValueError

    # document weighting
    if scheme_d == 'n':
        pass
    elif scheme_d in 'tpsd':
        if df is None:
            df = _document_frequency(tf)
        if scheme_d == 't':
            idf = np.log(float(df_n_samples) / df) + 1.0
        elif scheme_d == 's':
            idf = np.log(float(df_n_samples + 1) / (df + 1)) + 1.0
        elif scheme_d == 'p':
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                                        message="divide by zero encountered in log",  # noqa
                                        category=RuntimeWarning)
                idf = np.log((float(df_n_samples) - df)/df)
        elif scheme_d == 'd':
            idf = np.log((float(df_n_samples) + 1 - df)/(df + 1))
        _idf_diag = sp.spdiags(idf, diags=0, m=n_features,
                               n=n_features, format='csr')
        X = X.dot(_idf_diag)
    else:
        raise ValueError

    # normalization
    if scheme_n == 'n':
        pass
    elif scheme_n == 'c':
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    category=DataConversionWarning)
            X = normalize(X, norm="l2", copy=False)
    elif scheme_n == 'l':
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    category=DataConversionWarning)
            X = normalize(X, norm="l1", copy=False)
    elif scheme_n == 'u':
        X_norm = np.diff(X.indptr)
        X_norm[X_norm == 0] = 1.
        # empty documents (with a zero norm) don't need to be normalized
        _diag_norm = sp.spdiags(1./X_norm, diags=0, m=n_samples,
                                n=n_samples, format='csr')
        X = _diag_norm.dot(X)
    elif scheme_n in ['cp', 'lp', 'up']:
        if scheme_n == 'cp':
            X_norm = np.sqrt(csr_row_norms(X))
        elif scheme_n == 'lp':
            X_data = X.data.copy()
            X.data = np.abs(X.data)
            X_norm = np.squeeze(X.sum(axis=1).A)
            X.data = X_data
        elif scheme_n == 'up':
            X_norm = np.diff(X.indptr)

        if norm_pivot is None:
            norm_pivot = X_norm.mean()

        # empty documents (with a zero norm) don't need to be normalized
        X_norm[X_norm == 0] = 1.

        pivoted_norm = (1 - norm_alpha)*norm_pivot + norm_alpha*X_norm
        _diag_pivoted_norm = sp.spdiags(1./pivoted_norm, diags=0, m=n_samples,
                                        n=n_samples, format='csr')
        X = _diag_pivoted_norm.dot(X)
    else:
        raise ValueError
    if return_pivot:
        return X, norm_pivot
    else:
        return X
