
from itertools import product
from unittest import SkipTest

import pytest
import numpy as np
from numpy.testing import assert_allclose
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils.sparsefuncs_fast import csr_row_norms

from freediscovery.feature_weighting import _smart_tfidf, SmartTfidfTransformer
from freediscovery.feature_weighting import _validate_smart_notation

documents = ["Shipment of gold damaged in aa fire.",
             "Delivery of silver arrived in aa silver truck.",
             "Shipment of gold arrived in aa truck.",
             "short sentence"
             ]


# all division by zeros should be explicitly handled
@pytest.mark.parametrize('scheme, array_type', product(("".join(el)
                         for el in product('nlabLd', 'ntspd',
                                           ['n', 'c', 'l', 'u',
                                            'cp', 'lp', 'up'])),
                        ['sparse']))
@pytest.mark.filterwarnings('error')
def test_feature_weighting_empty_document(scheme, array_type):
    documents_new = documents + ['']
    tf = CountVectorizer().fit_transform(documents_new)
    # check that all weightings preserve zeros rows (with no tokens)
    # and that no overflow warnings are raised
    X = _smart_tfidf(tf, scheme)
    assert_allclose(X.A[-1], np.zeros(tf.shape[1]))


@pytest.mark.parametrize('weighting', ['nncp', 'nnlp', 'nnup'])
def test_pivoted_normalization(weighting):
    tf = CountVectorizer().fit_transform(documents)
    X_ref = _smart_tfidf(tf, 'nnc')
    if weighting == 'nncp':
        # pivoted cosine normalization == cosine normalization
        # when alpha == 1.0
        X = _smart_tfidf(tf, 'nncp', norm_alpha=1.0)
        assert_allclose(X.A, X_ref.A)
    X = _smart_tfidf(tf, weighting, norm_alpha=0.75)
    # shorter documents (last one) gets normalized by a larger value
    assert (X[-1].data < X_ref[-1].data).all()


def test_smart_tfidf_transformer_compatibility():
    raise SkipTest

    try:
        from sklearn.utils.estimator_checks import check_estimator
    except ImportError:
        raise SkipTest
    check_estimator(SmartTfidfTransformer)


@pytest.mark.parametrize('scheme', ("".join(el)
                                    for el in product('nlabLd', 'ntspd',
                                                      ['n', 'c', 'l', 'u',
                                                       'cp', 'lp', 'up']))
                         )
def test_smart_tfidf_transformer(scheme):
    tf = CountVectorizer().fit_transform(documents)

    estimator = SmartTfidfTransformer(weighting=scheme)

    X = estimator.fit_transform(tf)

    scheme_t, scheme_d, scheme_n = _validate_smart_notation(scheme)
    if scheme_d not in 'dp':
        # the resulting document term matrix should be positive
        # (unless we use probabilistic idf weighting)
        assert (X.A >= 0).all()

    # norm cannot be zero
    X_norm = csr_row_norms(X)
    assert (X_norm > 0).all()

    X_ref = None
    if scheme == 'nnn':
        X_ref = X
    elif scheme == 'nnc':
        X_ref = TfidfVectorizer(use_idf=False, smooth_idf=False).fit_transform(documents)
    elif scheme == 'ntc':
        X_ref = TfidfVectorizer(use_idf=True, smooth_idf=False).fit_transform(documents)
    elif scheme == 'lnn':
        X_ref = TfidfVectorizer(use_idf=False, sublinear_tf=True,
                                smooth_idf=False, norm=None).fit_transform(documents)
    elif scheme == 'ltc':
        X_ref = TfidfVectorizer(use_idf=True, sublinear_tf=True,
                                smooth_idf=False).fit_transform(documents)
    elif scheme == 'ltl':
        X_ref = TfidfVectorizer(use_idf=True, sublinear_tf=True,
                                smooth_idf=False, norm='l1').fit_transform(documents)

    if X_ref is not None:
        assert_allclose(X.A, X_ref.A, rtol=1e-7, atol=1e-6)

    assert len(estimator.dl_) == tf.shape[0]
    assert len(estimator.du_) == tf.shape[0]
    if scheme_d in ['tsp']:
        assert len(estimator.df_) == tf.shape[1]

    X_2 = SmartTfidfTransformer(weighting=scheme).fit(tf).transform(tf)
    assert_allclose(X.A, X_2.A, rtol=1e-6, atol=1e-6)

    if scheme_d in 'stp':
        assert estimator.df_ is not None

    sl = slice(2)
    tf_w_sl = estimator.transform(tf[sl])
    assert_allclose(X[sl].A, tf_w_sl.A)
