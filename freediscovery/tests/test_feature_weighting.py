
from itertools import product
from unittest import SkipTest

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils.sparsefuncs_fast import csr_row_norms

from freediscovery.feature_weighting import feature_weighting, FeatureWeightingTransformer
from freediscovery.feature_weighting import _validate_smart_notation

documents = ["Shipment of gold damaged in aa fire.",
             "Delivery of silver arrived in aa silver truck.",
             "Shipment of gold arrived in aa truck.",
             "short sentence"
             ]


@pytest.mark.parametrize('scheme, array_type', product(("".join(el)
                                    for el in product('nlabL', 'ntp', 'ncpu')),
                                    ['sparse']))
def test_smart_feature_weighting(scheme, array_type):
    tf = CountVectorizer().fit_transform(documents)
    if array_type == 'sparse':
        pass
    elif array_type == 'dense':
        tf = tf.A
        raise SkipTest
    else:
        raise ValueError
    X = feature_weighting(tf, scheme)


    _, scheme_d, _ = _validate_smart_notation(scheme)

    if scheme_d != 'p':
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

    if X_ref is not None:
        assert_allclose(X.A, X_ref.A)


@pytest.mark.parametrize('weighting', ['nnp', 'nnu'])
def test_pivoted_normalization(weighting):
    tf = CountVectorizer().fit_transform(documents)
    X_ref = feature_weighting(tf, 'nnc')
    if weighting == 'nnp':
        # pivoted cosine normalization == cosine normalization
        # when alpha == 1.0
        X = feature_weighting(tf, 'nnp', alpha=1.0)
        assert_allclose(X.A, X_ref.A)
    X = feature_weighting(tf, weighting, alpha=0.75)
    # shorter documents (last one) gets normalized by a larger value
    assert (X[-1].data < X_ref[-1].data).all()


def test_sublinear_normalization():
    from scipy.sparse import csr_matrix

    tf = csr_matrix([[0, 1, 1, 2],
                     [1, 1, 1, 0]])
    tfl = feature_weighting(tf, 'lnn')
    tfl = feature_weighting(tf, 'ltn')




def test_feature_weighting_transformer():

    raise SkipTest
    try:
        from sklearn.utils.estimator_checks import check_estimator
    except ImportError:
        raise SkipTest
    check_estimator(FeatureWeightingTransformer)
