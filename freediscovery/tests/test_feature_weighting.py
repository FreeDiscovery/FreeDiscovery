
from itertools import product
from unittest import SkipTest

import numpy as np
from numpy.testing import assert_allclose
import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from freediscovery.feature_weighting import feature_weighting, FeatureWeightingTransformer

documents = ["Shipment of gold damaged in aa fire.",
             "Delivery of silver arrived in aa silver truck.",
             "Shipment of gold arrived in aa truck.",
             ]


@pytest.mark.parametrize('scheme', ("".join(el)
                                    for el in product('nlabL', 'ntp', 'nc')))
def test_smart_feature_weighting(scheme):
    tf = CountVectorizer().fit_transform(documents)

    X = feature_weighting(tf, scheme)

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


def test_feature_weighting_transformer():

    try:
        from sklearn.utils.estimator_checks import check_estimator
    except ImportError:
        raise SkipTest
    check_estimator(FeatureWeightingTransformer)
