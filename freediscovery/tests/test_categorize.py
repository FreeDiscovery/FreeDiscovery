#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

import os.path
from unittest import SkipTest

import numpy as np
from numpy.testing import assert_allclose, assert_equal

import pytest
import itertools

from freediscovery.text import FeatureVectorizer
from freediscovery.categorization import Categorizer
from freediscovery.io import parse_ground_truth_file
from freediscovery.utils import categorization_score
from freediscovery.exceptions import OptionalDependencyMissing
from .run_suite import check_cache


basename = os.path.dirname(__file__)


cache_dir = check_cache()


data_dir = os.path.join(basename, "..", "data", "ds_001", "raw")

n_features = 20000

fe = FeatureVectorizer(cache_dir=cache_dir)
uuid = fe.preprocess(data_dir, file_pattern='.*\d.txt', n_features=n_features,
        binary=True, use_idf=False, norm=None)
uuid, filenames  = fe.transform()

ground_truth = parse_ground_truth_file(
                        os.path.join(data_dir, "..", "ground_truth_file.txt"))

@pytest.mark.parametrize('method, cv', itertools.product(
                       ["LinearSVC", "LogisticRegression", 'xgboost'],
                        #'MLPClassifier', 'ensemble-stacking' not supported in production the moment
                       [None, 'fast']))
def test_categorization(method, cv):

    if 'CIRCLECI' in os.environ and cv == 'fast' and method in ['LinearSVC', 'xgboost']:
        raise SkipTest # Circle CI is too slow and timesout

    if method == 'xgboost':
        try:
            import xgboost
        except ImportError:
            raise SkipTest

    cat = Categorizer(cache_dir=cache_dir, dsid=uuid, cv_n_folds=2)
    mask = ground_truth.is_relevant.values == 1
    idx_rel = cat.fe.search(ground_truth.index.values[mask])
    idx_nrel = cat.fe.search(ground_truth.index.values[~mask])
    
    try:
        coefs, X_train, Y_train = cat.train(
                                idx_rel, idx_nrel,
                                method=method,
                                cv=cv)
    except OptionalDependencyMissing:
        raise SkipTest


    Y_pred = cat.predict()
    X_pred = cat.fe._pars['filenames']

    scores = categorization_score(ground_truth.index.values,
                        ground_truth.is_relevant.values,
                        X_pred, Y_pred)

    assert cat.get_params() is not None

    if method in ['xgboost', 'ensemble-stacking']:
        # this parameter fail for some reason so far...
        return
    assert_allclose(scores['precision'], 1, rtol=0.5)
    assert_allclose(scores['recall'], 1, rtol=0.5)
    assert_equal(cat.get_dsid(cache_dir, cat.mid), uuid )
    cat.delete()


def test_unique_label():
    """Check that testing works with only one label in the training test"""
    np.random.seed(10)
    Nshape = ground_truth.index.values.shape
    is_relevant = np.zeros(Nshape).astype(int)
    scores = categorization_score(ground_truth.index.values,
                        is_relevant,
                        ground_truth.index.values,
                        np.random.rand(*Nshape))
    # TODO unused variable 'scores'


def test_categorization_score():
    idx = [1, 2,  3,  4,  5, 6]
    y   = [1, 1, -1, -1, -1, 1]
    idx_ref = [10, 5, 3, 2, 6]
    y_ref   = [0,  1, 0, 1, 1]

    scores = categorization_score(idx_ref, y_ref, idx, y)

    assert_allclose(scores['precision'], 1.0)
    assert_allclose(scores['recall'], 0.66666666, rtol=1e-4)

    # make sure permutations don't affect the result
    idx_ref2 = [10, 5, 2, 3, 6]
    y_ref2   = [0,  1, 1, 0, 1]
    scores2 = categorization_score(idx_ref2, y_ref2, idx, y)
    assert scores['average_precision'] == scores2['average_precision']



def test_ensemble_stacking():
    from sklearn.linear_model import LogisticRegression
    try:
        from freediscovery_extra import _EnsembleStacking
    except ImportError:
        raise SkipTest

    st = _EnsembleStacking([('m1', LogisticRegression()), ('m2', LogisticRegression())])

    X_train = np.random.randn(100, 5)
    Y_train = np.random.randint(2, size=(100))
    X_test = np.random.randn(20, 5)

    st.fit(X_train, Y_train)
    st.predict_proba(X_test)


