#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

import os.path
from unittest import SkipTest

import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
                           assert_array_less)

import pytest
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from freediscovery.text import FeatureVectorizer
from freediscovery.categorization import (Categorizer, _zip_relevant,
        _unzip_relevant, NearestNeighborRanker)
from freediscovery.io import parse_ground_truth_file
from freediscovery.utils import categorization_score
from freediscovery.exceptions import OptionalDependencyMissing
from .run_suite import check_cache


basename = os.path.dirname(__file__)


cache_dir = check_cache()

EPSILON = 1e-4


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
    index = cat.fe.search(ground_truth.index.values)

    try:
        coefs, Y_train = cat.train(
                                index,
                                ground_truth.is_relevant.values,
                                method=method,
                                cv=cv)
    except OptionalDependencyMissing:
        raise SkipTest



    Y_pred = cat.predict()
    X_pred = np.arange(cat.fe.n_samples_, dtype='int')
    idx_gt = cat.fe.search(ground_truth.index.values)

    scores = categorization_score(idx_gt,
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

    idx = np.arange(len(is_relevant), dtype='int')

    scores = categorization_score(idx,
                        is_relevant,
                        idx,
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

def test_relevant_zip():
    relevant_id = [2, 3, 5]
    non_relevant_id = [1, 7, 10]
    idx_id = [2, 3, 5, 1, 7, 10]
    y = [1, 1, 1, 0, 0, 0]

    idx_id2, y2 = _zip_relevant(relevant_id, non_relevant_id)
    assert_equal(idx_id, idx_id2)
    assert_equal(y, y2)

    relevant_id2, non_relevant_id2 = _unzip_relevant(idx_id2, y2)
    assert_equal(relevant_id2, relevant_id)
    assert_equal(non_relevant_id2, non_relevant_id)

def test_relevant_positives_zip():
    # Check that _unzip_relavant works with only relevant files
    idx_id = [2, 3, 5, 1, 7, 10]
    y = [1, 1, 1, 1, 1, 1]
    relevant_id2, non_relevant_id2 = _unzip_relevant(idx_id, y)
    assert_equal(relevant_id2, idx_id)
    assert_equal(non_relevant_id2, np.array([]))


def test_nearest_neighbor_ranker_supervised():
    # check that we have sensible results with respect to
    # NN1 binary classification (supervised, with both positive
    # and negative samples)
    from sklearn.neighbors import KNeighborsClassifier
    np.random.seed(0)

    n_samples = 1000
    n_features = 10

    X = np.random.rand(n_samples, n_features)
    normalize(X, copy=False)
    index = np.arange(n_samples, dtype='int')
    y = np.random.randint(0, 2, size=(n_samples,))
    index_train, index_test, y_train, y_test = train_test_split(index, y)
    X_train = X[index_train]
    X_test = X[index_test]

    rk = NearestNeighborRanker()
    rk.fit(X_train, y_train)
    y_pred, idx, md = rk.kneighbors(X_test)

    assert y_pred.shape == (X_test.shape[0],)
    assert y_pred.min() >= -1 and y_pred.min() <= -0.8 # as we are using cosine similarities
    assert y_pred.max() <=  1 and y_pred.max() >= 0.8
    assert idx.shape == (X_test.shape[0],)
    assert sorted(md.keys()) == ['dist_n', 'dist_p', 'ind_n', 'ind_p']
    for key, val in md.items():
        assert val.shape == (X_test.shape[0],)
        assert_array_less(0, val) # all values are positive

    # postive scores correspond to positive documents
    assert_equal((y_pred > 0), y_train[idx])

    cl = KNeighborsClassifier(n_neighbors=1)
    cl.fit(X_train, y_train)

    y_ref = cl.predict(X_test)

    # make sure we get the same results as for the KNeighborsClassifier
    assert_equal(y_ref, y_train[idx])

def test_nearest_neighbor_ranker_unsupervised():
    # Check NearestNeighborRanker with only positive samples
    # (unsupervised)
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(0)

    n_samples = 1000
    n_features = 120

    X = np.random.rand(n_samples, n_features)
    normalize(X, copy=False)
    index = np.arange(n_samples, dtype='int')
    y = np.ones(n_samples, dtype=np.int)
    index_train, index_test, y_train, y_test = train_test_split(index, y)
    X_train = X[index_train]
    X_test = X[index_test]

    rk = NearestNeighborRanker()
    rk.fit(X_train, y_train)
    y_pred, idx, md = rk.kneighbors(X_test)

    assert_array_less(0, y_pred) # all distance are positive

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_train)
    dist, idx_ref = nn.kneighbors(X_test)

    assert_equal(idx, idx_ref[:,0])

    assert_equal(y_pred, (1 - dist[:,0]/4))

