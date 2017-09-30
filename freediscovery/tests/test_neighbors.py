#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import (assert_allclose, assert_equal)

import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.externals import joblib

from freediscovery.neighbors import (NearestNeighborRanker,
                                     NearestCentroidRanker,
                                     _chunk_kneighbors)
from .run_suite import check_cache


basename = os.path.dirname(__file__)


cache_dir = check_cache()


@pytest.mark.parametrize('n_categories', [1, 2, 3, 4])
def test_nearest_neighbor_ranker(n_categories):
    # check that we have sensible results with respect to
    # NN1 binary classification (supervised, with both positive
    # and negative samples)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(0)

    n_samples = 110
    n_features = 10

    X = np.random.rand(n_samples, n_features)
    normalize(X, copy=False)
    index = np.arange(n_samples, dtype='int')
    y = np.random.randint(0, n_categories, size=(n_samples,))
    index_train, index_test, y_train, y_test = train_test_split(index, y)
    X_train = X[index_train]
    X_test = X[index_test]

    rk = NearestNeighborRanker()
    rk.fit(X_train, y_train)
    y_pred, idx_pred = rk.kneighbors(X_test, batch_size=90) # choose a batch size smaller
                                                           # than n_samples

    assert y_pred.shape == (X_test.shape[0], n_categories)
    assert y_pred.min() >= -1 and y_pred.max() <= 1 # as we are using cosine similarities
    assert idx_pred.shape == (X_test.shape[0], n_categories)

    # postive scores correspond to positive documents

    #assert_equal((y_pred > 0), y_train[idx])

    cl = KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='cosine')
    cl.fit(X_train, y_train)

    idx_ref_cl = cl.predict(X_test)

    # make sure we get the same results as for the KNeighborsClassifier
    label_pred = np.argmax(y_pred, axis=1)
    assert_equal(label_pred, idx_ref_cl)

    nn = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='cosine')
    nn.fit(X_train)
    S_ref_nn, idx_ref_nn = nn.kneighbors(X_test)

    assert_equal(idx_pred[range(len(label_pred)), label_pred], idx_ref_nn[:,0])
    assert_allclose(np.max(y_pred, axis=1)[:, None], 1 - S_ref_nn)


def test_nearest_neighbor_ranker_is_picklable():
    mod = NearestNeighborRanker()

    mod.fit([[0, 1], [1, 0]], [0, 1])

    try:
        tmp_file = os.path.join(cache_dir, 'tmp_NearestNeighborRanker.pkl')
        joblib.dump(mod, tmp_file)

        mod2 = joblib.load(tmp_file)
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


def test_nearest_centroid_ranker():
    # in the case where there is a single point by centroid,
    # nearest centroid should reduce to nearest neighbor
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(0)

    n_samples = 100
    n_features = 120
    X = np.random.rand(n_samples, n_features)
    normalize(X, copy=False)
    index = np.arange(n_samples, dtype='int')
    y = np.arange(n_samples, dtype='int')
    index_train, index_test, y_train, y_test = train_test_split(index, y)
    X_train = X[index_train]
    X_test = X[index_test]


    nn = NearestNeighbors(n_neighbors=1, algorithm='brute')
    nn.fit(X_train)
    dist_ref, idx_ref = nn.kneighbors(X_test)

    nc = NearestCentroidRanker()
    nc.fit(X_train, y_train)
    dist_pred = nc.decision_function(X_test)
    y_pred = nc.predict(X_test)

    # ensures that we have the same number of unique ouput points
    # (even if absolute labels are not preserved)
    assert np.unique(idx_ref[:,0]).shape ==  np.unique(y_pred).shape

    assert_allclose(dist_pred, dist_ref[:,0])


@pytest.mark.parametrize('batch_size', [1, 101, 100])
def test_nn_chunking(batch_size):

    shape = (1000, 10)

    X = 2*np.ones(shape)
    def func(X):
        assert X.shape[0] > 0 # we need at least one point
        return X**2, X - 1

    d, idx = _chunk_kneighbors(func, X, batch_size=batch_size)

    assert_allclose(d, 4*np.ones(shape))
    assert_allclose(idx, 1*np.ones(shape))
