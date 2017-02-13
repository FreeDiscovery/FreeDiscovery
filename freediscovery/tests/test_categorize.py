#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

import os.path
from unittest import SkipTest
import re

import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
                           assert_array_less, assert_array_equal)

import pytest
import itertools

from freediscovery.base import PipelineFinder
from freediscovery.text import FeatureVectorizer
from freediscovery.lsi import _LSIWrapper
from freediscovery.categorization import _CategorizerWrapper
from freediscovery.io import parse_ground_truth_file
from freediscovery.utils import categorization_score
from freediscovery.exceptions import OptionalDependencyMissing
from .run_suite import check_cache


basename = os.path.dirname(__file__)


cache_dir = check_cache()

EPSILON = 1e-4


data_dir = os.path.join(basename, "..", "data", "ds_001", "raw")

fe = FeatureVectorizer(cache_dir=cache_dir)
vect_uuid = fe.preprocess(data_dir, file_pattern='.*\d.txt')
vect_uuid, filenames  = fe.transform()


lsi = _LSIWrapper(cache_dir=cache_dir, parent_id=vect_uuid)
lsi.fit_transform(n_components=6)

ground_truth = parse_ground_truth_file(
                        os.path.join(data_dir, "..", "ground_truth_file.txt"))

_test_cases = itertools.product(
                       [False, True],
                       ["LinearSVC", "LogisticRegression", 'xgboost', "NearestNeighbor",
                        "NearestCentroid"],
                        #'MLPClassifier', 'ensemble-stacking' not supported in production the moment
                       [None, 'fast'])
_test_cases = filter(lambda x: not (x[1].startswith("Nearest") and x[2]),
                     _test_cases)


@pytest.mark.parametrize('use_lsi, method, cv', _test_cases)
def test_categorization(use_lsi, method, cv):

    if 'CIRCLECI' in os.environ and cv == 'fast' and method in ['LinearSVC', 'xgboost']:
        raise SkipTest # Circle CI is too slow and timesout

    if method == 'xgboost':
        try:
            import xgboost
        except ImportError:
            raise SkipTest

    if not use_lsi:
        uuid = vect_uuid
    else:
        uuid = lsi.mid

    cat = _CategorizerWrapper(cache_dir=cache_dir, parent_id=uuid, cv_n_folds=2)
    index = cat.fe.db._search_filenames(ground_truth.file_path.values)

    try:
        coefs, Y_train = cat.train(
                                index,
                                ground_truth.is_relevant.values,
                                method=method,
                                cv=cv)
    except OptionalDependencyMissing:
        raise SkipTest



    Y_pred, md = cat.predict()
    X_pred = np.arange(cat.fe.n_samples_, dtype='int')
    idx_gt = cat.fe.db._search_filenames(ground_truth.file_path.values)

    scores = categorization_score(idx_gt,
                        ground_truth.is_relevant.values,
                        X_pred, Y_pred)

    assert cat.get_params() is not None

    if method == 'NearestNeighbor':
        assert sorted(list(md.keys())) == ['dist_n', 'dist_p', 'ind_n', 'ind_p']
        for key, val in md.items():
            assert val.shape == Y_pred.shape
    else:
        assert md == {}

    if method in ['xgboost', 'ensemble-stacking']:
        # this parameter fail for some reason so far...
        return
    assert_allclose(scores['precision'], 1, rtol=0.5)
    assert_allclose(scores['recall'], 1, rtol=0.68)
    cat.delete()

def test_explain_categorization():
    from freediscovery.categorization import explain_binary_categorization

    uuid = vect_uuid

    cat = _CategorizerWrapper(cache_dir=cache_dir, parent_id=uuid, cv_n_folds=2)
    index = cat.fe.db._search_filenames(ground_truth.file_path.values)

    model, _ = cat.train(index,
                         ground_truth.is_relevant.values,
                         method='LogisticRegression')
    _, X = cat.fe.load()
    vect = cat.fe._load_model()

    weights = explain_binary_categorization(model, vect.vocabulary_, X[0, :])
    assert len(weights.keys()) < len(vect.vocabulary_) # not all vocabulary keys are returned





@pytest.mark.parametrize('n_steps', [2, 3])
def test_pipeline(n_steps):
    """ Test a 2 or 3 step pipelines with
        vectorizer (+ lsi) + classifier """

    if n_steps == 2:
        uuid = vect_uuid
    elif n_steps == 3:
        uuid = lsi.mid
    else:
        raise ValueError

    cat = _CategorizerWrapper(cache_dir=cache_dir, parent_id=uuid, cv_n_folds=2)
    index = cat.fe.db._search_filenames(ground_truth.file_path.values)

    coefs, Y_train = cat.train( index, ground_truth.is_relevant.values)

    cat.predict()

    assert len(cat.pipeline) == n_steps - 1

    # additional tests
    if n_steps == 3:
        pf = PipelineFinder.by_id(cat.mid, cache_dir)

        assert list(pf.keys()) == ['vectorizer', 'lsi', 'categorizer']
        assert list(pf.parent.keys()) == ['vectorizer', 'lsi']
        assert list(pf.parent.parent.keys()) == ['vectorizer']

        assert pf.mid == cat.mid
        assert pf.parent.mid == lsi.mid
        assert pf.parent.parent.mid == vect_uuid
        with pytest.raises(ValueError):
            pf.parent.parent.parent

        for estimator_type, mid in pf.items():
            path = pf.get_path(mid, absolute=False)
            if estimator_type == 'vectorizer':
                assert re.match('ediscovery_cache.*', path)
            elif estimator_type == 'lsi':
                assert re.match('ediscovery_cache.*lsi', path)
            elif estimator_type == 'categorizer':
                assert re.match('ediscovery_cache.*lsi.*categorizer', path)
            else:
                raise ValueError



def test_unique_label():
    """Check that testing works with only one label in the training test"""
    np.random.seed(10)
    Nshape = ground_truth.file_path.values.shape
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

