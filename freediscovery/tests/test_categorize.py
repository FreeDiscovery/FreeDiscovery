# -*- coding: utf-8 -*-

import os.path
from unittest import SkipTest
import re

import numpy as np
from numpy.testing import assert_allclose

import pandas as pd
import pytest
import itertools

from freediscovery.base import PipelineFinder
from freediscovery.text import FeatureVectorizer
from freediscovery.lsi import _LSIWrapper
from freediscovery.categorization import _CategorizerWrapper
from freediscovery.io import parse_ground_truth_file
from freediscovery.metrics import categorization_score
from freediscovery.exceptions import OptionalDependencyMissing, WrongParameter
from .run_suite import check_cache


basename = os.path.dirname(__file__)


cache_dir = check_cache()

EPSILON = 1e-4


data_dir = os.path.join(basename, "..", "data", "ds_001", "raw")

fe = FeatureVectorizer(cache_dir=cache_dir)
vect_uuid = fe.setup()
fe.ingest(data_dir, file_pattern='.*\d.txt')


lsi = _LSIWrapper(cache_dir=cache_dir, parent_id=vect_uuid)
lsi.fit_transform(n_components=6)

ground_truth = parse_ground_truth_file(
                        os.path.join(data_dir, "..", "ground_truth_file.txt"))

_test_cases = itertools.product(
                       [False, True],
                       ["LinearSVC", "LogisticRegression", 'xgboost',
                        "NearestNeighbor", "NearestCentroid"],
                       [None, 'fast'])

# 'MLPClassifier', 'ensemble-stacking' not supported in production the moment
_test_cases = filter(lambda x: not (x[1].startswith("Nearest") and x[2]),
                     _test_cases)


@pytest.mark.parametrize('use_lsi, method, cv', _test_cases)
def test_categorization(use_lsi, method, cv):

    if 'CIRCLECI' in os.environ and cv == 'fast'\
            and method in ['LinearSVC', 'xgboost']:
        raise SkipTest  # Circle CI is too slow and timesout

    if method == 'xgboost':
        try:
            import xgboost
        except ImportError:
            raise SkipTest

    if not use_lsi:
        uuid = vect_uuid
    else:
        uuid = lsi.mid

    cat = _CategorizerWrapper(cache_dir=cache_dir,
                              parent_id=uuid, cv_n_folds=2)
    cat.fe.db_.filenames_ = cat.fe.filenames_
    index = cat.fe.db_._search_filenames(ground_truth.file_path.values)

    try:
        model, Y_train = cat.fit(
                                index,
                                ground_truth.is_relevant.values,
                                method=method,
                                cv=cv)
    except OptionalDependencyMissing:
        raise SkipTest
    except WrongParameter:
        if method in ['NearestNeighbor', 'NearestCentroid']:
            return
        else:
            raise

    Y_pred, md = cat.predict()
    X_pred = np.arange(cat.fe.n_samples_, dtype='int')
    idx_gt = cat.fe.db_._search_filenames(ground_truth.file_path.values)

    scores = categorization_score(idx_gt,
                                  ground_truth.is_relevant.values,
                                  X_pred, np.argmax(Y_pred, axis=1))

    assert cat.get_params() is not None

    assert Y_pred.shape == (cat.fe.n_samples_,
                            len(np.unique(ground_truth.is_relevant.values)))

    if method == 'NearestNeighbor':
        assert md.shape == Y_pred.shape
    else:
        assert md is None

    if method in ['xgboost', 'ensemble-stacking']:
        # this parameter fail for some reason so far...
        return
    assert_allclose(scores['precision'], 1, rtol=0.5)
    assert_allclose(scores['recall'], 1, rtol=0.68)
    cat.delete()


@pytest.mark.parametrize('max_result_categories, sort_by, has_nn',
                         [(1, None, True),
                          (1, 'score', True),
                          (1, None, False),
                          (2, None, True), (3, '', True)])
def test_categorization2dict(max_result_categories, sort_by, has_nn):
    raise SkipTest
    import json
    Y_pred = np.array([[1.0, 0.0],
                       [0.6, 0.4],
                       [0.0, 1.0],
                       [0.65, 0.35],
                       [1.3, -0.3]])

    if has_nn:
        D_nn = np.array([[0, 2],
                         [0, 2],
                         [0, 2],
                         [0, 2],
                         [0, 2]])
    else:
        D_nn = None
    id_mapping = pd.DataFrame([{'internal_id': idx, 'document_id': idx**2}
                               for idx in range(Y_pred.shape[0])])
    res = _CategorizerWrapper.to_dict(Y_pred, D_nn,
                                      ['negative', 'positive'], id_mapping,
                                      max_result_categories=max_result_categories,
                                      sort_by=sort_by)
    assert list(res.keys()) == ['data']
    if has_nn:
        if max_result_categories >= 2 and not sort_by:
            assert res['data'][0] == {
                                        "internal_id": 0,
                                        "document_id": 0,
                                        "scores": [
                                            {
                                                "score": 1.0,
                                                "internal_id": 0,
                                                "category": "negative",
                                                "document_id": 0
                                            },
                                            {
                                                "score": 0.0,
                                                "internal_id": 2,
                                                "category": "positive",
                                                "document_id": 4
                                            }
                                        ]
                                      }
        elif max_result_categories == 1 and not sort_by:
            assert res['data'][0] == {
                                        "internal_id": 0,
                                        "document_id": 0,
                                        "scores": [
                                            {
                                                "score": 1.0,
                                                "internal_id": 0,
                                                "category": "negative",
                                                "document_id": 0
                                            },
                                        ]
                                      }

        elif max_result_categories == 1 and sort_by:
            assert res['data'][0] == {
                                        "internal_id": 4,
                                        "document_id": 16,
                                        "scores": [
                                            {
                                                "score": 1.3,
                                                "internal_id": 0,
                                                "category": "negative",
                                                "document_id": 0
                                            }
                                        ]
                                    }
        elif max_result_categories == 0 and not sort_by:
            assert res['data'][0]['scores'] == []
        else:
            raise NotImplementedError
    else:
        if max_result_categories == 1 and not sort_by:
            assert res['data'][0] == {
                                        "internal_id": 0,
                                        "document_id": 0,
                                        "scores": [
                                            {
                                                "score": 1.0,
                                                "category": "negative",
                                            },
                                        ]
                                      }
        else:
            raise NotImplementedError

    for row in res['data']:
        # we are in decreasing order
        assert (np.diff([el['score'] for el in row['scores']]) < 0.0).all()

    # check that we are picklable
    json.dumps(res)


def test_explain_categorization():
    from freediscovery.categorization import binary_sensitivity_analysis

    uuid = vect_uuid

    cat = _CategorizerWrapper(cache_dir=cache_dir, parent_id=uuid,
                              cv_n_folds=2)
    cat.fe.db_.filenames_ = cat.fe.filenames_
    index = cat.fe.db_._search_filenames(ground_truth.file_path.values)

    model, _ = cat.fit(index, ground_truth.is_relevant.values,
                       method='LogisticRegression')
    X = cat.fe._load_features()
    vect = cat.fe.vect_

    weights = binary_sensitivity_analysis(model, vect.vocabulary_, X[0, :])
    # not all vocabulary keys are returned
    assert len(weights.keys()) < len(vect.vocabulary_)


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

    cat = _CategorizerWrapper(cache_dir=cache_dir,
                              parent_id=uuid, cv_n_folds=2)
    cat.fe.db_.filenames_ = cat.fe.filenames_
    index = cat.fe.db_._search_filenames(ground_truth.file_path.values)

    coefs, Y_train = cat.fit(index, ground_truth.is_relevant.values)

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
            path = str(pf.get_path(mid, absolute=False))
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

    scores = categorization_score(idx, is_relevant, idx,
                                  np.random.rand(*Nshape))


def test_categorization_score():
    idx = [1, 2,  3,  4,  5, 6]
    y = [1, 1, -1, -1, -1, 1]
    idx_ref = [10, 5, 3, 2, 6]
    y_ref = [0,  1, 0, 1, 1]

    scores = categorization_score(idx_ref, y_ref, idx, y)

    assert_allclose(scores['precision'], 1.0)
    assert_allclose(scores['recall'], 0.66666666, rtol=1e-4)

    # make sure permutations don't affect the result
    idx_ref2 = [10, 5, 2, 3, 6]
    y_ref2 = [0, 1, 1, 0, 1]
    scores2 = categorization_score(idx_ref2, y_ref2, idx, y)
    assert scores['average_precision'] == scores2['average_precision']
