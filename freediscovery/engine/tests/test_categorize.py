# -*- coding: utf-8 -*-

from pathlib import Path
import os.path
from unittest import SkipTest
import re

import numpy as np
from numpy.testing import assert_allclose

import pandas as pd
import pytest
import itertools

from freediscovery.engine.pipeline import PipelineFinder
from freediscovery.engine.vectorizer import FeatureVectorizer
from freediscovery.engine.categorization import _CategorizerWrapper
from freediscovery.engine.lsi import _LSIWrapper
from freediscovery.io import parse_ground_truth_file
from freediscovery.metrics import categorization_score
from freediscovery.exceptions import OptionalDependencyMissing, WrongParameter
from freediscovery.tests.run_suite import check_cache


basename = Path(__file__).parent

cache_dir = check_cache()

EPSILON = 1e-4


data_dir = basename / ".." / ".." / "data" / "ds_001" / "raw"

ground_truth = parse_ground_truth_file(
                        str(data_dir / ".." / "ground_truth_file.txt"))

fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
vect_uuid = fe.setup()
fe.ingest(str(data_dir), file_pattern='.*\d.txt')


lsi = _LSIWrapper(cache_dir=cache_dir, parent_id=vect_uuid, mode='w')
lsi.fit_transform(n_components=6)


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
