# -*- coding: utf-8 -*-

import os.path
from unittest import SkipTest

import numpy as np
from numpy.testing import assert_allclose

import pytest
import itertools

from freediscovery.engine.vectorizer import FeatureVectorizer
from freediscovery.engine.categorization import _CategorizerWrapper
from freediscovery.engine.near_duplicates import _DuplicateDetectionWrapper
from freediscovery.engine.cluster import _ClusteringWrapper
from freediscovery.engine.lsi import _LSIWrapper
from freediscovery.io import parse_ground_truth_file
from freediscovery.metrics import categorization_score
from freediscovery.exceptions import OptionalDependencyMissing
from freediscovery.tests.run_suite import check_cache


basename = os.path.dirname(__file__)
data_dir = os.path.join(basename, "..", "..", "data", "ds_001", "raw")


@pytest.mark.parametrize('use_hashing, use_lsi, method',
                         itertools.product([False, True],
                                           [False, True],
                                           ['Categorization',
                                            'DuplicateDetection',
                                            'Clustering']))
def test_features_hashing(use_hashing, use_lsi, method):
    # check that models work both with and without hashing

    cache_dir = check_cache()

    n_features = 20000

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid = fe.setup(n_features=n_features, use_hashing=use_hashing)
    fe.ingest(data_dir, file_pattern='.*\d.txt')

    ground_truth = parse_ground_truth_file(os.path.join(data_dir,
                                           "..", "ground_truth_file.txt"))

    lsi = _LSIWrapper(cache_dir=cache_dir, parent_id=uuid, mode='w')
    lsi_res, exp_var = lsi.fit_transform(n_components=100)
    assert lsi._load_pars() is not None
    lsi._load_model()

    if method == 'Categorization':
        if use_lsi:
            parent_id = lsi.mid
            method = 'NearestNeighbor'
        else:
            parent_id = uuid
            method = 'LogisticRegression'
        cat = _CategorizerWrapper(cache_dir=cache_dir, parent_id=parent_id,
                                  cv_n_folds=2)
        cat.fe.db_.filenames_ = cat.fe.filenames_
        index = cat.fe.db_._search_filenames(ground_truth.file_path.values)

        try:
            coefs, Y_train = cat.fit(
                                    index,
                                    ground_truth.is_relevant.values,
                                    method=method
                                    )
        except OptionalDependencyMissing:
            raise SkipTest

        Y_pred, md = cat.predict()
        X_pred = np.arange(cat.fe.n_samples_, dtype='int')
        idx_gt = cat.fe.db_._search_filenames(ground_truth.file_path.values)

        scores = categorization_score(idx_gt,
                                      ground_truth.is_relevant.values,
                                      X_pred, np.argmax(Y_pred, axis=1))
        assert_allclose(scores['precision'], 1, rtol=0.5)
        assert_allclose(scores['recall'], 1, rtol=0.7)
        cat.delete()
    elif method == 'DuplicateDetection':
        dd = _DuplicateDetectionWrapper(cache_dir=cache_dir, parent_id=uuid)
        try:
            dd.fit()
        except ImportError:
            raise SkipTest
        cluster_id = dd.query(distance=10)
    elif method == 'Clustering':
        if not use_hashing:
            if use_lsi:
                parent_id = lsi.mid
                method = 'birch'
            else:
                parent_id = uuid
                method = 'k_means'
            cat = _ClusteringWrapper(cache_dir=cache_dir, parent_id=parent_id)
            cm = getattr(cat, method)
            labels = cm(2)

            htree = cat._get_htree(cat.pipeline.data)

            terms = cat.compute_labels(n_top_words=10)
        else:
            with pytest.raises(NotImplementedError):
                _ClusteringWrapper(cache_dir=cache_dir, parent_id=uuid)
    else:
        raise ValueError
