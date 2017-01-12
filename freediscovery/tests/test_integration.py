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
from freediscovery.dupdet import DuplicateDetection
from freediscovery.cluster import Clustering
from freediscovery.lsi import LSI
from freediscovery.io import parse_ground_truth_file
from freediscovery.utils import categorization_score
from freediscovery.exceptions import OptionalDependencyMissing
from .run_suite import check_cache


basename = os.path.dirname(__file__)
data_dir = os.path.join(basename, "..", "data", "ds_001", "raw")


@pytest.mark.parametrize('use_hashing, method', itertools.product([False, True],
                                                ['Categorization', 'LSI',
                                                 'DuplicateDetection', 'Clustering']))
def test_features_hashing(use_hashing, method):
    # check that models work both with and without hashing

    cache_dir = check_cache()

    n_features = 20000

    fe = FeatureVectorizer(cache_dir=cache_dir)
    uuid = fe.preprocess(data_dir, file_pattern='.*\d.txt', n_features=n_features,
                         use_hashing=use_hashing)
    uuid, filenames  = fe.transform()

    ground_truth = parse_ground_truth_file(
                            os.path.join(data_dir, "..", "ground_truth_file.txt"))


    if method == 'Categorization':
        cat = Categorizer(cache_dir=cache_dir, dsid=uuid, cv_n_folds=2)
        index = cat.fe.search(ground_truth.index.values)

        try:
            coefs, Y_train = cat.train(
                                    index,
                                    ground_truth.is_relevant.values,
                                    )
        except OptionalDependencyMissing:
            raise SkipTest

        Y_pred = cat.predict()
        X_pred = np.arange(cat.fe.n_samples_, dtype='int')
        idx_gt = cat.fe.search(ground_truth.index.values)

        scores = categorization_score(idx_gt,
                            ground_truth.is_relevant.values,
                            X_pred, Y_pred)
        assert_allclose(scores['precision'], 1, rtol=0.5)
        assert_allclose(scores['recall'], 1, rtol=0.5)
        cat.delete()
    elif method == 'LSI':
        lsi = LSI(cache_dir=cache_dir, dsid=uuid)
        lsi_res, exp_var = lsi.transform(n_components=100)  # TODO unused variables
        lsi_id = lsi.mid
        assert lsi.get_dsid(fe.cache_dir, lsi_id) == uuid
        assert lsi.get_path(lsi_id) is not None
        assert lsi._load_pars() is not None
        lsi.load(lsi_id)

        idx_gt = lsi.fe.search(ground_truth.index.values)
        idx_all = np.arange(lsi.fe.n_samples_, dtype='int')

        for method in ['nearest-neighbor-1', 'nearest-centroid']:
            _, Y_train, Y_pred, ND_train = lsi.predict(
                                    idx_gt,
                                    ground_truth.is_relevant.values,
                                    method=method)
            scores = categorization_score(idx_gt,
                                ground_truth.is_relevant.values,
                                idx_all, Y_pred)
            assert_allclose(scores['precision'], 1, rtol=0.5)
            assert_allclose(scores['recall'], 1, rtol=0.3)
    elif method == 'DuplicateDetection':
        dd = DuplicateDetection(cache_dir=cache_dir, dsid=uuid)
        try:
            dd.fit()
        except ImportError:
            raise SkipTest
        cluster_id = dd.query(distance=10)
    elif method =='Clustering':
        if not use_hashing:
            cat = Clustering(cache_dir=cache_dir, dsid=uuid)
            cm = getattr(cat,'k_means')
            labels, htree = cm(2, lsi_components=20)

            terms = cat.compute_labels(n_top_words=10)
        else:
            with pytest.raises(NotImplementedError):
                Clustering(cache_dir=cache_dir, dsid=uuid)


    else:
        raise ValueError
