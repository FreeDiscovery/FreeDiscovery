# -*- coding: utf-8 -*-

import os.path
from unittest import SkipTest
import numpy as np
import pytest

from freediscovery.engine.vectorizer import FeatureVectorizer
from freediscovery.tests.run_suite import check_cache


def fd_setup(**fe_options):
    basename = os.path.dirname(__file__)
    cache_dir = check_cache()
    data_dir = os.path.join(basename, "..", "..", "data", "ds_001", "raw")
    n_features = 110000
    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid = fe.setup(n_features=n_features, use_hashing=True,
                    stop_words='english',
                    **fe_options)
    fe.ingest(data_dir, file_pattern='.*\d.txt')
    return cache_dir, uuid, fe.filenames_, fe


@pytest.mark.parametrize('method, options, fe_options',
                         [['simhash', {'distance': 3}, {}],
                          ['simhash', {'distance': 10}, {}],
                          ['i-match', {}, {}]])
def test_dup_detection(method, options, fe_options):
    if method == 'simhash':
        try:
            import simhash
        except ImportError:
            raise SkipTest
    from freediscovery.engine.near_duplicates import _DuplicateDetectionWrapper
    cache_dir, uuid, filenames, fe = fd_setup(**fe_options)

    dd = _DuplicateDetectionWrapper(cache_dir=cache_dir, parent_id=uuid)
    dd.fit(method=method)
    cluster_id = dd.query(**options)
    # cannot have more cluster_id than elements in the dataset
    assert len(np.unique(cluster_id)) <= len(np.unique(filenames))
