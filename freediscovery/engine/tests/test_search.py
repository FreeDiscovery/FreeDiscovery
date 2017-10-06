# -*- coding: utf-8 -*-

import os.path

import pytest

from freediscovery.engine.vectorizer import FeatureVectorizer
from freediscovery.engine.lsi import _LSIWrapper
from freediscovery.engine.search import _SearchWrapper

from freediscovery.tests.run_suite import check_cache

basename = os.path.dirname(__file__)
cache_dir = check_cache()
data_dir = os.path.join(basename, "..", "..", "data", "ds_001", "raw")


@pytest.mark.parametrize('kind,', ['regular', 'semantic'])
def test_search_wrapper(kind):
    # check for syntax errors etc in the wrapper

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    vect_uuid = fe.setup()
    fe.ingest(data_dir, file_pattern='.*\d.txt')

    if kind == 'semantic':
        lsi = _LSIWrapper(cache_dir=cache_dir, parent_id=vect_uuid, mode='w')
        lsi.fit_transform(n_components=20)
        parent_id = lsi.mid
    else:
        parent_id = vect_uuid

    sw = _SearchWrapper(cache_dir=cache_dir, parent_id=parent_id)
    dist = sw.search("so that I can reserve a room")
    assert dist.shape == (fe.n_samples_,)
    # document 1 found by
    # grep -rn "so that I can reserve a room"
    # freediscovery/data/ds_001/raw/
    assert dist.argmax() == 1
