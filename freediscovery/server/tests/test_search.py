# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pytest
import json
import itertools
from unittest import SkipTest
import numpy as np
from numpy.testing import assert_equal, assert_array_less

from .. import fd_app
from ...utils import _silent, dict2type, sdict_keys
from ...ingestion import DocumentIndex
from ...exceptions import OptionalDependencyMissing
from ...tests.run_suite import check_cache

from .base import (parse_res, V01, app, app_notest, get_features_cached,
                   get_features_lsi, get_features_lsi_cached)


@pytest.mark.parametrize("method, min_score", [('regular', -1),
                                               ('semantic', -1),
                                               ('semantic', 0.5)])
def test_search(app, method, min_score):

    if method == 'semantic':
        dsid, lsi_id, _ = get_features_lsi_cached(app, hashed=False)
        parent_id = lsi_id
    elif method == 'regular':
        dsid, _ = get_features_cached(app, hashed=False)
        lsi_id = None
        parent_id = dsid
    query = """The American National Standards Institute sells ANSI standards, and also
    ISO (international) standards.  Their sales office is at 1-212-642-4900,
    mailing address is 1430 Broadway, NY NY 10018.  It helps if you have the
    complete name and number.
    """
    query_file_path = "02256.txt"

    res = app.post(V01 + "/search/", json=dict(parent_id=parent_id,
                                               min_score=min_score,
                                               query=query,
                                               ))
    assert res.status_code == 200
    data = parse_res(res)
    assert sorted(data.keys()) == ['data']
    data = data['data']
    for row in data:
        assert dict2type(row) == {'score': 'float',
                                  'internal_id': 'int',
                                  'document_id': 'int'}
    scores = np.array([row['score'] for row in data])
    assert_equal(np.diff(scores) <= 0, True)
    assert_array_less(min_score, scores)

    res = app.post(V01 + "/feature-extraction/{}/id-mapping".format(dsid), 
                   json={'data': [data[0]]})
    res = parse_res(res)
    assert res['data'][0]['file_path'] == query_file_path
