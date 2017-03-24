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


@pytest.mark.parametrize("method, min_score, max_results", [('regular', -1, None),
                                                            ('semantic', -1, None),
                                                            ('semantic', 0.5, None),
                                                            ('semantic', -1, 1000000),
                                                            ('semantic', -1, 3),
                                                            ('semantic', -1, 0),
                                                            ])
def test_search(app, method, min_score, max_results):

    if method == 'semantic':
        dsid, lsi_id, _, input_ds = get_features_lsi_cached(app, hashed=False)
        parent_id = lsi_id
    elif method == 'regular':
        dsid, _, input_ds = get_features_cached(app, hashed=False)
        lsi_id = None
        parent_id = dsid
    query = """The American National Standards Institute sells ANSI standards, and also
    ISO (international) standards.  Their sales office is at 1-212-642-4900,
    mailing address is 1430 Broadway, NY NY 10018.  It helps if you have the
    complete name and number.
    """
    query_file_path = "02256.txt"

    pars = dict(parent_id=parent_id,
                min_score=min_score,
                query=query)
    if max_results is not None:
        pars['max_results'] = max_results


    data = app.post_check(V01 + "/search/", json=pars)
    assert sorted(data.keys()) == ['data']
    data = data['data']
    for row in data:
        assert dict2type(row) == {'score': 'float',
                                  'document_id': 'int'}
    scores = np.array([row['score'] for row in data])
    assert_equal(np.diff(scores) <= 0, True)
    assert_array_less(min_score, scores)
    if max_results:
        assert len(data) == min(max_results, len(input_ds['dataset']))
    elif min_score > -1:
        pass
    else:
        assert len(data) == 1967

    data = app.post_check(V01 + "/feature-extraction/{}/id-mapping".format(dsid), 
                   json={'data': [data[0]]})
    if not max_results:
        assert data['data'][0]['file_path'] == query_file_path


def test_search_document_id(app):
    dsid, lsi_id, _, input_ds = get_features_lsi_cached(app, hashed=False)
    parent_id = lsi_id

    max_results = 2
    query_document_id = 3844

    pars = dict(parent_id=parent_id,
                max_results=max_results,
                sort=True,
                query_document_id=query_document_id)


    data = app.post_check(V01 + "/search/", json=pars)
    assert sorted(data.keys()) == ['data']
    data = data['data']
    for row in data:
        assert dict2type(row) == {'score': 'float',
                                  'document_id': 'int'}
    scores = np.array([row['score'] for row in data])
    assert_equal(np.diff(scores) <= 0, True)
    assert len(data) == min(max_results, len(input_ds['dataset']))
    assert data[0]['document_id'] == query_document_id
    assert data[0]['score'] >= 0.99
