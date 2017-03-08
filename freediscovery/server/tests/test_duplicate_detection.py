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
from numpy.testing import assert_equal, assert_almost_equal

from .. import fd_app
from ...utils import _silent, dict2type, sdict_keys
from ...ingestion import DocumentIndex
from ...exceptions import OptionalDependencyMissing
from ...tests.run_suite import check_cache

from .base import parse_res, V01, app, app_notest, get_features_cached

#=============================================================================#
#
#                     Duplicates detection
#
#=============================================================================#


@pytest.mark.parametrize('kind, options', [['simhash', {'distance': 3}],
                                             ['i-match', {}]])
def test_api_dupdetection(app, kind, options):

    if kind == 'simhash':
        try:
            import simhash
        except ImportError:
            raise SkipTest

    dsid, pars, _ = get_features_cached(app, hashed=False)

    method = V01 + "/feature-extraction/{}".format(dsid)
    data = app.get_check(method)

    url = V01 + "/duplicate-detection" 
    pars = { 'parent_id': dsid,
             'method': kind}
    data = app.post_check(url, json=pars)
    assert dict2type(data) == {'id': 'str'}
    mid = data['id']

    url += '/{}'.format(mid)
    data = app.get_check(url, query_string=options)

    assert dict2type(data, max_depth=1) == {'data': 'list'}
    for row in data['data']:
        assert dict2type(row, max_depth=1) == {'cluster_id': 'int',
                                               'cluster_similarity': 'float',
                                               'documents': 'list'}

    app.delete_check(url)
