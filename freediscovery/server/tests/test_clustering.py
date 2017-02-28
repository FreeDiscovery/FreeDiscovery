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

from .base import (parse_res, V01, app, app_notest, get_features_cached,
                                        get_features_lsi_cached)



#=============================================================================#
#
#                     Clustering
#
#=============================================================================#

@pytest.mark.parametrize("model, use_lsi", [('k-mean', False),
                                            ('birch', True),
                                            ('ward_hc', True),
                                            ('dbscan', True)])
def test_api_clustering(app, model, use_lsi):

    if use_lsi:
        dsid, lsi_id, _, _ = get_features_lsi_cached(app, hashed=False)
        parent_id = lsi_id
    else:
        dsid, _, _ = get_features_cached(app, hashed=False)
        lsi_id = None
        parent_id = dsid

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)  # TODO unused variable

    #if (model == 'birch' or model == "ward_hc"):

    url = V01 + "/clustering/" + model
    pars = { 'parent_id': parent_id, }
    if model != 'dbscan':
        pars['n_clusters'] = 13
    if model == 'dbscan':
        pars.update({'eps': 0.1, "min_samples": 2})
    res = app.post(url, json=pars)

    assert res.status_code == 200
    data = parse_res(res)
    assert sorted(data.keys()) == sorted(['id'])
    mid = data['id']

    url += '/{}'.format(mid)
    n_top_words = 2
    res = app.get(url, query_string={'n_top_words': n_top_words})
    assert res.status_code == 200
    data = parse_res(res)
    assert dict2type(data, max_depth=1) == {'data': 'list'}
    for row in data['data']:
        assert dict2type(row, max_depth=1) == {'cluster_id': 'int',
                                               'cluster_similarity': 'float',
                                               'cluster_label': 'str',
                                               'documents': 'list'}
        for irow in row['documents']:
            assert dict2type(irow) == {'document_id': 'int',
                                       'similarity': 'float'}
    if model != 'dbscan':
        assert len(data['data']) == 13

    #if data['htree']:
    #    assert sorted(data['htree'].keys()) == \
    #            sorted(['n_leaves', 'n_components', 'children'])

    res = app.delete(url)
    assert res.status_code == 200
