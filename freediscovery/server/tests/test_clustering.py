# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pytest
import json
import re
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

@pytest.mark.parametrize("model, use_lsi, n_clusters, optimal_sampling",
                         [('k-mean', False, 13, False),
                          ('birch', True, -1, False),
                          ('birch', True, -1, True),
                          ('birch', True, 13, False),
                          ('ward_hc', True, 13, False),
                          ('dbscan', True, None, False)])
def test_api_clustering(app, model, use_lsi, n_clusters, optimal_sampling):

    if use_lsi:
        dsid, lsi_id, _, _ = get_features_lsi_cached(app, hashed=False)
        parent_id = lsi_id
    else:
        dsid, _, _ = get_features_cached(app, hashed=False)
        lsi_id = None
        parent_id = dsid

    if model == 'birch' and n_clusters <= 0:
        is_hierarchical = True
    else:
        is_hierarchical = False

    method = V01 + "/feature-extraction/{}".format(dsid)
    data = app.get_check(method)

    url = V01 + "/clustering/" + model
    pars = { 'parent_id': parent_id, }
    if model != 'dbscan':
        pars['n_clusters'] = n_clusters
    if model == 'dbscan':
        pars.update({'eps': 0.1, "min_samples": 2})
    data = app.post_check(url, json=pars)

    assert sorted(data.keys()) == sorted(['id'])
    mid = data['id']

    url += '/{}'.format(mid)
    n_top_words = 2
    pars = {'n_top_words': n_top_words}
    if optimal_sampling:
        pars['return_optimal_sampling'] = True
    data = app.get_check(url, query_string=pars)

    assert dict2type(data, max_depth=1) == {'data': 'list'}
    for row in data['data']:
        ref_res = {'cluster_id': 'int', 'cluster_similarity': 'float',
                   'documents': 'list', 'cluster_size': 'int'}
        if is_hierarchical and not optimal_sampling:
            ref_res['children'] = 'list'
            ref_res['cluster_depth'] = 'int'
        if not optimal_sampling:
            ref_res['cluster_label'] = 'str'
            assert re.match('[^\[]+', row['cluster_label'])
        assert dict2type(row, max_depth=1) == ref_res
        # make sure we have space separated words, not a str(list)
        for irow in row['documents']:
            assert dict2type(irow) == {'document_id': 'int',
                                       'similarity': 'float'}
    if model != 'dbscan' and not is_hierarchical:
        assert len(data['data']) == 13

    #if data['htree']:
    #    assert sorted(data['htree'].keys()) == \
    #            sorted(['n_leaves', 'n_components', 'children'])

    app.delete_check(url)
