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

from .base import parse_res, V01, app, app_notest, get_features, get_features_lsi



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
        dsid, lsi_id, _ = get_features_lsi(app, hashed=False)
        parent_id = lsi_id
    else:
        dsid, _ = get_features(app, hashed=False)
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
        pars['n_clusters'] = 2
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
    assert sorted(data.keys()) == \
            sorted(['cluster_terms', 'labels', 'pars', 'htree'])
    assert len(data['cluster_terms'][0]) == 2

    if data['htree']:
        assert sorted(data['htree'].keys()) == \
                sorted(['n_leaves', 'n_components', 'children'])

    res = app.delete(method)
    assert res.status_code == 200
