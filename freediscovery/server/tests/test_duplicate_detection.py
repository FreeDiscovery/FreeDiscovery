# -*- coding: utf-8 -*-

import pytest
from unittest import SkipTest

from freediscovery.utils import dict2type

from .base import parse_res, V01, app, app_notest, get_features_cached

# ============================================================================#
#
#                     Duplicates detection
#
# ============================================================================#


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
    pars = {'parent_id': dsid,
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
