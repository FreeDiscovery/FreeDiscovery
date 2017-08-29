# -*- coding: utf-8 -*-

from freediscovery.utils import dict2type
from .base import parse_res, V01, app, app_notest, get_features, get_features_lsi

# ============================================================================#
#
#                     Custom Stop Words
#
# ============================================================================#

def test_stop_words(app):
    name = "test_acstw"
    tested_stop_words = ['one', 'two', 'three', 'foure', 'five', 'six']

    method = V01 + "/stop-words/"
    pars = dict(name=name, stop_words=tested_stop_words)
    data = app.post_check(method, json=pars)

    method = V01 + "/stop-words/{}".format(name)
    data = app.get_check(method)

    assert dict2type(data, collapse_lists=True) == {'name': 'str', 'stop_words': ['str']}
    assert data["stop_words"] == tested_stop_words

    method = V01 + "/stop-words/{}".format(name)
    app.delete_check(method)
