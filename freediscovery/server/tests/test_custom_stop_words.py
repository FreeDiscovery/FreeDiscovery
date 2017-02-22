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
from .base import parse_res, V01, app, app_notest, get_features, get_features_lsi

V01 = '/api/v0'


data_dir = os.path.dirname(__file__)
email_data_dir = os.path.join(data_dir, "..", "data", "fedora-devel-list-2008-October")
data_dir = os.path.join(data_dir, "..", "data", "ds_001", "raw")


#=============================================================================#
#
#                     Custom Stop Words
#
#=============================================================================#

def test_stop_words(app):
    name = "test_acstw"
    tested_stop_words = ['one', 'two', 'three', 'foure', 'five', 'six']

    method = V01 + "/stop-words/"
    pars = dict(name=name, stop_words=tested_stop_words)
    res = app.post(method, json=pars)
    assert res.status_code == 200

    method = V01 + "/stop-words/{}".format(name)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)

    assert dict2type(data, collapse_lists=True) == {'name': 'str', 'stop_words': ['str']}
    assert data["stop_words"] == tested_stop_words
    assert (len(data["stop_words"]) == len(tested_stop_words))
    i = 0
    for word in data["stop_words"]:
        assert word == tested_stop_words[i]
        i += 1

