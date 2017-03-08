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
    data = app.post_check(method, json=pars)

    method = V01 + "/stop-words/{}".format(name)
    data = app.get_check(method)

    assert dict2type(data, collapse_lists=True) == {'name': 'str', 'stop_words': ['str']}
    assert data["stop_words"] == tested_stop_words

    method = V01 + "/stop-words/{}".format(name)
    app.delete_check(method)
