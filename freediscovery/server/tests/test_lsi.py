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

from .base import (parse_res, V01, app, app_notest,
                   get_features_cached)


#=============================================================================#
#
#                     LSI
#
#=============================================================================#

def test_api_lsi(app):
    dsid, pars, _ = get_features_cached(app)
    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)
    method = V01 + "/lsi/"
    res = app.get_check(method, data=dict(parent_id=dsid))

    lsi_pars = dict(n_components=101, parent_id=dsid)
    method = V01 + "/lsi/"
    data = app.post_check(method, json=lsi_pars)
    assert sorted(data.keys()) == ['explained_variance', 'id']
    lid = data['id']

    # checking again that we can load all the lsi models
    method = V01 + "/lsi/"
    data = app.get(method, data=dict(parent_id=dsid,))

    method = V01 + "/lsi/{}".format(lid)
    data = app.get_check(method)
    for key, vals in lsi_pars.items():
        assert vals == data[key]

    assert sorted(data.keys()) == sorted(["n_components", "parent_id"])

    for key in data.keys():
        assert data[key] == lsi_pars[key]

