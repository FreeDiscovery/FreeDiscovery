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

from ...utils import dict2type, sdict_keys
from ...ingestion import DocumentIndex
from ...exceptions import OptionalDependencyMissing
from ...tests.run_suite import check_cache
from .base import (parse_res, V01, app, app_notest, get_features, get_features_lsi,
               email_data_dir)


#=============================================================================#
#
#                     Feature extraction
#
#=============================================================================#

def test_get_features(app):
    dsid, pars = get_features(app)

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200, method
    data = parse_res(res)
    for key, val in pars.items():
        if key in ['data_dir']:
            continue
        assert val == data[key]


def test_delete_feature_extraction(app):
    dsid, _ = get_features(app)

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.delete(method)
    assert res.status_code == 200


def test_get_feature_extraction_all(app):
    method = V01 + "/feature-extraction/"
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)
    for row in data:
        del row['norm']
        assert sdict_keys(row) == sdict_keys({'analyzer': 'str',
                     'ngram_range': ['int'], 'stop_words': 'NoneType',
                     'n_jobs': 'int', 'chunk_size': 'int',
                     'data_dir': 'str', 'id': 'str', 'n_samples': 'int',
                     'n_features': 'int', 'use_idf': 'bool',
                     'binary': 'bool', 'sublinear_tf': 'bool', 'use_hashing': 'bool'})


def test_get_feature_extraction(app):
    dsid, _ = get_features(app)
    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)
    assert dict2type(data, collapse_lists=True) == {'analyzer': 'str',
                     'ngram_range': ['int'], 'stop_words': 'NoneType',
                     'n_jobs': 'int', 'chunk_size': 'int', 'norm': 'NoneType',
                     'data_dir': 'str', 'n_samples': 'int',
                     'n_features': 'int', 'use_idf': 'bool',
                     'binary': 'bool', 'sublinear_tf': 'bool', 'use_hashing': 'bool',
                     'filenames': ['str'], 'max_df': 'float', 'min_df': 'float',
                     'n_samples_processed': 'int'}


@pytest.mark.parametrize('return_file_path', ['return_file_path', 'dont_return_file_path'])
def test_get_search_filenames(app, return_file_path):

    return_file_path = (return_file_path == 'return_file_path')

    dsid, _ = get_features(app)

    method = V01 + "/feature-extraction/{}/id-mapping/flat".format(dsid)
    for pars, indices in [
            ({ 'file_path': ['0.7.47.101442.txt', '0.7.47.117435.txt']}, [0, 1]),
            ({ 'file_path': ['0.7.6.28638.txt']}, [5])]:
        if return_file_path:
            pars['return_file_path'] = True
        else:
            pass # default to false


        res = app.post(method, json=pars)
        assert res.status_code == 200
        data = parse_res(res)
        response_ref = {'internal_id': ['int']}

        if return_file_path:
            response_ref['file_path'] = ['str']

        assert dict2type(data, collapse_lists=True) == response_ref
        assert_equal(data['internal_id'], indices)


