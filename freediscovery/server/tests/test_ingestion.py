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
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from ...utils import dict2type, sdict_keys
from ...ingestion import DocumentIndex
from ...exceptions import OptionalDependencyMissing
from ...tests.run_suite import check_cache
from .base import (parse_res, V01, app, app_notest, get_features, get_features_lsi,
               email_data_dir, get_features_cached)


#=============================================================================#
#
#                     Feature extraction
#
#=============================================================================#

def test_get_features(app):
    dsid, pars = get_features_cached(app)

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200, method
    data = parse_res(res)
    for key, val in pars.items():
        if key in ['data_dir', 'dataset_definition']:
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
    dsid, _ = get_features_cached(app)
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


def test_get_search_filenames(app):

    dsid, _ = get_features_cached(app)

    method = V01 + "/feature-extraction/{}/id-mapping".format(dsid)

    def _filter_dict(x, filter_field):
        return {key: val for key, val in x.items() if key == filter_field}

    # Query 1
    file_path_obj  = [{'file_path': val} for val in ['00401.txt', '00506.txt']]
    res = app.post(method, json={'data': file_path_obj})
    assert res.status_code == 200
    data = parse_res(res)['data']
    response_ref = {'internal_id': 'int',
                    'file_path' : 'str',
                    'document_id': 'int'}

    for idx in range(len(data)):
        assert dict2type(data[idx]) == response_ref
    assert [_filter_dict(row, 'file_path') for row in data] == file_path_obj
    assert_equal(np.asarray([row['internal_id'] for row in data])**2,
                 [row['document_id'] for row in data])


