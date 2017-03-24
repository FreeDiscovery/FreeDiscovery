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
from ...exceptions import OptionalDependencyMissing, NotFound
from ...tests.run_suite import check_cache
from .base import (parse_res, V01, app, app_notest, get_features,
               email_data_dir, get_features_cached)


#=============================================================================#
#
#                     Feature extraction
#
#=============================================================================#

def test_get_features(app):
    dsid, pars, _ = get_features_cached(app)

    method = V01 + "/feature-extraction/{}".format(dsid)
    data = app.get_check(method)
    for key, val in pars.items():
        if key in ['data_dir', 'dataset_definition']:
            continue
        assert val == data[key]

def test_delete_feature_extraction(app):
    dsid, _ = get_features(app)

    method = V01 + "/feature-extraction/{}".format(dsid)
    app.delete_check(method)


def test_get_feature_extraction_all(app):
    method = V01 + "/feature-extraction/"
    data = app.get_check(method)
    for row in data:
        del row['norm']
        assert sdict_keys(row) == sdict_keys({'analyzer': 'str',
                     'ngram_range': ['int'], 'stop_words': 'NoneType',
                     'n_jobs': 'int', 'chunk_size': 'int',
                     'data_dir': 'str', 'id': 'str', 'n_samples': 'int',
                     'n_features': 'int', 'use_idf': 'bool',
                     'binary': 'bool', 'sublinear_tf': 'bool', 'use_hashing': 'bool'})


@pytest.mark.parametrize('hashed', [True])
def test_get_feature_extraction(app, hashed):
    dsid, _, _ = get_features_cached(app, hashed=hashed)
    method = V01 + "/feature-extraction/{}".format(dsid)
    data = app.get_check(method)
    assert dict2type(data, collapse_lists=True) == {'analyzer': 'str',
                     'ngram_range': ['int'], 'stop_words': 'NoneType',
                     'n_jobs': 'int', 'chunk_size': 'int', 'norm': 'str',
                     'data_dir': 'str', 'n_samples': 'int',
                     'n_features': 'int', 'use_idf': 'bool',
                     'binary': 'bool', 'sublinear_tf': 'bool', 'use_hashing': 'bool',
                     'filenames': ['str'], 'max_df': 'float', 'min_df': 'float',
                     'n_samples_processed': 'int'}

@pytest.mark.parametrize('hashed', [True])
def test_stop_words_integration(app, hashed):
    url = V01 + '/stop-words/'

    sw_name = 'test1w'
    pars = {'name': sw_name,
            'stop_words': ['and', 'or', 'in']}

    res = app.post_check(url, json=pars)
    assert dict2type(res, collapse_lists=True) == {'name' : 'str'}
    assert res['name'] == sw_name

    res = app.get_check(url + sw_name)
    assert dict2type(res, collapse_lists=True) == {'name' : 'str',
                                                   'stop_words': ['str']}
    assert res['name'] == sw_name
    assert res['stop_words'] == pars['stop_words']

    dsid, _ = get_features(app, hashed=hashed, stop_words=sw_name)



def test_get_search_filenames(app):

    dsid, _, _ = get_features_cached(app)

    method = V01 + "/feature-extraction/{}/id-mapping".format(dsid)

    def _filter_dict(x, filter_field):
        return {key: val for key, val in x.items() if key == filter_field}

    response_ref = {'internal_id': 'int',
                    'file_path' : 'str',
                    'document_id': 'int'}

    # Query 1
    file_path_obj  = [{'file_path': val} for val in ['00401.txt', '00506.txt']]
    data = app.post_check(method, json={'data': file_path_obj})
    data = data['data']

    for idx in range(len(data)):
        assert dict2type(data[idx]) == response_ref
    assert [_filter_dict(row, 'file_path') for row in data] == file_path_obj
    assert_equal(np.asarray([row['internal_id'] for row in data])**2,
                 [row['document_id'] for row in data])


    with pytest.raises(NotFound):
        res = app.post(method, json={'data': [{'file_path': '00400.txt'}]})

    # Query 2
    file_path_obj  = [{'document_id': 4 }, {'document_id': 9}]
    data = app.post_check(method, json={'data': file_path_obj})
    data = data['data']

    for idx in range(len(data)):
        assert dict2type(data[idx]) == response_ref
    assert [_filter_dict(row, 'document_id') for row in data] == file_path_obj
    assert_equal(np.asarray([row['internal_id'] for row in data])**2,
                 [row['document_id'] for row in data])


