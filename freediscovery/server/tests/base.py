# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest
import json
import os.path

from .. import fd_app
from ...tests.run_suite import check_cache
from ...ingestion import DocumentIndex
from ...utils import dict2type, sdict_keys

from sklearn.externals.joblib import Memory

V01 = '/api/v0'

data_dir = os.path.dirname(__file__)
email_data_dir = os.path.join(data_dir, "..", "..", "data", "fedora-devel-list-2008-October")
data_dir = os.path.join(data_dir, "..", "..", "data", "ds_001", "raw")
cache_dir = check_cache()

def parse_res(res):
    return json.loads(res.data.decode('utf-8'))

def _internal2document_id(value):
    """A custom internal_id to document_id mapping used in tests"""
    return 2*value + 1

def _document2internal_id(value):
    """A custom internal_id to document_id mapping used in tests"""
    return (value - 1)//2

@pytest.fixture
def app():

    tapp = fd_app(cache_dir)

    tapp.config['TESTING'] = True
    return tapp.test_client()


@pytest.fixture
def app_notest():
    tapp = fd_app(cache_dir)
    tapp.config['TESTING'] = False

    return tapp.test_client()

memory = Memory(cachedir=os.path.join(cache_dir, '_joblib_cache', str(os.getpid())), verbose=0)

#=============================================================================#
#
#                     Feature extraction
#
#=============================================================================#

def get_features(app, hashed=True, metadata_fields='data_dir'):
    method = V01 + "/feature-extraction/"
    pars = { "use_hashing": hashed}
    if metadata_fields == 'data_dir':
        pars["data_dir"] = data_dir
    elif metadata_fields == 'dataset_definition':

        index = DocumentIndex.from_folder(data_dir)
        pars["dataset_definition"] = []
        for idx, file_path in enumerate(index.filenames):
            row = {'file_path': file_path,
                   'document_id': _internal2document_id(idx)}
            pars["dataset_definition"].append(row)
    elif metadata_fields is None:
        pass # don't provide data_dir and dataset_definition
    else:
        raise NotImplementedError('metadata_fields={} is not implemented')


    res = app.post(method, json=pars)

    assert res.status_code == 200, method
    data = parse_res(res)
    assert dict2type(data, collapse_lists=True) == {'filenames': ['str'], 'id': 'str'}
    dsid = data['id']

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.post(method)
    assert res.status_code == 200, method
    data = parse_res(res)
    assert dict2type(data) == {'id': 'str'}
    return dsid, pars

@memory.cache(ignore=['app'])
def get_features_cached(app, hashed=True, n_categories=2):
    url = V01 + '/example-dataset/20newsgroups_3categories'
    res = app.get(url, json={'n_categories': n_categories})
    assert res.status_code == 200, url
    input_ds = parse_res(res)

    pars = { "use_hashing": hashed}
    data_dir = input_ds['metadata']['data_dir']
    pars['dataset_definition'] = [{'document_id': row['document_id'],
                                   'file_path': os.path.join(data_dir, row['file_path'])} \
                                   for row in input_ds['dataset']]

    method = V01 + "/feature-extraction/"
    res = app.post(method, json=pars)

    assert res.status_code == 200, method
    data = parse_res(res)
    assert dict2type(data, collapse_lists=True) == {'filenames': ['str'], 'id': 'str'}
    dsid = data['id']

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.post(method)
    assert res.status_code == 200, method
    data = parse_res(res)
    assert dict2type(data) == {'id': 'str'}
    return dsid, pars

def get_features_lsi(app, hashed=True, metadata_fields='data_dir'):
    dsid, pars = get_features(app, hashed=hashed,
                              metadata_fields=metadata_fields)
    lsi_pars = dict( n_components=101, parent_id=dsid)
    method = V01 + "/lsi/"
    res = app.post(method, json=lsi_pars)
    assert res.status_code == 200
    data = parse_res(res)
    assert dict2type(data) == {'explained_variance': 'float', 'id': 'str'}
    lsi_id = data['id']
    return dsid, lsi_id, pars
