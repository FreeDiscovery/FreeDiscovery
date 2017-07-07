# -*- coding: utf-8 -*-

import pytest
import json
import os.path

import numpy as np

from .. import fd_app
from ...tests.run_suite import check_cache
from ...utils import dict2type

from sklearn.externals.joblib import Memory

V01 = '/api/v0'

data_dir = os.path.dirname(__file__)
email_data_dir = os.path.join(data_dir, "..", "..", "data", "fedora-devel-list-2008-October")
data_dir = os.path.join(data_dir, "..", "..", "data", "ds_001", "raw")
CACHE_DIR = check_cache()

np.random.seed(43)


def parse_res(res):
    return json.loads(res.data.decode('utf-8'))


def app_call_wrapper(func):
    """Wrapp the POST, GET, DELETE methods of flask.testing.FlaskClient
    in order to make the response_code checks and convert the
    json output to dict that we always need to do
    """

    def inner_function(*args, valid_return_codes=None, **kwargs):
        if valid_return_codes is None:
            valid_return_codes = [200]
        res = func(*args, **kwargs)
        assert res.status_code in valid_return_codes, args[0]
        data = parse_res(res)
        return data
    return inner_function


@pytest.fixture
def app():
    tapp = fd_app(CACHE_DIR)
    tapp.config['TESTING'] = True

    client = tapp.test_client()
    client.post_check = app_call_wrapper(client.post)
    client.get_check = app_call_wrapper(client.get)
    client.delete_check = app_call_wrapper(client.delete)
    return client


@pytest.fixture
def app_notest():
    tapp = fd_app(CACHE_DIR)
    tapp.config['TESTING'] = False

    client = tapp.test_client()
    client.post_check = app_call_wrapper(client.post)
    client.get_check = app_call_wrapper(client.get)
    client.delete_check = app_call_wrapper(client.delete)
    return client


memory = Memory(cachedir=os.path.join(CACHE_DIR, '_joblib_cache'), verbose=0)

# ===========================================================================#
#
#                     Feature extraction
#
# ===========================================================================#


def get_features(app, hashed=False, metadata_fields='data_dir',
                 n_categories=2, dataset='20_newsgroups_3categories', **kwargs):

    method = V01 + "/feature-extraction/"
    pars = {'use_hashing': hashed}
    pars.update(kwargs)
    data = app.post_check(method, json=pars)

    assert dict2type(data, collapse_lists=True) == {'id': 'str'}
    dsid = data['id']

    pars = {}
    if not kwargs.get('parse_email_headers'):
        url = V01 + '/example-dataset/{}'.format(dataset)
        res = app.get(url, json={'n_categories': n_categories})
        assert res.status_code == 200, url
        input_ds = parse_res(res)
        data_dir = input_ds['metadata']['data_dir']
        pars['dataset_definition'] = [{'document_id': row['document_id'],
                                       'file_path': os.path.join(data_dir,
                                                                 row['file_path'])}
                                      for row in input_ds['dataset']]
    else:
        pars['data_dir'] = email_data_dir
        input_ds = None

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.post_check(method, json=pars)
    assert dict2type(res) == {'id': 'str'}
    pars = app.get_check(method)
    pars.pop('filenames')
    pars.pop('dataset_definition', None)
    return dsid, pars, input_ds


@memory.cache(ignore=['app'])
def get_features_cached(app, hashed=False, n_categories=2,
                        dataset='20_newsgroups_3categories', **kwargs):
    return get_features(app, hashed=hashed, n_categories=n_categories,
                        dataset=dataset, **kwargs)


def get_features_lsi(app, hashed=False, metadata_fields='data_dir', **kwargs):
    dsid, pars = get_features(app, hashed=hashed,
                              metadata_fields=metadata_fields, **kwargs)
    lsi_pars = dict(n_components=201, parent_id=dsid)
    method = V01 + "/lsi/"
    res = app.post(method, json=lsi_pars)
    assert res.status_code == 200
    data = parse_res(res)
    assert dict2type(data) == {'explained_variance': 'float', 'id': 'str'}
    lsi_id = data['id']
    return dsid, lsi_id, pars


@memory.cache(ignore=['app'])
def get_features_lsi_cached(app, hashed=False, n_categories=2, n_components=201,
                            dataset="20_newsgroups_3categories"):
    dsid, pars, input_ds = get_features_cached(app, hashed=hashed,
                                               n_categories=n_categories)
    lsi_pars = dict(n_components=n_components, parent_id=dsid)
    method = V01 + "/lsi/"
    res = app.post(method, json=lsi_pars)
    assert res.status_code == 200
    data = parse_res(res)
    assert dict2type(data) == {'explained_variance': 'float', 'id': 'str'}
    lsi_id = data['id']
    return dsid, lsi_id, pars, input_ds
