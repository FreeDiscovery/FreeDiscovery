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

from ..server import fd_app
from ..utils import _silent
from ..exceptions import OptionalDependencyMissing
from .run_suite import check_cache
from numpy.testing import assert_equal

V01 = '/api/v0'


data_dir = os.path.dirname(__file__)
data_dir = os.path.join(data_dir, "..", "data", "ds_001", "raw")


def parse_res(res):
    return json.loads(res.data.decode('utf-8'))


@pytest.fixture
def app():

    cache_dir = check_cache()

    tapp = fd_app(cache_dir)

    tapp.config['TESTING'] = True
    return tapp.test_client()


@pytest.fixture
def app_notest():
    cache_dir = check_cache()
    tapp = fd_app(cache_dir)
    tapp.config['TESTING'] = False

    return tapp.test_client()


def features_hashed(app):
    method = V01 + "/feature-extraction/"
    pars = dict(data_dir=data_dir, n_features=100000,
                analyzer='word', stop_words='None',
                ngram_range=[1, 1])
    res = app.post(method, data=pars)

    assert res.status_code == 200, method
    data = parse_res(res)
    assert sorted(data.keys()) ==  ['filenames', 'id']
    dsid = data['id']

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.post(method)
    assert res.status_code == 200, method
    data = parse_res(res)
    assert sorted(data.keys()) == ['id']
    return dsid, pars


def test_features_hashed(app):
    dsid, pars = features_hashed(app)

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200, method
    data = parse_res(res)
    for key, val in pars.items():
        if key in ['data_dir']:
            continue
        assert val == data[key]


def features_non_hashed(app):
    method = V01 + "/feature-extraction/"
    res = app.post(method,
            data=dict(data_dir=data_dir, n_features=100000,
                analyzer='word', stop_words='english',
                ngram_range=[1, 1], use_hashing=0))

    assert res.status_code == 200, method
    data = parse_res(res)
    assert sorted(data.keys()) == ['filenames', 'id']
    dsid = data['id']

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.post(method)
    assert res.status_code == 200, method
    data = parse_res(res)
    assert sorted(data.keys()) == ['id']
    return dsid


def test_api_lsi(app):
    dsid, _ = features_hashed(app)
    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)
    method = V01 + "/lsi/"
    res = app.get(method,
            data=dict(
                dataset_id=dsid,
                )
            )
    assert res.status_code == 200

    filenames = data['filenames']
    # we train the model on 5 samples / 6 and see what happens
    index_filenames = filenames[:1] + filenames[3:]
    y = [1, 1,  0, 0, 0]

    method = V01 + "/feature-extraction/{}/index".format(dsid)
    res = app.get(method, data={'filenames': index_filenames})
    assert res.status_code == 200, method
    index = parse_res(res)['indices']

    lsi_pars = dict( n_components=101, dataset_id=dsid)
    method = V01 + "/lsi/"
    res = app.post(method, data=lsi_pars)
    assert res.status_code == 200
    data = parse_res(res)
    assert sorted(data.keys()) == ['explained_variance', 'id']
    lid = data['id']


    # checking again that we can load all the lsi models
    method = V01 + "/lsi/"
    res = app.get(method,
            data=dict(
                dataset_id=dsid,
                )
            )
    assert res.status_code == 200
    data = parse_res(res)  # TODO unused variable

    method = V01 + "/lsi/{}".format(lid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)
    for key, vals in lsi_pars.items():
        assert vals == data[key]


    method = V01 + "/lsi/{}/predict".format(lid)
    res = app.post(method,
            data={
              'index': index,
              'y': y,
              })

    assert res.status_code == 200
    data = parse_res(res)
    assert sorted(data.keys()) == sorted(['prediction',
        'prediction_rel', 'prediction_nrel',
        'nearest_rel_doc', 'nearest_nrel_doc',
        'f1', 'recall', 'precision', 'roc_auc', 'average_precision'])

    method = V01 + "/lsi/{}/test".format(lid)
    res = app.post(method,
            data={
              'relevant_id': relevant_id,
              'non_relevant_id': non_relevant_id,
              'ground_truth_filename': os.path.join(data_dir, '..', 'ground_truth_file.txt')
              })

    assert res.status_code == 200
    data = parse_res(res)
    assert sorted(data.keys()) == sorted(['precision', 'recall',
                                          'f1', 'roc_auc', 'average_precision'])


@pytest.mark.parametrize("solver,cv", itertools.product(
                   ["LinearSVC", "LogisticRegression", 'xgboost'],
                   [0, 1]))
def test_api_categorization(app, solver, cv):

    if 'CIRCLECI' in os.environ and cv == 1 and solver in ['LinearSVC', 'xgboost']:
        raise SkipTest # Circle CI is too slow and timesout

    dsid, _ = features_hashed(app)

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)

    filenames = data['filenames']
    # we train the model on 5 samples / 6 and see what happens
    index_filenames = filenames[:2] + filenames[3:]
    y = [1, 1,  0, 0, 0]

    method = V01 + "/feature-extraction/{}/index".format(dsid)
    res = app.get(method, data={'filenames': index_filenames})
    assert res.status_code == 200, method
    index = parse_res(res)['indices']


    pars = {
          'dataset_id': dsid,
          'index': index,
          'y': y,
          'method': solver,
          'cv': cv}

    method = V01 + "/categorization/"
    try:
        res = app.post(method, data=pars)
    except OptionalDependencyMissing:
        raise SkipTest

    data = parse_res(res)
    assert res.status_code == 200, method
    assert sorted(data.keys()) == sorted(['id', 'recall',
                                          'f1', 'precision', 'roc_auc', 'average_precision'])
    mid = data['id']

    method = V01 + "/categorization/{}".format(mid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)
    assert sorted(data.keys()) == \
            sorted(["index", "y",
                    "method", "options"])

    for key in ["index", "y", "method"]:
        if key in ['index', 'y']:
            assert len(pars[key]) == len(data[key])
        else:
            assert pars[key] == data[key]

    method = V01 + "/categorization/{}/predict".format(mid)
    res = app.get(method)
    data = parse_res(res)
    assert sorted(data.keys()) == ['prediction']

    method = V01 + "/categorization/{}/test".format(mid)
    res = app.post(method,
            data={'ground_truth_filename':
                os.path.join(data_dir, '..', 'ground_truth_file.txt')})
    data = parse_res(res)
    assert sorted(data.keys()) == sorted(['precision', 'recall',
                        'f1', 'roc_auc', 'average_precision'])

    method = V01 + "/categorization/{}".format(mid)
    res = app.delete(method)
    assert res.status_code == 200


@pytest.mark.parametrize("model", ['k-mean', 'birch', 'ward_hc', 'dbscan'])
def test_api_clustering(app, model):

    dsid = features_non_hashed(app)

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)  # TODO unused variable

    for lsi_components in [4, -1]:
        if lsi_components == -1 and (model == 'birch' or model == "ward_hc"):
            continue
        url = V01 + "/clustering/" + model
        pars = { 'dataset_id': dsid, }
        if model != 'dbscan':
            pars['n_clusters'] = 2
        if model != "k-mean":
           pars["lsi_components"] = lsi_components 
        if model == 'dbscan':
            pars.update({'eps': 0.1, "min_samples": 2})
        res = app.post(url, data=pars)

        assert res.status_code == 200
        data = parse_res(res)
        assert sorted(data.keys()) == sorted(['id'])
        mid = data['id']

        url += '/{}'.format(mid)
        res = app.get(url)
        assert res.status_code == 200
        data = parse_res(res)
        assert sorted(data.keys()) == \
                sorted(['cluster_terms', 'labels', 'pars', 'htree'])

        if data['htree']:
            assert sorted(data['htree'].keys()) == \
                    sorted(['n_leaves', 'n_components', 'children'])

    res = app.delete(method)
    assert res.status_code == 200


@pytest.mark.parametrize('kind, options', [['simhash', {'distance': 3}],
                                             ['i-match', {}]])
def test_api_dupdetection(app, kind, options):

    if kind == 'simhash':
        try:
            import simhash
        except ImportError:
            raise SkipTest

    dsid = features_non_hashed(app)

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)  # TODO unused variable

    url = V01 + "/duplicate-detection" 
    pars = { 'dataset_id': dsid,
             'method': kind}
    res = app.post(url, data=pars)
    assert res.status_code == 200
    data = parse_res(res)
    assert sorted(data.keys()) == sorted(['id'])
    mid = data['id']

    url += '/{}'.format(mid)
    res = app.get(url, data=options)
    assert res.status_code == 200
    data = parse_res(res)
    assert sorted(data.keys()) == \
            sorted(['cluster_id'])

    res = app.delete(method)
    assert res.status_code == 200


def test_delete_feature_extraction(app):
    dsid, _ = features_hashed(app)

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.delete(method)
    assert res.status_code == 200


def test_get_feature_extraction_all(app):
    method = V01 + "/feature-extraction/"
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)
    for row in data:
        assert sorted(row.keys()) == \
                 sorted(['analyzer', 'ngram_range', 'stop_words',
                     'n_jobs', 'chunk_size', 'norm',
                     'data_dir', 'id', 'n_samples', 'n_features', 'use_idf',
                     'binary', 'sublinear_tf', 'use_hashing',
                     'max_df', 'min_df'])


def test_get_feature_extraction(app):
    dsid, _ = features_hashed(app)
    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)
    assert sorted(data.keys()) == \
             sorted(['data_dir', 'filenames', 'n_samples', 'norm',
                 'n_samples_processed', 'n_features', 'n_jobs', 'chunk_size',
                 'analyzer', 'ngram_range', 'stop_words', 'use_idf',
                 'binary', 'sublinear_tf', 'use_hashing',
                 'max_df', 'min_df'])


def test_get_search_filenames(app):
    dsid, _ = features_hashed(app)
    method = V01 + "/feature-extraction/{}/index".format(dsid)
    for pars, indices in [
            ( { 'filenames': ['0.7.47.101442.txt', '0.7.47.117435.txt']}, [0, 1]),
            ({ 'filenames': ['0.7.6.28638.txt']}, [5])]:

        res = app.get(method, data=pars)
        assert res.status_code == 200
        data = parse_res(res)
        assert sorted(data.keys()) ==  sorted(['indices'])
        assert_equal(sorted(data['indices']), indices)


@pytest.mark.parametrize("method", ['feature-extraction', 'categorization', 'lsi', 'clustering'])
def test_get_model_404(app_notest, method):
    method = V01 + "/{}/DOES_NOT_EXISTS".format(method)
    with _silent('stderr'):
        res = app_notest.get(method)

    data = parse_res(res)
    assert res.status_code in [500, 404] # depends on the url
    #assert '500' in data['message']

    assert sorted(data.keys()) == \
                    sorted(['message'])


@pytest.mark.parametrize("method", ['feature-extraction', 'categorization', 'lsi', 'clustering'])
def test_get_model_train_404(app_notest, method):
    method = V01 + "/{}/DOES_NOT_EXISTS/DOES_NOT_EXIST".format(method)
    with _silent('stderr'):
        res = app_notest.get(method)

    assert res.status_code in [404, 500]


@pytest.mark.parametrize("method", ['feature-extraction', 'categorization', 'lsi', 'clustering'])
def test_get_model_predict_404(app_notest, method):

    method = V01 + "/{0}/DOES_NOT_EXISTS/DOES_NOT_EXIST/predict".format(method)
    with _silent('stderr'):
        res = app_notest.get(method)

    assert res.status_code == 404

    method = V01 + "/{0}/DOES_NOT_EXISTS/DOES_NOT_EXIST/test".format(method)
    with _silent('stderr'):
        res = app_notest.post(method)

    assert res.status_code == 404


def test_exception_handling(app_notest):
    dsid, _ = features_hashed(app_notest)

    method = V01 + "/categorization/"
    with _silent('stderr'):
        res = app_notest.post(method,
                        data={
                              'dataset_id': dsid,
                              'non_relevant_id': [0, 0, 0],       # just something wrong
                              'relevant_id': ['ds', 'dsd', 'dsd'],
                              'method': "LogisticRegression",
                              'cv': 0,
                              })
    data = parse_res(res)
    assert res.status_code in [500, 422]
    assert sorted(data.keys()) == ['messages']
    #assert 'ValueError' in data['message'] # check that the error message has the traceback
