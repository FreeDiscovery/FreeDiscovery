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

from ..server import fd_app
from ..utils import _silent
from ..exceptions import OptionalDependencyMissing
from .run_suite import check_cache

V01 = '/api/v0'


data_dir = os.path.dirname(__file__)
email_data_dir = os.path.join(data_dir, "..", "data", "fedora-devel-list-2008-October")
data_dir = os.path.join(data_dir, "..", "data", "ds_001", "raw")

#=============================================================================#
#
#                     Helper functions / features
#
#=============================================================================#


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



#=============================================================================#
#
#                     Feature extraction
#
#=============================================================================#

def get_features(app, hashed=True):
    method = V01 + "/feature-extraction/"
    pars = dict(data_dir=data_dir, n_features=100000,
                analyzer='word', stop_words='None',
                ngram_range=[1, 1], use_hashing=hashed)
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


def get_features_lsi(app, hashed=True):
    dsid, pars = get_features(app, hashed=hashed)
    lsi_pars = dict( n_components=101, parent_id=dsid)
    method = V01 + "/lsi/"
    res = app.post(method, data=lsi_pars)
    assert res.status_code == 200
    data = parse_res(res)
    assert sorted(data.keys()) == ['explained_variance', 'id']
    lsi_id = data['id']
    return dsid, lsi_id, pars


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
        assert sorted(row.keys()) == \
                 sorted(['analyzer', 'ngram_range', 'stop_words',
                     'n_jobs', 'chunk_size', 'norm',
                     'data_dir', 'id', 'n_samples', 'n_features', 'use_idf',
                     'binary', 'sublinear_tf', 'use_hashing'])


def test_get_feature_extraction(app):
    dsid, _ = get_features(app)
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
    dsid, _ = get_features(app)

    method = V01 + "/feature-extraction/{}/index".format(dsid)
    for pars, indices in [
            ({ 'filenames': ['0.7.47.101442.txt', '0.7.47.117435.txt']}, [0, 1]),
            ({ 'filenames': ['0.7.6.28638.txt']}, [5])]:

        res = app.get(method, data=pars)
        assert res.status_code == 200
        data = parse_res(res)
        assert sorted(data.keys()) ==  sorted(['index'])
        assert_equal(data['index'], indices)

#=============================================================================#
#
#                     Email Parsing
#
#=============================================================================#

def parse_emails(app):
    method = V01 + "/email-parser/"
    pars = dict(data_dir=email_data_dir)

    res = app.post(method, data=pars)

    assert res.status_code == 200, method
    data = parse_res(res)
    assert sorted(data.keys()) ==  ['filenames', 'id']
    dsid = data['id']

    return dsid, pars

def test_parse_emails(app):
    dsid, pars = parse_emails(app)

    method = V01 + "/email-parser/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200, method
    data = parse_res(res)
    for key, val in pars.items():
        if key in ['data_dir']:
            continue
        assert val == data[key]


def test_delete_parsed_emails(app):
    dsid, _ = parse_emails(app)

    method = V01 + "/email-parser/{}".format(dsid)
    res = app.delete(method)
    assert res.status_code == 200


def test_get_email_parser_all(app):
    method = V01 + "/email-parser/"
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)
    for row in data:
        assert sorted(row.keys()) == sorted([ 'data_dir', 'id', 'encoding', 'n_samples']) 


def test_get_email_parser(app):
    dsid, _ = parse_emails(app)
    method = V01 + "/email-parser/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)
    assert sorted(data.keys()) == \
             sorted(['data_dir', 'filenames', 'encoding', 'n_samples', 'type'])


def test_get_search_emails_by_filename(app):
    dsid, _ = parse_emails(app)

    method = V01 + "/email-parser/{}/index".format(dsid)
    for pars, indices in [
            ({ 'filenames': ['1', '2']}, [0, 1]),
            ({ 'filenames': ['5']}, [4])]:

        res = app.get(method, data=pars)
        assert res.status_code == 200
        data = parse_res(res)
        assert sorted(data.keys()) ==  sorted(['index'])
        assert_equal(data['index'], indices)

#=============================================================================#
#
#                     LSI
#
#=============================================================================#

def test_api_lsi(app):
    dsid, _ = get_features(app)
    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)
    method = V01 + "/lsi/"
    res = app.get(method,
            data=dict(
                parent_id=dsid,
                )
            )
    assert res.status_code == 200

    lsi_pars = dict( n_components=101, parent_id=dsid)
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
                parent_id=dsid,
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


_categoriazation_pars = itertools.product( ["LinearSVC", "LogisticRegression",
                                            "NearestCentroid", "NearestNeighbor", 'xgboost'],
                                            [0, 1])

_categoriazation_pars = filter(lambda el: not ((el[0].startswith('Nearest') and el[1])),
                               _categoriazation_pars)

@pytest.mark.parametrize("solver, cv", _categoriazation_pars)
def test_api_categorization(app, solver, cv):

    if 'CIRCLECI' in os.environ and cv == 1 and solver in ['LinearSVC', 'xgboost']:
        raise SkipTest # Circle CI is too slow and timesout

    if solver.startswith('Nearest'):
        dsid, lsi_id, _ = get_features_lsi(app)
        parent_id = lsi_id
    else:
        dsid, _ = get_features(app)
        lsi_id = None
        parent_id = dsid

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)

    filenames = data['filenames']
    # we train the model on 5 samples / 6 and see what happens
    index_filenames = filenames[:3] + filenames[3:]
    y = [1, 1, 1,  0, 0, 0]

    method = V01 + "/feature-extraction/{}/index".format(dsid)
    res = app.get(method, data={'filenames': index_filenames})
    assert res.status_code == 200, method
    index = parse_res(res)['index']


    pars = {
          'parent_id': parent_id,
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
    if solver == 'NearestNeighbor':
        assert sorted(data.keys()) == ['dist_n', 'dist_p', 'ind_n', 'ind_p', 'prediction']
    else:
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

#=============================================================================#
#
#                     Clustering
#
#=============================================================================#

@pytest.mark.parametrize("model, use_lsi", [('k-mean', False),
                                            ('birch', True),
                                            ('ward_hc', True),
                                            ('dbscan', True)])
def test_api_clustering(app, model, use_lsi):

    if use_lsi:
        dsid, lsi_id, _ = get_features_lsi(app, hashed=False)
        parent_id = lsi_id
    else:
        dsid, _ = get_features(app, hashed=False)
        lsi_id = None
        parent_id = dsid

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)  # TODO unused variable

    #if (model == 'birch' or model == "ward_hc"):

    url = V01 + "/clustering/" + model
    pars = { 'parent_id': parent_id, }
    if model != 'dbscan':
        pars['n_clusters'] = 2
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

#=============================================================================#
#
#                     Duplicates detection
#
#=============================================================================#


@pytest.mark.parametrize('kind, options', [['simhash', {'distance': 3}],
                                             ['i-match', {}]])
def test_api_dupdetection(app, kind, options):

    if kind == 'simhash':
        try:
            import simhash
        except ImportError:
            raise SkipTest

    dsid, pars = get_features(app, hashed=False)

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)  # TODO unused variable

    url = V01 + "/duplicate-detection" 
    pars = { 'parent_id': dsid,
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
    assert sorted(data.keys()) == sorted(['cluster_id'])

    res = app.delete(method)
    assert res.status_code == 200


#=============================================================================#
#
#                     Duplicates detection
#
#=============================================================================#


def test_api_thread_emails(app):

    dsid, _ = parse_emails(app)

    method = V01 + "/email-parser/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)  # TODO unused variable

    url = V01 + "/email-threading" 
    pars = { 'parent_id': dsid }
             
    res = app.post(url, data=pars)
    assert res.status_code == 200
    data = parse_res(res)
    assert sorted(data.keys()) == sorted(['data', 'id'])
    mid = data['id']

    tree_ref = [ {'id': 0, 'parent': None, 'children': [
                  {'id': 1, 'children': [], 'parent': 0},
                  {'id': 2, 'parent': 0,  'children': [
                         {'id': 3, 'children': [], 'parent': 2},
                         {'id': 4, 'children': [], 'parent': 2}],
                         }]
                  }]

    def remove_subject_field(d):
        del d['subject']
        for el in d['children']:
            remove_subject_field(el)

    tree_res = data['data']
    for el in tree_res:
        remove_subject_field(el)

    assert data['data'] == tree_ref

    url += '/{}'.format(mid)
    res = app.get(url)
    assert res.status_code == 200
    data = parse_res(res)
    assert sorted(data.keys()) == sorted(['group_by_subject'])

    res = app.delete(method)
    assert res.status_code == 200



#=============================================================================#
#
#                     Exception handling
#
#=============================================================================#


@pytest.mark.parametrize("method", ['feature-extraction', 'categorization', 'lsi', 'clustering'])
def test_get_model_404(app_notest, method):
    method = V01 + "/{}/DOES_NOT_EXISTS".format(method)
    with _silent('stderr'):
        res = app_notest.get(method)

    data = parse_res(res)
    assert res.status_code in [500, 404] # depends on the url
    #assert '500' in data['message']

    assert sorted(data.keys()) == sorted(['message'])


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
    dsid, _ = get_features(app_notest)

    method = V01 + "/categorization/"
    with _silent('stderr'):
        res = app_notest.post(method,
                        data={
                              'parent_id': dsid,
                              'index': [0, 0, 0],       # just something wrong
                              'y': ['ds', 'dsd', 'dsd'],
                              'method': "LogisticRegression",
                              'cv': 0,
                              })
    data = parse_res(res)
    assert res.status_code in [500, 422]
    assert sorted(data.keys()) == ['messages']
    #assert 'ValueError' in data['message'] # check that the error message has the traceback



@pytest.mark.parametrize('metrics',
                         itertools.combinations(['precision', 'recall', 'f1', 'roc_auc'], 3))
def test_categorization_metrics_get(app, metrics):
    metrics = list(metrics)
    url = V01 + '/metrics/categorization'
    y_true = [0, 0, 0, 1, 1, 0, 1, 0]
    y_pred = [0, 0, 1, 1, 1, 0, 1, 1]

    pars = {'y_true': y_true, 'y_pred': y_pred, 'metrics': metrics}
    res = app.get(url, data=pars)
    assert res.status_code == 200

    data = parse_res(res)
    assert sorted(data.keys()) == sorted(metrics)
    if 'precision' in metrics:
        assert_almost_equal(data['precision'], 0.6)
    if 'recall' in metrics:
        assert_almost_equal(data['recall'], 1.0)
    if 'f1' in metrics:
        assert_almost_equal(data['f1'], 0.75)
    if 'roc_auc' in metrics:
        assert_almost_equal(data['roc_auc'], 0.8)


@pytest.mark.parametrize('metrics',
                         itertools.combinations(['adjusted_rand', 'adjusted_mutual_info', 'v_measure'], 2))
def test_clustering_metrics_get(app, metrics):
    metrics = list(metrics)
    url = V01 + '/metrics/clustering'
    labels_true = [0, 0, 1, 2]
    labels_pred = [0, 0, 1, 1]

    pars = {'labels_true': labels_true, 'labels_pred': labels_pred, 'metrics': metrics}
    res = app.get(url, data=pars)
    assert res.status_code == 200

    data = parse_res(res)
    assert sorted(data.keys()) == sorted(metrics)
    if 'adjusted_rand' in metrics:
        assert_almost_equal(data['adjusted_rand'], 0.5714, decimal=4)
    if 'adjusted_mutual_info' in metrics:
        assert_almost_equal(data['adjusted_mutual_info'], 0.4)
    if 'v_measure' in metrics:
        assert_almost_equal(data['v_measure'], 0.8)


@pytest.mark.parametrize('metrics',
                         itertools.combinations(['ratio_duplicates', 'f1_same_duplicates', 'mean_duplicates_count'], 2))
def test_dupdetection_metrics_get(app, metrics):
    metrics = list(metrics)
    url = V01 + '/metrics/duplicate-detection'
    labels_true = [0, 1, 1, 2, 3, 2]
    labels_pred = [0, 1, 3, 2, 5, 2]

    pars = {'labels_true': labels_true, 'labels_pred': labels_pred, 'metrics': metrics}
    res = app.get(url, data=pars)
    assert res.status_code == 200

    data = parse_res(res)
    assert sorted(data.keys()) == sorted(metrics)
    if 'ratio_duplicates' in metrics:
        assert_almost_equal(data['ratio_duplicates'], 0.5)
    if 'f1_same_duplicates' in metrics:
        assert_almost_equal(data['f1_same_duplicates'], 0.667, decimal=3)
    if 'mean_duplicates_count' in metrics:
        assert_almost_equal(data['mean_duplicates_count'], 0.75)
