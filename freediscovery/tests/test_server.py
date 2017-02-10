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
from ..ingestion import DocumentIndex
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

def _internal2document_id(value):
    """A custom internal_id to document_id mapping used in tests"""
    return 2*value + 1

def _document2internal_id(value):
    """A custom internal_id to document_id mapping used in tests"""
    return (value - 1)//2

def test_consistent_id_mapping():
    internal_id = 2
    document_id = _internal2document_id(internal_id)
    assert _document2internal_id(document_id) == internal_id


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

def get_features(app, hashed=True, ingestion_method='data_dir'):
    method = V01 + "/feature-extraction/"
    pars = { "use_hashing": hashed}
    if ingestion_method == 'data_dir':
        pars["data_dir"] = data_dir
    elif ingestion_method == 'dataset_definition':

        index = DocumentIndex.from_folder(data_dir)
        pars["dataset_definition"] = []
        for idx, file_path in enumerate(index.filenames):
            row = {'file_path': file_path,
                   'document_id': _internal2document_id(idx)}
            pars["dataset_definition"].append(row)
    elif ingestion_method is None:
        pass # don't provide data_dir and dataset_definition
    else:
        raise NotImplementedError('ingestion_method={} is not implemented')


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


def get_features_lsi(app, hashed=True, ingestion_method='data_dir'):
    dsid, pars = get_features(app, hashed=hashed,
                              ingestion_method=ingestion_method)
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


        res = app.get(method, data=pars)
        assert res.status_code == 200
        data = parse_res(res)
        if return_file_path:
            assert sorted(data.keys()) ==  sorted(['internal_id', 'file_path'])
        else:
            assert sorted(data.keys()) ==  sorted(['internal_id'])
        assert_equal(data['internal_id'], indices)


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


_categoriazation_pars = itertools.product(  ['data_dir'],
                                            ["LinearSVC", "LogisticRegression",
                                            "NearestCentroid", "NearestNeighbor", 'xgboost'],
                                            ['', 'cv'])

_categoriazation_pars = filter(lambda args: not (args[1].startswith('Nearest') and args[2] == 'cv'),
                               _categoriazation_pars)


def _api_categorization_wrapper(app, ingestion_method, solver, cv, supervision_mode):

    cv = (cv == 'cv')

    if 'CIRCLECI' in os.environ and cv == 1 and solver in ['LinearSVC', 'xgboost']:
        raise SkipTest # Circle CI is too slow and timesout

    if solver.startswith('Nearest'):
        dsid, lsi_id, _ = get_features_lsi(app, ingestion_method=ingestion_method)
        parent_id = lsi_id
    else:
        dsid, _ = get_features(app, ingestion_method=ingestion_method)
        lsi_id = None
        parent_id = dsid

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)

    filenames = data['filenames']
    # we train the model on 5 samples / 6 and see what happens
    index_filenames = filenames[:3] + filenames[3:]
    if supervision_mode == 'supervised':
        y = [1, 1, 1, 0, 0, 0]
    else:
        y = [1, 1, 1, 1, 1, 1]


    method = V01 + "/feature-extraction/{}/id-mapping/flat".format(dsid)
    res = app.get(method, data={'file_path': index_filenames})
    assert res.status_code == 200, method
    index = parse_res(res)['internal_id']


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
    assert sorted(data.keys()) == ['data']
    assert len(data['data']) == len(y)
    if solver == 'NearestNeighbor':
        for row in data['data']:
            assert sorted(row.keys()) == sorted(['internal_id', 'score',
                                                 'nn_positive', 'nn_negative'])
            nn_p = row['nn_positive']
            assert sorted(nn_p.keys()) == sorted(['internal_id', 'distance'])
    else:
        for row in data['data']:
            assert sorted(row.keys()) == sorted(['internal_id', 'score'])

    if supervision_mode == 'supervised':
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

@pytest.mark.parametrize("ingestion_method, solver, cv", _categoriazation_pars)
def test_api_categorization(app, ingestion_method, solver, cv):
    _api_categorization_wrapper(app, ingestion_method, solver, cv, 'supervised')

@pytest.mark.parametrize("ingestion_method", ['data_dir', "dataset_definition", None])
def test_api_categorization_ingestion_method(app, ingestion_method):
    
    if ingestion_method == 'data_dir':
        _api_categorization_wrapper(app, ingestion_method, 'LogisticRegression', '', 'supervised')
    elif ingestion_method == 'dataset_definition':
        # Internal testing doesn't currently support nested dicts cf.
        # https://github.com/pallets/werkzeug/blob/master/werkzeug/test.py#L233
        raise SkipTest
    else:
        with pytest.raises(ValueError):
            _api_categorization_wrapper(app, ingestion_method, 'LogisticRegression', False, 'supervised')


@pytest.mark.parametrize('supervision_mode', ['supervised', 'unsupervised'])
def test_api_categorization_supervision_mode(app, supervision_mode):
    _api_categorization_wrapper(app, 'data_dir', 'NearestNeighbor', '', supervision_mode)


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
                         itertools.combinations(['precision', 'recall', 'f1',
                                                 'roc_auc', 'average_precision'], 3))
def test_categorization_metrics_get(app, metrics):
    metrics = list(metrics)
    url = V01 + '/metrics/categorization'
    y_true = [0, 0, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0.1, 0, 1, 1, 1, 0, 1, 1, 1]

    pars = {'y_true': y_true, 'y_pred': y_pred, 'metrics': metrics}
    res = app.get(url, data=pars)
    assert res.status_code == 200

    data = parse_res(res)
    assert sorted(data.keys()) == sorted(metrics)
    for key in metrics:
        assert data[key] > 0.5
        assert data[key] <= 1.0



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



@pytest.mark.parametrize("method", ['regular', 'semantic'])
def test_api_clustering(app, method):

    if method == 'regular':
        dsid, lsi_id, _ = get_features_lsi(app, hashed=False)
        parent_id = lsi_id
    elif method == 'semantic':
        dsid, _ = get_features(app, hashed=False)
        lsi_id = None
        parent_id = dsid

    method = V01 + "/search/"
    res = app.get(method, data=dict(parent_id=parent_id, query="so that I can reserve a room"))
    assert res.status_code == 200
    data = parse_res(res)
    assert sorted(data.keys()) == ['data']
    for row in data['data']:
        assert sorted(row.keys()) == sorted(['score', 'internal_id'])
