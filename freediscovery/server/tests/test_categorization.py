# -*- coding: utf-8 -*-

import os
import pytest
import itertools
from unittest import SkipTest
import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import pandas as pd

from ...utils import dict2type
from ...sklearn_compat import sklearn_version
from ...exceptions import OptionalDependencyMissing
from .base import (parse_res, V01, app, app_notest, get_features, get_features_lsi,
                   get_features_lsi_cached, get_features_cached)


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
    res = app.get_check(method, data=dict(parent_id=dsid,))

    lsi_pars = dict(n_components=101, parent_id=dsid)
    method = V01 + "/lsi/"
    data = app.post_check(method, json=lsi_pars)

    assert sorted(data.keys()) == ['explained_variance', 'id']
    lid = data['id']

    # checking again that we can load all the lsi models
    method = V01 + "/lsi/"
    data = app.get_check(method, data=dict(parent_id=dsid,))
    method = V01 + "/lsi/{}".format(lid)
    data = app.get_check(method)
    for key, vals in lsi_pars.items():
        assert vals == data[key]

    assert sorted(data.keys()) == sorted(["n_components", "parent_id"])

    for key in data.keys():
        assert data[key] == lsi_pars[key]


def _api_categorization_wrapper(app, solver, cv, n_categories,
                                n_categories_train=None, max_results=None,
                                subset='all', metric='cosine'):
    cv = (cv == 'cv')

    if 'CIRCLECI' in os.environ and cv == 1 and solver in ['LinearSVC',
                                                           'xgboost']:
        raise SkipTest  # Circle CI is too slow and timesout

    if solver.startswith('Nearest'):
        dsid, lsi_id, _, ds_input = get_features_lsi_cached(app, n_categories=n_categories)
        parent_id = lsi_id
    else:
        dsid, _, ds_input = get_features_cached(app, n_categories=n_categories)
        lsi_id = None
        parent_id = dsid

    method = V01 + "/feature-extraction/{}".format(dsid)
    data = app.get_check(method)

    categories_list = list(set([row['category'] for row in ds_input['dataset']]))

    if n_categories_train is None:
        training_set = ds_input['training_set']
    else:
        assert n_categories_train <= n_categories
        training_set = list(filter(lambda x: x['category'] in categories_list[:n_categories_train],
                            ds_input['training_set']))

    pars = {
          'parent_id': parent_id,
          'data': training_set,
          'method': solver,
          'cv': cv,
          'training_scores': True}

    method = V01 + "/categorization/"
    try:
        data = app.post_check(method, json=pars)
    except OptionalDependencyMissing:
        raise SkipTest

    assert dict2type(data) == {'id': 'str',
                               'training_scores': {'recall': 'float',
                                                   'f1': 'float',
                                                   'precision': 'float',
                                                   'roc_auc': 'float',
                                                   'average_precision': 'float',
                                                   'recall_at_20p': 'float'}}

    training_scores = data['training_scores']

    # it is very likely that there is an issue in the training scores
    if n_categories_train == 1:
        if sklearn_version < (0, 18, 0):
            pass
        else:
            # this yields wrong results for sklearn 0.17
            assert training_scores['f1'] > 0.99

    elif n_categories == 2:
        assert training_scores['average_precision'] > 0.73
        if solver == 'NearestNeighbor':
            assert training_scores['roc_auc'] > 0.7
        else:
            assert training_scores['roc_auc'] >= 0.5
    elif n_categories == 3 and solver == 'NearestNeighbor':
        assert training_scores['f1'] > 0.6
    else:
        assert training_scores['f1'] > 0.3

    mid = data['id']

    method = V01 + "/categorization/{}".format(mid)
    data = app.get_check(method)

    for key in ["method"]:
        assert pars[key] == data[key]

    method = V01 + "/categorization/{}/predict".format(mid)
    if max_results:
        json_data = {'max_results': max_results}
    else:
        json_data = {}
    json_data['subset'] = subset
    json_data['metric'] = metric

    data = app.get_check(method, json=json_data)
    data = data['data']
    response_ref = {'document_id': 'int',
                    'scores': [{'category': 'str',
                                'score': 'float',
                                }
                               ]}

    if solver == 'NearestNeighbor':
        response_ref['scores'][0]['document_id'] = 'int'

    if max_results is None:
        if subset == 'all':
            assert len(data) == len(ds_input['dataset'])
        elif subset == 'train':
            assert len(data) == len(training_set)
        elif subset == 'test':
            assert len(data) == len(ds_input['dataset']) - len(training_set)
        else:
            raise ValueError

    for row in data:
        assert dict2type(row) == response_ref

    if solver == 'NearestNeighbor':
        training_document_id = np.array([row['document_id'] for row in training_set])
        training_document_id_res = np.array([row['scores'][0]['document_id'] for row in data])
        assert_equal(np.in1d(training_document_id_res, training_document_id), True)

    method = V01 + "/metrics/categorization"
    data = app.post_check(method,
            json={'y_true': ds_input['dataset'],
                  'y_pred': data})

    assert dict2type(data) == {'precision': 'float',
                               'recall': 'float',
                               'f1': 'float',
                               'roc_auc': 'float',
                               'average_precision': 'float',
                               'recall_at_20p': 'float'}
    if n_categories == 2:
        assert data['average_precision'] > 0.7
        assert data['roc_auc'] > 0.7
        assert data['recall_at_20p'] > 0.2 # that's a very loose criterion
    else:
        if sklearn_version <= (0, 18, 0) and n_categories_train == 1: 
            pass
        else:
            # this yields wrong results for sklearn 0.17
            assert data['f1'] > 0.32

    method = V01 + "/categorization/{}".format(mid)
    res = app.delete_check(method)


_categoriazation_pars = itertools.product(["LinearSVC", "LogisticRegression",
                                           "NearestNeighbor", 'xgboost'],
                                          ['', 'cv'])

_categoriazation_pars = filter(lambda args: not (args[0].startswith('Nearest') and args[1] == 'cv'),
                               _categoriazation_pars)


@pytest.mark.parametrize("solver, cv", _categoriazation_pars)
def test_api_categorization_2cat(app, solver, cv):
    _api_categorization_wrapper(app, solver, cv, 2)


@pytest.mark.parametrize("n_categories", [1, 2, 3])
def test_api_categorization_2cat_unsupervised(app, n_categories):
    _api_categorization_wrapper(app, 'NearestNeighbor', cv='',
                                n_categories=n_categories,
                                n_categories_train=1)


@pytest.mark.parametrize("solver", ["LogisticRegression", "NearestNeighbor"])
def test_api_categorization_3cat(app, solver):
    _api_categorization_wrapper(app, solver, '', 3)


def test_api_categorization_max_results(app):
    _api_categorization_wrapper(app, 'LogisticRegression', '',
                                2, max_results=100)


@pytest.mark.parametrize("subset", ['all', 'test', 'train'])
def test_api_categorization_3cat(app, subset):
    _api_categorization_wrapper(app, 'LogisticRegression', '', 2, subset=subset)


def test_api_categorization_cosine_positive(app):
    _api_categorization_wrapper(app, 'NearestNeighbor', '', 2, metric='cosine-positive')


def test_api_categorization_subset_document_id(app):
    n_categories = 1
    dsid, lsi_id, _, ds_input = get_features_lsi_cached(app, n_categories=n_categories)
    method = V01 + "/feature-extraction/{}".format(dsid)
    data = app.get_check(method)

    training_set = ds_input['training_set']

    pars = {
          'parent_id': lsi_id,
          'data': training_set,
          'method': 'NearestNeighbor'}

    method = V01 + "/categorization/"
    data = app.post_check(method, json=pars)
    mid = data['id']

    method = V01 + "/categorization/{}/predict".format(mid)

    data = app.get_check(method, json={'batch_id': -1,
                                       'subset_document_id': [222784, 5184,
                                                              929296, 999999999]})
    data = data['data']
    assert len(data) == 3
    # check that only 3 subset documents are returned (9999999 is not a valid id)
    # and that they are in a correct order
    assert_array_equal([row['document_id'] for row in data], [5184, 222784, 929296])

    method = V01 + "/categorization/{}".format(mid)
    app.delete_check(method)


@pytest.mark.parametrize('sort_by', ['comp.graphics', 'rec.sport.baseball'])
def test_api_categorization_sort(app, sort_by):
    n_categories = 2
    dsid, lsi_id, _, ds_input = get_features_lsi_cached(app, n_categories=n_categories)
    method = V01 + "/feature-extraction/{}".format(dsid)
    data = app.get_check(method)

    training_set = ds_input['training_set']

    pars = {
          'parent_id': lsi_id,
          'data': training_set,
          'method': 'NearestNeighbor'}

    method = V01 + "/categorization/"
    data = app.post_check(method, json=pars)
    mid = data['id']

    method = V01 + "/categorization/{}/predict".format(mid)

    data = app.get_check(method, json={'batch_id': -1, "sort_by": sort_by})

    res = []
    for row in data['data']:
        res_el = {'document_id': row['document_id']}
        for scores in row['scores']:
            res_el[scores['category']] = scores['score']
        res.append(res_el)

    df = pd.DataFrame(res)
    df = df.set_index('document_id')

    if sort_by in df.columns:
        mask = pd.notnull(df[sort_by])
        assert_array_equal(df[mask].index.values,
                           df[mask].sort_values(sort_by, ascending=False).index.values)
