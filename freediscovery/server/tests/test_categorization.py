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
                   get_features_lsi_cached, get_features_cached)



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
    res = app.post(method, json=lsi_pars)
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

    assert sorted(data.keys()) == \
            sorted(["n_components", "parent_id"])
    
    for key in data.keys():
        assert data[key] == lsi_pars[key]




def _api_categorization_wrapper(app, solver, cv, n_categories, n_categories_train=None):

    cv = (cv == 'cv')

    if 'CIRCLECI' in os.environ and cv == 1 and solver in ['LinearSVC', 'xgboost']:
        raise SkipTest # Circle CI is too slow and timesout

    if solver.startswith('Nearest'):
        dsid, lsi_id, _, ds_input = get_features_lsi_cached(app, n_categories=n_categories)
        parent_id = lsi_id
    else:
        dsid, _, ds_input = get_features_cached(app, n_categories=n_categories)
        lsi_id = None
        parent_id = dsid

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)

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
          'cv': cv}

    method = V01 + "/categorization/"
    try:
        res = app.post(method, json=pars)
    except OptionalDependencyMissing:
        raise SkipTest

    data = parse_res(res)
    assert res.status_code == 200, method
    assert dict2type(data) == {'id' : 'str',
                               'training_scores': {'recall': 'float',
                                                    'f1': 'float',
                                                    'precision': 'float',
                                                    'roc_auc': 'float',
                                                    'average_precision': 'float'}}

    print(data)
    if n_categories_train == 1:
        pass
    elif n_categories == 2:
        assert data['training_scores']['average_precision'] > 0.73
        if solver == 'NearestNeighbor':
            assert data['training_scores']['roc_auc'] > 0.7
        else:
            assert data['training_scores']['roc_auc'] >= 0.5
    elif n_categories == 3 and solver == 'NearestNeighbor':
        assert data['training_scores']['f1'] > 0.6
    else:
        assert data['training_scores']['f1'] > 0.3

    mid = data['id']

    method = V01 + "/categorization/{}".format(mid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)

    for key in ["method"]:
        assert pars[key] == data[key]

    method = V01 + "/categorization/{}/predict".format(mid)
    res = app.get(method)
    data = parse_res(res)
    data = data['data']
    response_ref = {'internal_id': 'int',
                    'document_id': 'int',
                     'scores': [ {'category': 'str',
                                  'score': 'float',
                                 }
                               ]}

    if solver == 'NearestNeighbor':
        response_ref['scores'][0]['internal_id'] = 'int'
        response_ref['scores'][0]['document_id'] = 'int'


    for row in data:
        assert dict2type(row) == response_ref


    method = V01 + "/metrics/categorization"
    res = app.post(method,
            json={'y_true': ds_input['dataset'],
                  'y_pred': data})
    data = parse_res(res)


    assert dict2type(data) == {'precision': 'float',
                               'recall': 'float',
                               'f1': 'float',
                               'roc_auc': 'float',
                               'average_precision': 'float'}
    if n_categories == 2:
        assert data['average_precision'] > 0.7
        assert data['roc_auc'] > 0.7
    else:
        pass

    method = V01 + "/categorization/{}".format(mid)
    res = app.delete(method)
    assert res.status_code == 200


_categoriazation_pars = itertools.product( ["LinearSVC", "LogisticRegression",
                                            #"NearestCentroid",
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
                                n_categories=n_categories, n_categories_train=1)

@pytest.mark.parametrize("solver", ["LogisticRegression", "NearestNeighbor"])
def test_api_categorization_3cat(app, solver):
    _api_categorization_wrapper(app, solver, '', 3)
