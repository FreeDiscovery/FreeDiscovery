# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pytest
import itertools
from unittest import SkipTest
from numpy.testing import assert_equal, assert_almost_equal

from ...utils import dict2type, sdict_keys
from ...ingestion import DocumentIndex
from ...exceptions import OptionalDependencyMissing
from ...tests.run_suite import check_cache
from .base import parse_res, V01, app, app_notest, get_features, get_features_lsi


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


_categoriazation_pars = itertools.product(  ['data_dir'],
                                            ["LinearSVC", "LogisticRegression",
                                            "NearestCentroid", "NearestNeighbor", 'xgboost'],
                                            ['', 'cv'])

_categoriazation_pars = filter(lambda args: not (args[1].startswith('Nearest') and args[2] == 'cv'),
                               _categoriazation_pars)


def _api_categorization_wrapper(app, metadata_fields, solver, cv, n_categories):

    cv = (cv == 'cv')

    if 'CIRCLECI' in os.environ and cv == 1 and solver in ['LinearSVC', 'xgboost']:
        raise SkipTest # Circle CI is too slow and timesout

    if solver.startswith('Nearest'):
        dsid, lsi_id, _ = get_features_lsi(app, metadata_fields=metadata_fields)
        parent_id = lsi_id
    else:
        dsid, _ = get_features(app, metadata_fields=metadata_fields)
        lsi_id = None
        parent_id = dsid

    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)

    filenames = data['filenames']
    # we train the model on 5 samples / 6 and see what happens
    index_filenames = filenames[:3] + filenames[3:]
    if n_categories == 1:
        y = [1, 1, 1, 1, 1, 1]
    elif n_categories == 2:
        y = [1, 1, 1, 0, 0, 0]
    elif n_categories == 3:
        y = [1, 2, 1, 0, 0, 0]



    method = V01 + "/feature-extraction/{}/id-mapping/flat".format(dsid)
    res = app.post(method, json={'file_path': index_filenames})
    assert res.status_code == 200, method
    index = parse_res(res)['internal_id']

    data_request = [{'internal_id': internal_id, 'category': str(cat_id)} \
                     for (internal_id, cat_id) in zip(index, y)]

    pars = {
          'parent_id': parent_id,
          'data': data_request,
          'method': solver,
          'cv': cv}

    method = V01 + "/categorization/"
    try:
        res = app.post(method, json=pars)
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
    #assert dict2data.keys()) == \
    #        sorted(["index", "y",
    #                "method", "options"])

    for key in ["method"]:
        assert pars[key] == data[key]

    method = V01 + "/categorization/{}/predict".format(mid)
    res = app.get(method)
    data = parse_res(res)
    assert len(data['data']) == len(y)
    response_ref = {'internal_id': 'int',
                     'scores': [ {'category': 'str',
                                  'score': 'float',
                                 }
                               ]}
    if metadata_fields == 'dataset_definition':
        response_ref['document_id'] = 'int'

    if solver == 'NearestNeighbor':
        response_ref['scores'][0]['internal_id'] = 'int'
        if metadata_fields == 'dataset_definition':
            response_ref['scores'][0]['document_id'] = 'int'


    for row in data['data']:
        assert dict2type(row) == response_ref

   #     method = V01 + "/categorization/{}/test".format(mid)
   #     res = app.post(method,
   #             json={'ground_truth_filename':
   #                 os.path.join(data_dir, '..', 'ground_truth_file.txt')})
   #     data = parse_res(res)
   #     assert sorted(data.keys()) == sorted(['precision', 'recall',
   #                         'f1', 'roc_auc', 'average_precision'])

    method = V01 + "/categorization/{}".format(mid)
    res = app.delete(method)
    assert res.status_code == 200

@pytest.mark.parametrize("metadata_fields, solver, cv", _categoriazation_pars)
def test_api_categorization(app, metadata_fields, solver, cv):
    _api_categorization_wrapper(app, metadata_fields, solver, cv, 2)

@pytest.mark.parametrize("metadata_fields", ["data_dir", "dataset_definition", None])
def test_api_categorization_metadata_fields(app, metadata_fields):
    
    if metadata_fields == 'data_dir':
        _api_categorization_wrapper(app, metadata_fields, 'LogisticRegression', '', 2)
    elif metadata_fields == 'dataset_definition':
        _api_categorization_wrapper(app, metadata_fields, 'LogisticRegression', '', 2)
    else:
        with pytest.raises(ValueError):
            _api_categorization_wrapper(app, metadata_fields, 'LogisticRegression', False, 2)
