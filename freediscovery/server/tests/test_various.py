# -*- coding: utf-8 -*-

from ...utils import dict2type

from .base import parse_res, V01, app, app_notest, get_features, get_features_lsi


def test_example_dataset(app):
    data = app.get_check(V01 + '/example-dataset/20_newsgroups_micro')
    assert dict2type(data, max_depth=1) == {'dataset': 'list',
                                            'training_set': 'list',
                                            'metadata': 'dict'}
    assert dict2type(data['metadata']) == {'data_dir': 'str',
                                           'name': 'str'}
    assert dict2type(data['training_set'][0]) == {'category': 'str',
                                               'document_id': 'int',
                                               'internal_id': 'int',
                                               'file_path': 'str'}
    assert dict2type(data['dataset'][0]) == {'category': 'str',
                                           'document_id': 'int',
                                           'internal_id': 'int',
                                           'file_path': 'str'
                                          }



def test_openapi_specs(app):
    data = app.get_check('/openapi-specs.json')
    assert data['swagger'] == '2.0'


def test_version(app):
    data = app.get_check('/')
    assert dict2type(data) == {'version': {'number': 'str'},
                               'env': {'python_version': 'str'},
                               'config': 'NoneType'}
