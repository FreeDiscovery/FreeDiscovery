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

from .. import fd_app
from ...utils import _silent, dict2type, sdict_keys
from ...ingestion import DocumentIndex
from ...exceptions import OptionalDependencyMissing
from ...tests.run_suite import check_cache

from .base import parse_res, V01, app, app_notest, get_features, get_features_lsi


def test_example_dataset(app):
    data = app.get_check(V01 + '/example-dataset/20newsgroups_micro')
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

