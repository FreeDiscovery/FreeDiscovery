#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from .run_suite import check_cache
from ..utils import dict2type

from freediscovery.datasets import load_dataset
from unittest import SkipTest
import json

cache_dir = check_cache()

@pytest.mark.parametrize('name', ['20newsgroups_micro'])
def test_load_20newsgoups_dataset(name):
    md, training_set, test_set = load_dataset(name, force=True, cache_dir=cache_dir,
                                training_set_fields=[], test_set_fields=['document_id'])

    assert dict2type(md) == {'data_dir': 'str', 'name': 'str'}

    assert dict2type(test_set[0]) == { "document_id": 'int',
                                       "category": "str" }
    assert training_set is None

    categories = sorted(list(set([row['category'] for row in test_set])))
    for training_set_fields, test_set_fields, categories_sel in \
            [(['document_id'], ['document_id'], [categories[0]]),
             (['file_path'],  ['document_id'], [categories[0]]),
             (['document_id', 'file_path'],  ['document_id'], [categories[0], categories[1]]),
             (['document_id', 'file_path'],  ['document_id', 'internal_id'], [categories[1]]),
             (['file_path'],  ['document_id'], categories)]:

        md, training_set, test_set = load_dataset(name, cache_dir=cache_dir,
                                    training_set_fields=training_set_fields,
                                    test_set_fields=test_set_fields,
                                    categories=categories_sel)

        for resp, set_fields in [(training_set, training_set_fields),
                                (test_set, test_set_fields)]:
            response_ref = { "category": "str" }
            if 'document_id' in set_fields:
                response_ref['document_id'] = 'int'
            if 'file_path' in set_fields:
                response_ref['file_path'] = 'str'
            if 'internal_id' in set_fields:
                response_ref['internal_id']= 'int'



            assert dict2type(resp[0]) ==  response_ref
            result_fields = list(set([el['category'] for el in resp]))

            # the opposite if not always true (e.g. for small training sets)
            for key in result_fields:
                assert key in categories_sel





