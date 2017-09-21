# -*- coding: utf-8 -*-

import sys

import pytest
import pandas as pd

from .run_suite import check_cache
from ..utils import dict2type
from unittest import SkipTest

from freediscovery.datasets import load_dataset, IR_DATASETS

cache_dir = check_cache()


@pytest.mark.parametrize('name', ['20_newsgroups_micro'])
def test_load_20newsgoups_dataset(name):
    if (sys.version_info < (3, 0)):
        raise SkipTest
    md, training_set, dataset = load_dataset(name,
                                             cache_dir=cache_dir)

    response_ref = {"document_id": 'int',
                    "file_path": "str",
                    "internal_id": "int"}
    if '20_newsgroups_' in name or 'treclegal09' in name:
        response_ref["category"] = "str"

    assert dict2type(md) == {'data_dir': 'str', 'name': 'str'}

    assert dict2type(dataset[0]) == response_ref
    assert dict2type(training_set[1]) == response_ref

    categories = sorted(list(set([row['category'] for row in dataset])))
    for categories_sel in \
            [[categories[0]],
             [categories[1]],
             [categories[0], categories[1]],
             [categories[1]],
             categories]:

        md, training_set, dataset = load_dataset(name, cache_dir=cache_dir,
                                                 categories=categories_sel)

        for resp in [training_set, dataset]:

            assert dict2type(resp[0]) == response_ref
            result_fields = list(set([el['category'] for el in resp]))

            # the opposite if not always true (e.g. for small training sets)
            for key in result_fields:
                assert key in categories_sel

        training_set = pd.DataFrame(training_set)
        dataset = pd.DataFrame(dataset)
        if name == 'treclegal09_2k_subset':
            if categories_sel == ['positive']:
                assert dataset.shape[0] == 12
            elif categories_sel == categories:
                assert dataset.shape[0] == 2465
                assert (training_set.category == 'positive').sum() == 5
        elif name == '20_newsgroups_micro':
            if categories_sel == ['comp.graphics']:
                assert dataset.shape[0] == 3
            elif categories_sel == categories:
                assert dataset.shape[0] == 7
                assert training_set.shape[0] == 4

# # Uncomment the following section to test the dataset downloading
# #
#
# test_datasets = list(IR_DATASETS.keys())
# test_datasets.remove('legal09int')
#
# @pytest.mark.parametrize('name', test_datasets)
# def test_dataset_downloads(name):
#     load_dataset(name, cache_dir=cache_dir)
