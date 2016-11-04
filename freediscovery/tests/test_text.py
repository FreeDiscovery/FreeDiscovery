#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
from numpy.testing import assert_equal
import scipy.sparse
import itertools
import pytest

from freediscovery.text import FeatureVectorizer
from .run_suite import check_cache

basename = os.path.dirname(__file__)
data_dir = os.path.join(basename, "..", "data", "ds_001", "raw")
n_features = 1100000


fe_cases = list(itertools.product(['word', 'char'], ['None', 'english'], [[1,1], [1, 2], [4,4]],
                [True, False], [True, False], [True, False], [True, False]))
fe_names = 'analyzer, stop_words, ngram_range, use_idf, sublinear_tf, binary, use_hashing'


def filter_fe_cases(x):
    key_list = fe_names.split(', ')
    res = dict(zip(key_list, x))
    if res['analyzer'] != 'word' and res['stop_words'] != "None":
        return 0
    if not res['use_idf'] and not res['sublinear_tf']:
        return 0 # this is not used anyway
    return 1


@pytest.mark.parametrize(fe_names, filter(filter_fe_cases, fe_cases))
def test_feature_extraction(analyzer, stop_words, ngram_range, use_idf, sublinear_tf, binary, use_hashing):
    cache_dir = check_cache()

    fe = FeatureVectorizer(cache_dir=cache_dir)
    uuid = fe.preprocess(data_dir, file_pattern='.*\d.txt', n_features=n_features,
            analyzer=analyzer, stop_words=stop_words, ngram_range=ngram_range,
            use_idf=use_idf, binary=binary, use_hashing=use_hashing, sublinear_tf=sublinear_tf)  # TODO unused (overwritten on the next line)
    uuid, filenames = fe.transform()

    filenames2, res2 = fe.load(uuid)
    assert_equal(filenames2, filenames)
    assert isinstance(res2,  np.ndarray) or scipy.sparse.issparse(res2), "not an array {}".format(res2)

    fe.search(['0.7.47.117435.txt'])
    fe.search(['DOES_NOT_EXIST.txt'])
    fe.list_datasets
    assert np.isfinite(res2.data).all()

    if not use_hashing:
        n_top_words = 5
        terms = fe.query_features([2, 3, 5], n_top_words=n_top_words)
        assert len(terms) == n_top_words

    fe.delete()
