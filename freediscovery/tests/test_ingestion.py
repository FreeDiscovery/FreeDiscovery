# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path
import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import itertools
import pytest

from freediscovery.text import (FeatureVectorizer,
                                _FeatureVectorizerSampled)
from freediscovery.ingestion import DocumentIndex
from .run_suite import check_cache

basename = os.path.dirname(__file__)
data_dir = os.path.join(basename, "..", "data", "ds_001", "raw")
cache_dir = check_cache()


fnames_in = ['0.7.47.101442.txt',
             '0.7.47.117435.txt',
             '0.7.6.28635.txt',
             '0.7.6.28636.txt',
             '0.7.6.28637.txt',
             '0.7.6.28638.txt']
fnames_in_abs = [os.path.join(data_dir, el) for el in fnames_in]

def test_ingestion_base_dir():
    dbi = DocumentIndex.from_folder(data_dir)
    data_dir_res, filenames, db = dbi.data_dir, dbi.filenames, dbi.data
    assert data_dir_res == os.path.normpath(data_dir)
    assert_array_equal(db.columns.values, ['file_path', 'internal_id'])
    assert_array_equal(db.file_path.values, fnames_in)
    assert_array_equal([os.path.normpath(el) for el in  filenames],
                       [os.path.join(data_dir_res, el) for el in db.file_path.values])


def test_ingestion_pickling():
    from sklearn.externals import joblib
    db = DocumentIndex.from_folder(data_dir)
    fname = os.path.join(cache_dir, 'document_index')
    # check that db is picklable
    joblib.dump(db, fname)
    db2 = joblib.load(fname)
    os.remove(fname)


@pytest.mark.parametrize('n_fields', [1, 2, 3])
def test_ingestion_metadata(n_fields):
    metadata = []
    for idx, fname in enumerate(fnames_in_abs):
        el = {'file_path': fname }
        if n_fields >= 2:
            el['document_id'] = 'a' + str(idx + 100)
        if n_fields >= 3:
            el['rendition_id'] = 1
        metadata.append(el)

    dbi = DocumentIndex.from_list(metadata)
    data_dir_res, filenames, db = dbi.data_dir, dbi.filenames, dbi.data

    assert data_dir_res == os.path.normpath(data_dir)
    assert filenames == fnames_in_abs
    if n_fields == 1:
        columns_ref = sorted(['file_path', 'internal_id'])
    elif n_fields == 2:
        columns_ref = sorted(['file_path', 'document_id', 'internal_id'])
    elif n_fields == 3:
        columns_ref = sorted(['file_path', 'document_id', 'rendition_id', 'internal_id'])

    assert_array_equal(sorted(db.columns.values), columns_ref)
    assert_array_equal([os.path.normpath(el) for el in  filenames],
                       [os.path.join(data_dir_res, el) for el in db.file_path.values])
