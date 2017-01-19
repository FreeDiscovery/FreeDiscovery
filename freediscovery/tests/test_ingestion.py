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
from freediscovery.ingestion import _prepare_data_ingestion
from .run_suite import check_cache

basename = os.path.dirname(__file__)
data_dir = os.path.join(basename, "..", "data", "ds_001", "raw")


fnames_in = ['0.7.47.101442.txt',
             '0.7.47.117435.txt',
             '0.7.6.28635.txt',
             '0.7.6.28636.txt',
             '0.7.6.28637.txt',
             '0.7.6.28638.txt']
fnames_in_abs = [os.path.join(data_dir, el) for el in fnames_in]

def test_ingestion_base_dir():
    data_dir_res, filenames, db = _prepare_data_ingestion(data_dir, None)
    assert data_dir_res == os.path.normpath(data_dir)
    assert_array_equal(db.columns.values, ['file_path', 'internal_id'])
    assert_array_equal(db.file_path.values, fnames_in)


def test_ingestion_metadata():


    metadata = [ {'file_path': fname } for fname in fnames_in_abs]

    data_dir_res, filenames, md_table = _prepare_data_ingestion(None, metadata)

    assert data_dir_res == os.path.normpath(data_dir)
    assert filenames == fnames_in_abs






