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



def test_ingestion_base_dir():
    data_dir_res, filenames, md = _prepare_data_ingestion(data_dir, None, None, None)
    assert data_dir_res == os.path.normpath(data_dir)
    print(filenames)

def test_ingestion_metadata():
    fnames_in = ['0.7.47.101442.txt',
          '0.7.47.117435.txt',
          '0.7.6.28635.txt',
          '0.7.6.28636.txt',
          '0.7.6.28637.txt',
          '0.7.6.28638.txt']

    fnames_in = [os.path.join(data_dir, el) for el in fnames_in]

    metadata = [ {'file_path': fname } for fname in fnames_in]

    data_dir_res, filenames, md = _prepare_data_ingestion(data_dir, None, None, None)






