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

from freediscovery.datasets import load_dataset
from unittest import SkipTest


def test_load_dataset():
    try:
        from unittest.mock import patch, MagicMock
    except ImportError:
        raise SkipTest


    cache_dir = check_cache()
    m = MagicMock()
    m2 = MagicMock()
    with patch.dict("sys.modules", requests=m, tarfile=m2):
        res = load_dataset(verbose=False, force=True, cache_dir=cache_dir,
                       load_ground_truth=False, verify_checksum=False)
    assert sorted(res.keys()) == sorted([#"ground_truth_file", "seed_non_relevant_files",
                                         #"seed_relevant_files",
                                         "base_dir", "data_dir"])

