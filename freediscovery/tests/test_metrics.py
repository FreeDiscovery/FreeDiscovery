#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from freediscovery.text import FeatureVectorizer
from .run_suite import check_cache

from freediscovery.metrics import (ratio_duplicates_score,
                                   f1_same_duplicates_score,
                                   mean_duplicates_count_score)

def test_duplicate_metrics():
    x1 = np.array([0, 1, 1, 2, 3, 2])
    x2 = np.array([2, 3, 3, 4, 5, 4])  # just an ids renaming
    x3 = np.array([1, 2, 3, 4, 5, 6])  # no duplicates
    x4 = np.array([0, 1, 3, 2, 5, 2])  # half of original duplicates

    assert ratio_duplicates_score(x1, x2) == 1.0
    assert ratio_duplicates_score(x1, x3) == 0.0
    assert ratio_duplicates_score(x1, x4) == 0.5

    assert f1_same_duplicates_score(x1, x2) == 1.0
    assert f1_same_duplicates_score(x1, x3) == 0.0
    assert 0.25 < f1_same_duplicates_score(x1, x4) < 0.75  # more loose condition

    assert mean_duplicates_count_score(x1, x2) == 1.0
    assert mean_duplicates_count_score(x1, x3) == 0.5
    assert mean_duplicates_count_score(x1, x4) == 0.75


