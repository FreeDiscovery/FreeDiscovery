# -*- coding: utf-8 -*-

import os

import numpy as np
from numpy.testing import assert_allclose

from freediscovery.io import parse_ground_truth_file
from freediscovery.metrics import categorization_score
from freediscovery.tests.run_suite import check_cache


basename = os.path.dirname(__file__)

cache_dir = check_cache()

data_dir = os.path.join(basename, "..", "data", "ds_001", "raw")

ground_truth = parse_ground_truth_file(
                  os.path.join(data_dir, "..", "ground_truth_file.txt"))


def test_unique_label():
    """Check that testing works with only one label in the training test"""
    np.random.seed(10)
    Nshape = ground_truth.file_path.values.shape
    is_relevant = np.zeros(Nshape).astype(int)

    idx = np.arange(len(is_relevant), dtype='int')

    categorization_score(idx, is_relevant, idx, np.random.rand(*Nshape))


def test_categorization_score():
    idx = [1, 2,  3,  4,  5, 6]
    y = [1, 1, -1, -1, -1, 1]
    idx_ref = [10, 5, 3, 2, 6]
    y_ref = [0,  1, 0, 1, 1]

    scores = categorization_score(idx_ref, y_ref, idx, y)

    assert_allclose(scores['precision'], 1.0)
    assert_allclose(scores['recall'], 0.66666666, rtol=1e-4)

    # make sure permutations don't affect the result
    idx_ref2 = [10, 5, 2, 3, 6]
    y_ref2 = [0, 1, 1, 0, 1]
    scores2 = categorization_score(idx_ref2, y_ref2, idx, y)
    assert scores['average_precision'] == scores2['average_precision']
