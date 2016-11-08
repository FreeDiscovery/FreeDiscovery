#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
from numpy.testing import assert_allclose

from freediscovery.io import parse_ground_truth_file


def test_parse_ground_truth_file():
    basename = os.path.dirname(__file__)
    filename = os.path.join(basename, "..","data", "ds_001", "ground_truth_file.txt")
    res = parse_ground_truth_file(filename)
    assert_allclose(res.values[:,0] , [1, 1, 1, 0, 0, 0])
