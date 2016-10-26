#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals
import os.path

import numpy as np
from numpy.testing import assert_allclose, assert_equal
from freediscovery.io import parse_ground_truth_file


def test_parse_ground_truth_file():
    basename = os.path.dirname(__file__)

    filename = os.path.join(basename, "..","data", "ds_001", "ground_truth_file.txt")
    res = parse_ground_truth_file(filename)
    assert_allclose(res.values[:,0] , [1, 1, 1, 0, 0, 0])
