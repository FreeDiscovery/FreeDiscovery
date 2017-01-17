# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os.path
from unittest import SkipTest

import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
                           assert_array_less)

import pytest
from freediscovery.utils import categorization_score
from freediscovery.pipeline import _split_path, PipelineFinder
from .run_suite import check_cache


def test_split_path():
    assert _split_path('abc/test4') ==  ['abc', 'test4']
    assert _split_path('abc/test4/') ==  ['abc', 'test4']
    if os.name == 'nt': # on windows
        assert _split_path('C:\\abc\\test4') ==  ["C:\\", 'abc', 'test4']
        assert _split_path('C:\\abc\\test4\\') ==  ["C:\\", 'abc', 'test4']
        assert _split_path('/test/test4') ==  ['\\', 'test', 'test4']
    else:
        assert _split_path('/test/test4') ==  ['/', 'test', 'test4']
        # this raises an error on windows for some reason
        assert _split_path('//abc/test4/') ==  ["//", 'abc', 'test4']


def test_normalize_cachedir():
    _normalize_cachedir = PipelineFinder._normalize_cachedir

    assert _normalize_cachedir('/tmp/') == os.path.normpath('/tmp/ediscovery_cache')
    assert _normalize_cachedir('/tmp/ediscovery_cache') == os.path.normpath('/tmp/ediscovery_cache')

