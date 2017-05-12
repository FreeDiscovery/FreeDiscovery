# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os.path
import warnings

from numpy.testing import (assert_allclose, assert_equal,
                           assert_array_less)

import pytest
from freediscovery.metrics import categorization_score
from freediscovery.pipeline import _split_path, PipelineFinder
from .run_suite import check_cache


def test_split_path():
    assert _split_path('abc/test4') == ['abc', 'test4']
    assert _split_path('abc/test4/') == ['abc', 'test4']
    if os.name == 'nt':  # on windows
        assert _split_path('C:\\abc\\test4') == ["C:\\", 'abc', 'test4']
        assert _split_path('C:\\abc\\test4\\') == ["C:\\", 'abc', 'test4']
        assert _split_path('/test/test4') == ['\\', 'test', 'test4']
    else:
        assert _split_path('/test/test4') == ['/', 'test', 'test4']
        # this raises an error on windows for some reason
        assert _split_path('//abc/test4/') == ["//", 'abc', 'test4']


def test_normalize_cachedir():
    _normalize_cachedir = PipelineFinder._normalize_cachedir

    assert str(_normalize_cachedir('/tmp/')) == os.path.normpath('/tmp/ediscovery_cache')
    assert str(_normalize_cachedir('/tmp/ediscovery_cache')) == os.path.normpath('/tmp/ediscovery_cache')


@pytest.yield_fixture(autouse=True, scope='session')
def test_suite_cleanup():
    import shutil
    import traceback
    # setup
    yield
    # teardown - put your command here
    cache_dir = check_cache()
    try:
        shutil.rmtree(cache_dir)
        print('\nSucessfully removed cache_dir={}'.format(cache_dir))
    except Exeption as e:
        warnings.warn('Failed to remove cache_dir={}, \n{}'.format(
	                       cache_dir, traceback.print_tb(e.__traceback__)))
