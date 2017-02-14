#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

import sys
import os
import os.path
from unittest import SkipTest

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import scipy.sparse
import itertools
import pytest
from textwrap import dedent


def test_count_duplicates():
    from freediscovery.utils import _count_duplicates

    x = np.array([1, 2, 1, 2, 2, 0])

    y = _count_duplicates(x)
    assert_equal(y, np.array([2, 3, 2, 3, 3, 1]))


def test_docstring_description():
    from freediscovery.utils import _docstring_description
    from freediscovery.datasets import load_dataset

    res = _docstring_description(dedent(load_dataset.__doc__))

    assert len(res.splitlines()) == 21

def test_dictkey2type():
    from freediscovery.utils import dict2type

    assert dict2type('djsk')  == 'str'
    assert dict2type(['t', 1]) == ['str', 'int']
    assert dict2type({'t': {'b': 0.1}}) == {'t': {'b': 'float'}}


def test_check_dict():
    from freediscovery.utils import assert_equal_dict_keys

    a = {'a': 3, 'b': 2}
    b = {'b': 3, 'a': 2}

    assert_equal_dict_keys(a, b)

    a = {'a': 3, 'd': 2}
    b = {'b': 3, 'a': 2}

    with pytest.raises(AssertionError):
        assert_equal_dict_keys(a, b)

    a = {'a': 3, 'b': 2, 'c': {'x': 3}}
    b = {'b': 3, 'a': 2, 'c': {'x': -1}}

    assert_equal_dict_keys(a, b)

    a = {'a': 3, 'b': 2, 'c': {'z': 3}}
    b = {'b': 3, 'a': 2, 'c': {'x': -1}}

    with pytest.raises(AssertionError):
        assert_equal_dict_keys(a, b)

    a = {'a': [{'a': 3}, {'a': 4}]}
    b = {'a': [{'a': 4}, {'b': 9}]}

    # this test passes as the list does not have a dict with the same keys
    assert_equal_dict_keys(a, b)

    a = {'a': [{'a': 3}, {'a': 4}]}
    b = {'a': [{'b': 4}, {'b': 9}]}

    with pytest.raises(AssertionError):
        assert_equal_dict_keys(a, b)
