#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

import os.path
from unittest import SkipTest

import numpy as np
from numpy.testing import assert_allclose, assert_equal

import pytest
import itertools

from freediscovery.parsers import EmailParser
from freediscovery.threading import (EmailThreading)
from freediscovery.exceptions import OptionalDependencyMissing
from .run_suite import check_cache


basename = os.path.dirname(__file__)
data_dir = os.path.join(basename, "..", "data", "fedora-devel-list-2008-October")


def test_threading():
    cache_dir = check_cache()

    fe = EmailParser(cache_dir=cache_dir)
    uuid = fe.transform(data_dir, file_pattern='.*\d')

    filenames, res = fe.load(uuid)


    cat = EmailThreading(cache_dir=cache_dir, dsid=uuid)
    res = cat.thread()

    cat.get_params()
    assert len(filenames) == sum([el.size for el in res])
    assert len(filenames) == 5
    assert len(filenames) == len([el for x in res for el in x.flatten()])
