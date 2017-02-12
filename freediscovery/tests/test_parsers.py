#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
from numpy.testing import assert_equal
import scipy.sparse
import itertools
import pytest

from freediscovery.parsers import EmailParser
from freediscovery.externals.jwzthreading import Message
from .run_suite import check_cache

basename = os.path.dirname(__file__)
data_dir = os.path.join(basename, "..", "data", "fedora-devel-list-2008-October")


def test_email_parser():
    cache_dir = check_cache()

    fe = EmailParser(cache_dir=cache_dir)
    uuid = fe.transform(data_dir, file_pattern='.*\d')

    filenames, res = fe.load(uuid)
    assert_equal(filenames, fe._pars['filenames'])
    assert len(filenames) == len(res)
    assert len(filenames) == 5

    for message in res:
        assert isinstance(message, Message)
    fe.delete()

def test_search_filenames():
    cache_dir = check_cache()

    fe = EmailParser(cache_dir=cache_dir)
    fe.transform(data_dir, file_pattern='.*\d')

    filenames = fe._pars['filenames']

    for low, high, step in [(0, 1, 1),
                            (0, 4, 1),
                            (3, 1, -1)]:
        idx_slice = list(range(low, high, step))
        filenames_slice = [filenames[idx] for idx in idx_slice]
        idx0 = fe.search(filenames_slice)
        assert_equal(idx0, idx_slice)
        assert_equal(filenames_slice, fe[idx0])

    with pytest.raises(KeyError):
        fe.search(['DOES_NOT_EXIST.txt'])

    fe.list_datasets()
