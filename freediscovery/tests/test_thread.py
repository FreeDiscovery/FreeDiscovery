#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

import os.path

from freediscovery.parsers import EmailParser
from freediscovery.email_threading import _EmailThreadingWrapper
from .run_suite import check_cache
from itertools import groupby


basename = os.path.dirname(__file__)
data_dir = os.path.join(basename, "..", "data", "fedora-devel-list-2008-October")


def test_threading():
    cache_dir = check_cache()

    fe = EmailParser(cache_dir=cache_dir)
    uuid = fe.transform(data_dir, file_pattern='.*\d')

    filenames, res = fe.load(uuid)


    cat = _EmailThreadingWrapper(cache_dir=cache_dir, parent_id=uuid)

    tree = cat.thread()
    cat.get_params()

    tree_ref = [ {'id': 0, 'parent': None, 'children': [
                  {'id': 1, 'children': [], 'parent': 0},
                  {'id': 2, 'parent': 0,  'children': [
                         {'id': 3, 'children': [], 'parent': 2},
                         {'id': 4, 'children': [], 'parent': 2}],
                         }]
                  }]

    assert [el.to_dict() for el in tree] == tree_ref


    assert len(filenames) == sum([el.size for el in tree])
    assert len(filenames) == 5

    #res2 = {idx: [el0[0] for el0 in el] for idx, el in \
    #    groupby(enumerate(tree), key=lambda x: x[1])}


  
    #for key, idx in res2.items():
    #    print('Thread id: {}'.format(key))
    #    print('Relevant documents: {}'.format(idx))
    #    print('Parent documents  : {}'.format([parent[el] for el in idx]))
