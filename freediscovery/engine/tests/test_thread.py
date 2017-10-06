# -*- coding: utf-8 -*-

import os.path

from freediscovery.engine.vectorizer import FeatureVectorizer
from freediscovery.engine.email_threading import _EmailThreadingWrapper
from freediscovery.tests.run_suite import check_cache


basename = os.path.dirname(__file__)
data_dir = os.path.join(basename, "..", "..", "data",
                        "fedora-devel-list-2008-October")


def test_threading():
    cache_dir = check_cache()

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid = fe.setup()
    fe.ingest(data_dir=data_dir)
    fe.parse_email_headers()

    cat = _EmailThreadingWrapper(cache_dir=cache_dir, parent_id=uuid)

    tree = cat.thread()
    cat.get_params()

    tree_ref = [{'id': 0, 'parent': None, 'children': [
                {'id': 1, 'children': [], 'parent': 0},
                {'id': 2, 'parent': 0,  'children': [
                         {'id': 3, 'children': [], 'parent': 2},
                         {'id': 4, 'children': [], 'parent': 2}]}]}]

    assert [el.to_dict() for el in tree] == tree_ref

    assert len(fe.filenames_) == sum([el.tree_size for el in tree])
    assert len(fe.filenames_) == 5
    assert len(tree[0].flatten()) == 5

    # res2 = {idx: [el0[0] for el0 in el] for idx, el in \
    #    groupby(enumerate(tree), key=lambda x: x[1])}

    # for key, idx in res2.items():
    #    print('Thread id: {}'.format(key))
    #    print('Relevant documents: {}'.format(idx))
    #    print('Parent documents  : {}'.format([parent[el] for el in idx]))
