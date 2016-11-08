#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
from numpy.testing import assert_allclose
import pytest

from freediscovery.text import FeatureVectorizer
from freediscovery.clustering import Clustering, select_top_words
from .run_suite import check_cache


NCLUSTERS = 2


def fd_setup():
    basename = os.path.dirname(__file__)
    cache_dir = check_cache()
    data_dir = os.path.join(basename, "..", "data", "ds_001", "raw")
    n_features = 110000
    fe = FeatureVectorizer(cache_dir=cache_dir)
    uuid = fe.preprocess(data_dir, file_pattern='.*\d.txt',
                         n_features=n_features, use_hashing=False,
                         stop_words='english')  # TODO unused variable 'uuid' (overwritten on the next line)
    uuid, filenames = fe.transform()
    return cache_dir, uuid, filenames


def check_cluster_consistency(labels, terms):
    assert isinstance(labels, (list, np.ndarray))
    assert isinstance(terms, list)
    assert len(np.unique(labels)) == len(terms)


@pytest.mark.parametrize('method, lsi_components, args, cl_args',
                          [['k_means', None, {}, {}],
                           ['k_means', 20,   {}, {}],
                           ['birch', 20, {'threshold': 0.5}, {}],
                           ['ward_hc', 20, {'n_neighbors': 5}, {}],
                          ])
def test_clustering(method, lsi_components, args, cl_args):
    cache_dir, uuid, filenames = fd_setup()
    np.random.seed(1)
    n_top_words = 9

    cat = Clustering(cache_dir=cache_dir, dsid=uuid)
    cm = getattr(cat, method)
    labels, htree = cm(NCLUSTERS, lsi_components=lsi_components, **args)

    terms = cat.compute_labels(n_top_words=n_top_words, **cl_args)
    mid = cat.mid

    if method == 'ward_hc':
        assert sorted(htree.keys()) == sorted(['n_leaves', 'n_components', 'children'])
    else:
        assert htree == {}

    check_cluster_consistency(labels, terms)
    cat.scores(np.random.randint(0, NCLUSTERS-1, size=len(labels)), labels)
    # load the model saved to disk
    km = cat.load(mid)
    assert_allclose(labels, km.labels_)
    assert len(terms) == NCLUSTERS

    for el in terms:
        assert len(el) == n_top_words
    assert len(np.unique(labels)) == NCLUSTERS
    cluster_indices = np.nonzero(labels == 0)
    if lsi_components is not None:
        # not supported for now otherwise
        terms2 = cat.compute_labels(cluster_indices=cluster_indices, **cl_args)
        # 70% of the terms at least should match
        assert sum([el in terms[0] for el in terms2[0]]) > 0.7*len(terms2[0])

    cat2 = Clustering(cache_dir=cache_dir, mid=mid) # make sure we can load it  # TODO unused variable
    cat.delete()


def test_denrogram_children():
    # temporary solution for https://stackoverflow.com/questions/40239956/node-indexing-in-hierarachical-clustering-dendrograms
    import numpy as np
    from scipy.cluster.hierarchy import dendrogram, linkage
    from freediscovery.clustering import _DendrogramChildren

    # generate two clusters: a with 10 points, b with 5:
    np.random.seed(1)
    a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[10,])
    b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[5,])
    X = np.concatenate((a, b),)
    Z = linkage(X, 'ward')
    # make distances between pairs of children uniform 
    # (re-scales the horizontal (distance) axis when plotting)
    Z[:, 2] = np.arange(Z.shape[0])+1

    ddata = dendrogram(Z, no_plot=True)
    dc = _DendrogramChildren(ddata)
    idx = 0
    # check that we can compute children for all nodes
    for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
        node_children = dc.query(idx)
        idx += 1
    # last level node should encompass all samples
    assert len(node_children) == X.shape[0]
    assert_allclose(sorted(node_children), np.arange(X.shape[0]))


def test_select_top_words():
    words_list = ['apple', 'apples', 'test', 'go']
    n_words = 2
    res = select_top_words(words_list, n=n_words)
    assert len(res) == n_words
    assert res == ['apple', 'test']
