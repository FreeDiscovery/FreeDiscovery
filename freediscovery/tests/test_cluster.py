#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from freediscovery.text import FeatureVectorizer
from freediscovery.cluster import _ClusteringWrapper, select_top_words
from freediscovery.lsi import _LSIWrapper
from .run_suite import check_cache


NCLUSTERS = 2


def fd_setup():
    basename = os.path.dirname(__file__)
    cache_dir = check_cache()
    np.random.seed(1)
    data_dir = os.path.join(basename, "..", "data", "ds_001", "raw")
    n_features = 110000
    fe = FeatureVectorizer(cache_dir=cache_dir)
    dsid = fe.preprocess(data_dir, file_pattern='.*\d.txt',
                         n_features=n_features, use_hashing=False,
                         stop_words='english',
                         min_df=0.1, max_df=0.9)  # TODO unused variable 'uuid' (overwritten on the next line)
    dsid, filenames = fe.transform()

    lsi = _LSIWrapper(cache_dir=cache_dir, parent_id=dsid)
    lsi.fit_transform(n_components=6)
    return cache_dir, dsid, filenames, lsi.mid


def check_cluster_consistency(labels, terms):
    assert isinstance(labels, (list, np.ndarray))
    assert isinstance(terms, list)
    assert len(np.unique(labels)) == len(terms)


@pytest.mark.parametrize('method, use_lsi, args, cl_args',
                          [['k_means', None, {}, {}],
                           ['k_means', True,   {}, {}],
                           ['birch', True, {'threshold': 0.5}, {}],
                           ['ward_hc', True, {'n_neighbors': 5}, {}],
                           ['dbscan', False, {'eps':0.5, 'min_samples': 2}, {}],
                           ['dbscan', True,   {'eps':0.5, 'min_samples': 2}, {}],
                          ])
def test_clustering(method, use_lsi, args, cl_args):

    cache_dir, uuid, filenames, lsi_id = fd_setup()
    np.random.seed(1)
    n_top_words = 9

    if use_lsi:
        parent_id = lsi_id
    else:
        parent_id = uuid

    cat = _ClusteringWrapper(cache_dir=cache_dir, parent_id=parent_id)
    cm = getattr(cat, method)
    labels, htree = cm(NCLUSTERS, **args)

    terms = cat.compute_labels(n_top_words=n_top_words, **cl_args)
    mid = cat.mid

    if method == 'ward_hc':
        assert sorted(htree.keys()) == sorted(['n_leaves', 'n_components', 'children'])
    else:
        assert htree == {}

    if method == 'dbscan':
        assert (labels != -1).all()

    check_cluster_consistency(labels, terms)
    cat.scores(np.random.randint(0, NCLUSTERS-1, size=len(labels)), labels)
    # load the model saved to disk
    km = cat._load_model()
    assert_allclose(labels, km.labels_)
    if method != 'dbscan':
        # DBSCAN does not take the number of clusters as input
        assert len(terms) == NCLUSTERS
        assert len(np.unique(labels)) == NCLUSTERS

    for el in terms:
        assert len(el) == n_top_words
    cluster_indices = np.nonzero(labels == 0)
    if use_lsi:
        # use_lsi=False is not supported for now
        terms2 = cat.compute_labels(cluster_indices=cluster_indices, **cl_args)
        # 70% of the terms at least should match
        if method != 'dbscan':
            assert sum([el in terms[0] for el in terms2[0]]) > 0.7*len(terms2[0])


    cat2 = _ClusteringWrapper(cache_dir=cache_dir, mid=mid) # make sure we can load it  # TODO unused variable
    cat.delete()


def test_denrogram_children():
    # temporary solution for https://stackoverflow.com/questions/40239956/node-indexing-in-hierarachical-clustering-dendrograms
    import numpy as np
    from scipy.cluster.hierarchy import dendrogram, linkage
    from freediscovery.cluster import _DendrogramChildren

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


def test_dbscan_noisy_utils():
    from freediscovery.cluster.utils import (_dbscan_noisy2unique,
                                             _dbscan_unique2noisy)
    from sklearn.metrics import v_measure_score

    x_ref = np.array([-1, 0, -1,  1, 1, -1,  0])
    y_ref = np.array([2, 0, 3, 1, 1, 4, 0])

    y = _dbscan_noisy2unique(x_ref)
    assert v_measure_score(y, y_ref) == 1

    # check inverse transform
    x = _dbscan_unique2noisy(y_ref)
    assert v_measure_score(x, x_ref) == 1



def test_binary_linkage2clusters():
    from freediscovery.cluster.utils import _binary_linkage2clusters
    from sklearn.metrics import v_measure_score
    n_samples = 10
    linkage = np.array([[1, 2],
                        [2, 3],
                        [5, 7],
                        [6, 9]])

    cluster_id = _binary_linkage2clusters(linkage, n_samples)

    cluster_id_ref = np.array([0, 1, 1, 1, 2, 3, 4, 3, 5, 4])

    assert cluster_id.shape == cluster_id_ref.shape
    assert v_measure_score(cluster_id, cluster_id_ref) == 1.0 # i.e. same clusters


def test_merge_clusters():
    from freediscovery.cluster.utils import _merge_clusters

    X = np.array([[1, 2, 7, 9, 7, 8]]).T

    y = _merge_clusters(X)
    assert_equal(X, y[:,None])

    X_new = np.concatenate((X, X, X, X), axis=1)
    y = _merge_clusters(X_new)
    assert_equal(X, y[:,None])


    X = np.array([[1, 1, 2, 2, 3, 1, 3],
                  [2, 4, 2, 5, 1, 1, 3]]).T
    y = _merge_clusters(X)
    assert_equal(y, [1, 1, 1, 1, 3, 1, 3])

def test_select_top_words():
    words_list = ['apple', 'apples', 'test', 'go']
    n_words = 2
    res = select_top_words(words_list, n=n_words)
    assert len(res) == n_words
    assert res == ['apple', 'test']
