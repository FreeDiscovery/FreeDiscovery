# -*- coding: utf-8 -*-

import os

import numpy as np
from unittest import SkipTest
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
import pytest

from sklearn.preprocessing import normalize
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_blobs

from freediscovery.cluster import select_top_words
from freediscovery.cluster.hierarchy import _check_birch_tree_consistency
from freediscovery.cluster import compute_optimal_sampling, centroid_similarity
from freediscovery.cluster import Birch, birch_hierarchy_wrapper



NCLUSTERS = 2

@pytest.mark.parametrize('dataset, optimal_sampling',
                         [('random', False),
                          ('birch_hierarchical', False),
                          ('birch_hierarchical', True)])
def test_birch_make_hierarchy(dataset, optimal_sampling):

    if dataset == 'random':
        np.random.seed(9999)

        X = np.random.rand(1000, 100)
        normalize(X)
        branching_factor = 10
    elif dataset == 'birch_hierarchical':
        basename = os.path.dirname(__file__)
        X = np.load(os.path.join(basename, '..', 'data',
                    'ds_lsi_birch', 'data.npy'))
        branching_factor = 2

    mod = Birch(n_clusters=None, threshold=0.1,
                branching_factor=branching_factor, compute_labels=False,
                compute_sample_indices=True)
    mod.fit(X)

    htree, n_subclusters = birch_hierarchy_wrapper(mod)

    # let's compute cluster similarity
    for row in htree.flatten():
        inertia, S_sim = centroid_similarity(X,
                                             row['document_id_accumulated'])
        row['document_similarity'] = S_sim
        row['cluster_similarity'] = inertia

    assert htree.tree_size == n_subclusters

    doc_count = 0
    for el in htree.flatten():
        doc_count += len(el['document_id'])
        el.current_depth
        el.document_id_accumulated
    assert doc_count == len(htree['document_id_accumulated'])
    assert doc_count == X.shape[0]
    assert htree.document_count == X.shape[0]
    if optimal_sampling:
        s_samples_1 = compute_optimal_sampling(htree, min_similarity=0.85,
                                               min_coverage=0.9)

        for row in s_samples_1:
            assert len(row['document_similarity']) == 1
            assert len(row['document_id_accumulated']) == 1
        s_samples_2 = compute_optimal_sampling(htree, min_similarity=0.85,
                                               min_coverage=0.2)
        s_samples_3 = compute_optimal_sampling(htree, min_similarity=0.9,
                                               min_coverage=0.9)

        assert len(s_samples_1) > len(s_samples_2)
        assert len(s_samples_1) < len(s_samples_3)


def test_birch_hierarchy_fitted():
    model = Birch()

    with pytest.raises(NotFittedError):
        birch_hierarchy_wrapper(model)


def test_birch_hierarchy_validation():
    with pytest.raises(ValueError):
        birch_hierarchy_wrapper("some other object")


@pytest.mark.parametrize('example_id', [12, 34])
def test_birch_example_reproducibility(example_id):
    # check reproducibility of the Birch example
    rng = np.random.RandomState(42)

    X, y = make_blobs(n_samples=1000, n_features=10, random_state=rng)

    cluster_model = Birch(threshold=0.9, branching_factor=20,
                          compute_sample_indices=True)
    cluster_model.fit(X)
    # assert len(cluster_model.root_.subclusters_[1].child_.subclusters_) == 3

    htree, n_subclusters = birch_hierarchy_wrapper(cluster_model)

    assert htree.tree_size == n_subclusters

    # same random seed as in the birch hierarchy example
    assert htree.tree_size == 78
    sc = htree.flatten()[example_id]
    if example_id == 34:
        # this is true in both cases, but example_id fails on circle ci
        assert sc.current_depth == 1
        assert len(sc.children) == 3

    assert_array_equal([sc['cluster_id'] for sc in htree.flatten()],
                       np.arange(htree.tree_size))


def test_denrogram_children():
    # temporary solution for
    # https://stackoverflow.com/questions/40239956/node-indexing-in-hierarachical-clustering-dendrograms
    import numpy as np
    from scipy.cluster.hierarchy import dendrogram, linkage
    from freediscovery.cluster import _DendrogramChildren

    # generate two clusters: a with 10 points, b with 5:
    np.random.seed(1)
    a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]],
                                      size=[10, ])
    b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]],
                                      size=[5, ])
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
    # i.e. same clusters
    assert v_measure_score(cluster_id, cluster_id_ref) == 1.0


def test_merge_clusters():
    from freediscovery.cluster.utils import _merge_clusters

    X = np.array([[1, 2, 7, 9, 7, 8]]).T

    y = _merge_clusters(X)
    assert_equal(X, y[:, None])

    X_new = np.concatenate((X, X, X, X), axis=1)
    y = _merge_clusters(X_new)
    assert_equal(X, y[:, None])

    X = np.array([[1, 1, 2, 2, 3, 1, 3],
                  [2, 4, 2, 5, 1, 1, 3]]).T
    y = _merge_clusters(X)
    assert_equal(y, [1, 1, 1, 1, 3, 1, 3])


def test_select_top_words():
    try:
        import nltk
    except ImportError:
        raise SkipTest
    words_list = ['apple', 'apples', 'test', 'go']
    n_words = 2
    res = select_top_words(words_list, n=n_words)
    assert len(res) == n_words
    assert res == ['apple', 'test']


def test_birch_clusterig_single_nodes():

    basename = os.path.dirname(__file__)
    X = np.load(os.path.join(basename, '..', 'data',
                'ds_lsi_birch', 'data.npy'))
    branching_factor = 5

    mod = Birch(n_clusters=None, threshold=0.1,
                branching_factor=branching_factor, compute_labels=False,
                compute_sample_indices=True)
    mod.fit(X)

    htree, n_subclusters = birch_hierarchy_wrapper(mod)

    # let's compute cluster similarity
    for row in htree.flatten():
        inertia, S_sim = centroid_similarity(X,
                                             row['document_id_accumulated'])
        row['document_similarity'] = S_sim
        row['cluster_similarity'] = inertia

    assert htree.tree_size == n_subclusters

    doc_count = 0
    for el in htree.flatten():
        doc_count += len(el['document_id'])
        el.current_depth
        el.document_id_accumulated
    assert doc_count == len(htree['document_id_accumulated'])
    assert doc_count == X.shape[0]
    assert htree.document_count == X.shape[0]

    # make sure that we have no clusters with a single child
    assert sum(len(el.children) == 1 for el in htree.flatten()) == 0
