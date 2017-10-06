# -*- coding: utf-8 -*-

import os.path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from freediscovery.engine.vectorizer import FeatureVectorizer
from freediscovery.engine.cluster import _ClusteringWrapper
from freediscovery.engine.lsi import _LSIWrapper
from freediscovery.tests.run_suite import check_cache


NCLUSTERS = 2


def fd_setup():
    basename = os.path.dirname(__file__)
    cache_dir = check_cache()
    np.random.seed(1)
    data_dir = os.path.join(basename, "..", "..",
                            "data", "ds_001", "raw")
    n_features = 110000
    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    dsid = fe.setup(n_features=n_features, use_hashing=False,
                    stop_words='english',
                    min_df=0.1, max_df=0.9)
    fe.ingest(data_dir, file_pattern='.*\d.txt')

    lsi = _LSIWrapper(cache_dir=cache_dir, parent_id=dsid, mode='w')
    lsi.fit_transform(n_components=6)
    return cache_dir, dsid, fe.filenames_, lsi.mid


def check_cluster_consistency(labels, terms):
    assert isinstance(labels, (list, np.ndarray))
    assert isinstance(terms, list)
    assert len(np.unique(labels)) == len(terms)


@pytest.mark.parametrize('method, use_lsi, args, cl_args',
                         [['k_means', None, {}, {}],
                          ['k_means', True,   {}, {}],
                          ['birch', True, {'threshold': 0.5}, {}],
                          ['birch', True, {'threshold': 0.5, 'branching_factor': 3, 'n_clusters': None}, {}],
                          #['ward_hc', True, {'n_neighbors': 5}, {}],
                          ['dbscan', False, {'eps': 0.5, 'min_samples': 2}, {}],
                          ['dbscan', True,   {'eps': 0.5, 'min_samples': 2}, {}]])
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
    if 'n_clusters' not in args:
        args['n_clusters'] = NCLUSTERS
    labels = cm(**args)

    htree = cat._load_htree()

    mid = cat.mid

    if method == 'birch' and cat._pars['is_hierarchical']:
        assert htree != {}
        flat_tree = htree.flatten()

        terms = cat.compute_labels(n_top_words=n_top_words,
                                   cluster_indices=[row['document_id_accumulated']
                                                    for row in flat_tree])
        for label, row in zip(terms, flat_tree):
            row['cluster_label'] = label
    else:
        terms = cat.compute_labels(n_top_words=n_top_words, **cl_args)

        if method == 'ward_hc':
            assert sorted(htree.keys()) == sorted(['n_leaves',
                                                   'n_components', 'children'])
        else:
            assert not htree

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
            terms2 = cat.compute_labels(cluster_indices=[cluster_indices],
                                        **cl_args)
            # 70% of the terms at least should match
            if method != 'dbscan':
                assert sum([el in terms[0]
                            for el in terms2[0]]) > 0.7*len(terms2[0])

    # make sure we can load it
    cat2 = _ClusteringWrapper(cache_dir=cache_dir, mid=mid)
    cat.delete()
