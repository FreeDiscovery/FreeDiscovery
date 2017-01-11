#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
from numpy.testing import assert_allclose

from freediscovery.text import FeatureVectorizer
from freediscovery.lsi import LSI, TruncatedSVD_LSI
from freediscovery.utils import categorization_score
from freediscovery.io import parse_ground_truth_file
from .run_suite import check_cache


def test_lsi():
    basename = os.path.dirname(__file__)

    cache_dir = check_cache()
    data_dir = os.path.join(basename, "..", "data", "ds_001", "raw")
    n_features = 110000

    fe = FeatureVectorizer(cache_dir=cache_dir)
    uuid = fe.preprocess(data_dir, file_pattern='.*\d.txt',
                         n_features=n_features)  # TODO unused variable (overwritten on the next line)
    uuid, filenames = fe.transform()
    ground_truth = parse_ground_truth_file(
                        os.path.join(data_dir, "..", "ground_truth_file.txt"))

    lsi = LSI(cache_dir=cache_dir, dsid=uuid)
    lsi_res, exp_var = lsi.transform(n_components=100)  # TODO unused variables
    lsi_id = lsi.mid
    assert lsi.get_dsid(fe.cache_dir, lsi_id) == uuid
    assert lsi.get_path(lsi_id) is not None
    assert lsi._load_pars(lsi_id) is not None
    lsi.load(lsi_id)

    idx_gt = lsi.fe.search(ground_truth.index.values)
    idx_all = np.arange(lsi.fe.n_samples_, dtype='int')

    for method in ['nearest-max', 'centroid-max']:
                        #'nearest-diff', 'nearest-combine', 'stacking']:
        _, Y_train, Y_pred, ND_train = lsi.predict(
                                idx_gt,
                                ground_truth.is_relevant.values,
                                method=method)
        scores = categorization_score(idx_gt,
                            ground_truth.is_relevant.values,
                            idx_all, Y_pred)
        assert_allclose(scores['precision'], 1, rtol=0.5)
        assert_allclose(scores['recall'], 1, rtol=0.3)


    lsi.list_models()
    lsi.delete()


def test_lsi_book_example():
    """ LSI example taken from the "Information retrieval" (2004) book by Grossman & Ophir

    This illustrates the general principle of LSI using sklearn API with TruncatedSVD_LSI
    """

    # replacing "a" with "aa" as the former seems to be ignored by the CountVectorizer
    documents = ["Shipment of gold damaged in aa fire.",
                 "Delivery of silver arrived in aa silver truck.",
                 "Shipment of gold arrived in aa truck.",
                 ]
    querry = "gold silver truck"
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import scipy.linalg
    dm_vec = CountVectorizer()
    dm_vec.fit(documents)
    X = dm_vec.transform(documents)

    assert X.shape[1] == 11
    assert X.sum() == 22  # checking the total number of elements in the document matrix

    #print(X.todense().T)
    q = dm_vec.transform([querry])

    lsi = TruncatedSVD_LSI(n_components=2)  #, algorithm='arpack')

    lsi.fit(X)
    X_p = lsi.transform_lsi(X)
    q_p = lsi.transform_lsi(q)

    U, s, Vh = scipy.linalg.svd(X.todense().T, full_matrices=False)
    #print(' ')
    #print(U[:, :-1])

    q_p_2 = q.dot(U[:,:-1]).dot(np.diag(1./s[:-1]))
    assert_allclose(np.abs(q_p_2), np.array([[0.2140, 0.1821]]), 1e-3)
    X_p_2 = X.dot(U[:,:-1]).dot(np.diag(1./s[:-1]))

    assert_allclose(np.abs(np.abs(X_p_2)), np.abs(X_p))
    assert_allclose(np.abs(np.abs(q_p_2)), np.abs(q_p))
    #print(lsi.Sigma)
    #print(' ')
    #print(X_p)
    #print(q_p)

    D = cosine_similarity(X_p, q_p)

    assert_allclose(D[:2], np.array([ -0.05, 0.9910, 0.9543])[:2,None], 2e-2, 1e-2)
