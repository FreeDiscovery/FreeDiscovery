# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_allclose

from freediscovery.lsi import (_TruncatedSVD_LSI,
                               _compute_lsi_dimensionality)


def test_compute_lsi_dimensionality():
    n_components = 200
    n_components_res = _compute_lsi_dimensionality(n_components, 10000, 100000)
    assert n_components_res == n_components
    n_components_res = _compute_lsi_dimensionality(n_components, 500, 100000)
    assert n_components_res < n_components


def test_lsi_helper_class():
    import scipy.sparse

    X = scipy.sparse.rand(100, 10000)
    lsi = _TruncatedSVD_LSI(n_components=20)
    lsi.fit(X)
    X_p = lsi.transform_lsi(X)
    X_p2 = lsi.transform_lsi_norm(X)
    assert lsi.components_.shape == (20, X.shape[1])
    assert X_p.shape == (100, 20)
    assert X_p2.shape == (100, 20)


def test_lsi_book_example():
    """ LSI example taken from the "Information retrieval" (2004)
    book by Grossman & Ophir

    This illustrates the general principle of LSI using
    sklearn API with _TruncatedSVD_LSI
    """

    # replacing "a" with "aa" as the former seems
    # to be ignored by the CountVectorizer
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
    # checking the total number of elements in the document matrix
    assert X.sum() == 22

    q = dm_vec.transform([querry])

    lsi = _TruncatedSVD_LSI(n_components=2)  # algorithm='arpack')

    lsi.fit(X)
    X_p = lsi.transform_lsi(X)
    q_p = lsi.transform_lsi(q)

    U, s, Vh = scipy.linalg.svd(X.todense().T, full_matrices=False)

    q_p_2 = q.dot(U[:, :-1]).dot(np.diag(1./s[:-1]))
    assert_allclose(np.abs(q_p_2), np.array([[0.2140, 0.1821]]), 1e-3)
    X_p_2 = X.dot(U[:, :-1]).dot(np.diag(1./s[:-1]))

    assert_allclose(np.abs(np.abs(X_p_2)), np.abs(X_p))
    assert_allclose(np.abs(np.abs(q_p_2)), np.abs(q_p))

    D = cosine_similarity(X_p, q_p)

    assert_allclose(D[:2],
                    np.array([-0.05, 0.9910, 0.9543])[:2, None], 2e-2, 1e-2)
