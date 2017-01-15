#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re

import numpy as np
from numpy.testing import assert_allclose
import pytest

from freediscovery.base import PipelineFinder
from freediscovery.text import FeatureVectorizer
from freediscovery.lsi import _LSIWrapper, _TruncatedSVD_LSI
from freediscovery.utils import categorization_score
from freediscovery.io import parse_ground_truth_file
from .run_suite import check_cache


def test_lsi():
    basename = os.path.dirname(__file__)

    cache_dir = check_cache()
    data_dir = os.path.join(basename, "..", "data", "ds_001", "raw")
    n_features = 110000
    n_components = 5

    fe = FeatureVectorizer(cache_dir=cache_dir)
    uuid = fe.preprocess(data_dir, file_pattern='.*\d.txt',
                         n_features=n_features)  # TODO unused variable (overwritten on the next line)
    uuid, filenames = fe.transform()

    lsi = _LSIWrapper(cache_dir=cache_dir, parent_id=uuid)
    lsi_res, exp_var = lsi.transform(n_components=n_components)  # TODO unused variables
    lsi_id = lsi.mid
    assert lsi_res.components_.shape == (n_components, n_features)
    assert lsi.get_dsid(fe.cache_dir, lsi_id) == uuid
    assert lsi.get_path(lsi_id) is not None
    assert lsi._load_pars() is not None
    lsi._load_model()

    # test pipeline
    pf = PipelineFinder.by_id(lsi_id, cache_dir)

    assert list(pf.keys()) == ['vectorizer', 'lsi']
    assert list(pf.parent.keys()) == ['vectorizer']

    assert pf.mid == lsi_id
    assert pf.parent.mid == uuid
    with pytest.raises(ValueError):
        pf.parent.parent

    for estimator_type, mid in pf.items():
        path = pf.get_path(mid, absolute=False)
        if estimator_type == 'vectorizer':
            assert re.match('ediscovery_cache.*', path)
        elif estimator_type == 'lsi':
            assert re.match('ediscovery_cache.*lsi', path)
        else:
            raise ValueError


    lsi.list_models()
    lsi.delete()


def test_lsi_helper_class():
    import scipy.sparse

    X = scipy.sparse.rand(100, 10000)
    lsi = _TruncatedSVD_LSI(n_components=20)
    lsi.fit(X)
    X_p = lsi.transform_lsi(X)
    X_p2 = lsi.transform_lsi_norm(X)
    assert lsi.components_.shape == (20, 10000)
    assert X_p.shape == (100, 20)
    assert X_p2.shape == (100, 20)



def test_lsi_book_example():
    """ LSI example taken from the "Information retrieval" (2004) book by Grossman & Ophir

    This illustrates the general principle of LSI using sklearn API with _TruncatedSVD_LSI
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

    lsi = _TruncatedSVD_LSI(n_components=2)  #, algorithm='arpack')

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
