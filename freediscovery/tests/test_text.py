#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
from numpy.testing import assert_equal, assert_array_equal
from numpy.testing import assert_allclose
import scipy.sparse
import itertools
import pytest
from sklearn.preprocessing import normalize

from freediscovery.text import (FeatureVectorizer,
                                _FeatureVectorizerSampled)
from .run_suite import check_cache
from freediscovery.exceptions import (NotFound)

basename = os.path.dirname(__file__)
data_dir = os.path.join(basename, "..", "data", "ds_001", "raw")
n_features = 1100000


@pytest.mark.parametrize("analyzer, ngram_range, use_hashing",
                         list(itertools.product(['word', 'char'],
                                                [[1, 1], [1, 2]],
                                                ['hashed', 'non-hashed'])))
def test_feature_extraction_tokenization(analyzer, ngram_range, use_hashing):
    cache_dir = check_cache()
    use_hashing = (use_hashing == 'hashed')

    fe = FeatureVectorizer(cache_dir=cache_dir)
    uuid = fe.preprocess(data_dir, file_pattern='.*\d.txt',
                         analyzer=analyzer, ngram_range=ngram_range,
                         use_hashing=use_hashing)
    fe.transform()

    res2 = fe._load_features(uuid)
    assert isinstance(res2,  np.ndarray) or scipy.sparse.issparse(res2), "not an array {}".format(res2)

    assert np.isfinite(res2.data).all()

    assert_allclose(normalize(res2).data, res2.data)  # data is l2 normalized

    fe.delete()


@pytest.mark.parametrize("sublinear_tf, use_idf, binary, use_hashing",
                         list(itertools.product(['TF', 'sublinear TF'],
                                                ['', 'IDF'],
                                                ['binary', ''],
                                                ['hashed', ''])))
def test_feature_extraction_weighting(use_idf, sublinear_tf, binary,
                                      use_hashing):
    cache_dir = check_cache()

    use_idf = (use_idf == 'IDF')
    sublinear_tf = (sublinear_tf == 'sublinear TF')
    binary = (binary == 'binary')
    use_hashing = (use_hashing == 'hashed')

    fe = FeatureVectorizer(cache_dir=cache_dir)
    uuid = fe.preprocess(data_dir, file_pattern='.*\d.txt',
                         use_idf=use_idf, binary=binary,
                         use_hashing=use_hashing, sublinear_tf=sublinear_tf)
    fe.transform()

    res2 = fe._load_features(uuid)
    assert isinstance(res2,  np.ndarray) or scipy.sparse.issparse(res2), \
        "not an array {}".format(res2)

    assert np.isfinite(res2.data).all()
    assert_allclose(normalize(res2).data, res2.data)  # data is l2 normalized

    fe.delete()


@pytest.mark.parametrize("n_features, use_idf, use_hashing",
                         list(itertools.product([None, 4, 1000],
                                                ['', 'IDF'],
                                                ['hashed', ''])))
def test_feature_extraction_nfeatures(n_features, use_idf, use_hashing):
    cache_dir = check_cache()

    use_hashing = (use_hashing == 'hashed')
    use_idf = (use_idf == 'IDF')

    fe = FeatureVectorizer(cache_dir=cache_dir)
    uuid = fe.preprocess(data_dir, file_pattern='.*\d.txt',
                         n_features=n_features,
                         use_idf=use_idf, use_hashing=use_hashing)
    fe.transform()

    res2 = fe._load_features(uuid)
    assert isinstance(res2,  np.ndarray) or scipy.sparse.issparse(res2), \
        "not an array {}".format(res2)

    assert np.isfinite(res2.data).all()

    assert res2.shape[1] == fe.n_features_

    fe.delete()


@pytest.mark.parametrize('use_hashing,', [True, False])
def test_search_filenames(use_hashing):
    cache_dir = check_cache()

    fe = FeatureVectorizer(cache_dir=cache_dir)
    uuid = fe.preprocess(data_dir, file_pattern='.*\d.txt',
                         use_hashing=use_hashing)
    fe.transform()

    assert fe.db_ is not None

    for low, high, step in [(0, 1, 1),
                            (0, 4, 1),
                            (3, 1, -1)]:
        idx_slice = list(range(low, high, step))
        filenames_slice = [fe.filenames_[idx] for idx in idx_slice]
        idx0 = fe.db_._search_filenames(filenames_slice)
        assert_equal(idx0, idx_slice)
        assert_equal(filenames_slice, fe[idx0])

    with pytest.raises(NotFound):
        fe.db_._search_filenames(['DOES_NOT_EXIST.txt'])

    if not use_hashing:
        n_top_words = 5
        terms = fe.query_features([2, 3, 5], n_top_words=n_top_words)
        assert len(terms) == n_top_words

    fe.list_datasets()


@pytest.mark.parametrize('use_hashing, min_df, max_df', [[False, 0.1, 0.6],
                                                         [True,  0.1, 0.6]])
def test_df_filtering(use_hashing, min_df, max_df):
    cache_dir = check_cache()

    fe = FeatureVectorizer(cache_dir=cache_dir)
    uuid = fe.preprocess(data_dir, use_hashing=use_hashing,
                         min_df=min_df, max_df=max_df)
    fe.transform()

    X = fe._load_features(uuid)

    fe2 = FeatureVectorizer(cache_dir=cache_dir)
    uuid2 = fe2.preprocess(data_dir, use_hashing=use_hashing)
    fe2.transform()

    X2 = fe2._load_features(uuid2)

    if use_hashing:
        # min/max_df does not affect the number of features
        assert X.shape[1] == X2.shape[1]
    else:
        # min/max_df removes some features
        assert X.shape[1] < X2.shape[1]

    fe.delete()


def test_sampling_filenames():
    cache_dir = check_cache()

    fe_pars = {'binary': True, 'norm': None, 'sublinear_tf': False}

    fe = FeatureVectorizer(cache_dir=cache_dir)
    with pytest.warns(UserWarning):
        # there is a warning because we don't use norm='l2'
        uuid = fe.preprocess(data_dir, file_pattern='.*\d.txt',
                             use_hashing=True, **fe_pars)
    fe.transform()
    X = fe._load_features(uuid)

    # don't use any sampling
    fes = _FeatureVectorizerSampled(cache_dir=cache_dir, dsid=uuid,
                                    sampling_filenames=None)
    X_s = fes._load_features(uuid)
    pars = fe.pars_
    fnames = fe.filenames_
    fnames_s = fes.filenames_
    assert_array_equal(fnames, fnames_s)
    assert_array_equal(X.data, X_s.data)
    assert fes.n_samples_ == len(fnames)

    fes = _FeatureVectorizerSampled(cache_dir=cache_dir, dsid=uuid,
                                    sampling_filenames=fnames[::-1])

    assert fes.sampling_index is not None
    X_s = fes._load_features(uuid)
    pars_s = fes.pars_
    fnames_s = fes.filenames_
    assert_array_equal(fnames[::-1], fnames_s)
    assert_array_equal(X[::-1, :].data, X_s.data)
    for key in pars:
        if key == 'filenames':
            assert pars[key][::-1] == pars_s[key]
        else:
            assert pars[key] == pars_s[key]

    # repeat twice the filenames
    fes = _FeatureVectorizerSampled(cache_dir=cache_dir, dsid=uuid,
                                    sampling_filenames=(fnames+fnames))

    assert fes.sampling_index is not None
    X_s = fes._load_features(uuid)
    pars_s = fes.pars_
    fnames_s = fes.filenames_
    assert_array_equal(fnames + fnames, fnames_s)
    assert_array_equal(X.data, X_s[:len(fnames)].data)
    assert_array_equal(X.data, X_s[len(fnames):].data)
    assert fes.n_samples_ == len(fnames)*2
    # for key in pars:
    #    assert pars[key] == pars_s[key]

    # downsample the filenames
    N = len(fnames)//2

    np.random.seed(1)

    idx = np.random.choice(fe.n_samples_, size=(N,))
    fnames_s_in = np.array(fnames)[idx].tolist()

    fes = _FeatureVectorizerSampled(cache_dir=cache_dir, dsid=uuid,
                                    sampling_filenames=fnames_s_in)

    assert fes.sampling_index is not None
    X_s = fes._load_features(uuid)
    pars_s = fes.pars_
    fnames_s = fes.filenames_
    assert_array_equal(fnames_s_in, fnames_s)
    assert_array_equal(X[idx].data, X_s.data)
    assert fes.n_samples_ == N

    fe.delete()


@pytest.mark.parametrize("use_hashing", ['hashed', 'non-hashed'])
def test_feature_extraction_cyrillic(use_hashing):
    data_dir = os.path.join(basename, "..", "data", "ds_002", "raw")
    cache_dir = check_cache()
    use_hashing = (use_hashing == 'hashed')

    fe = FeatureVectorizer(cache_dir=cache_dir)
    uuid = fe.preprocess(data_dir, file_pattern='.*\d.txt',
                         use_hashing=use_hashing)
    fe.transform()

    res2 = fe._load_features(uuid)

    filenames = fe.filenames_
    fe._filenames = None
    filenames2 = fe.filenames_

    assert_equal(filenames2, filenames)
    assert isinstance(res2,  np.ndarray) or scipy.sparse.issparse(res2),\
        "not an array {}".format(res2)

    assert np.isfinite(res2.data).all()
    fe.delete()


def test_email_parsing():
    data_dir = os.path.join(basename, "..", "data",
                            "fedora-devel-list-2008-October")
    cache_dir = check_cache()

    fe = FeatureVectorizer(cache_dir=cache_dir)
    uuid = fe.preprocess(data_dir)
    fe.transform()

    email_md = fe.parse_email_headers()
    assert len(fe.filenames_) == len(email_md)

    fe.delete()
