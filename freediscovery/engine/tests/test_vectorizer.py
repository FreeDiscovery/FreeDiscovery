# -*- coding: utf-8 -*-

import os.path
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_array_equal
from numpy.testing import assert_allclose
import scipy.sparse
import itertools
import pytest
from sklearn.preprocessing import normalize

from freediscovery.engine.vectorizer import (FeatureVectorizer,
                                             _FeatureVectorizerSampled)
from freediscovery.engine.ingestion import DocumentIndex

from freediscovery.tests.run_suite import check_cache
from freediscovery.exceptions import (NotFound, WrongParameter)

basename = os.path.dirname(__file__)
data_dir = os.path.join(basename, "..", "..", "data", "ds_001", "raw")
n_features = 1100000


@pytest.mark.parametrize("analyzer, ngram_range, use_hashing",
                         list(itertools.product(['word', 'char'],
                                                [[1, 1], [1, 2]],
                                                ['hashed', 'non-hashed'])))
def test_feature_extraction_tokenization(analyzer, ngram_range, use_hashing):
    cache_dir = check_cache()
    use_hashing = (use_hashing == 'hashed')

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid = fe.setup(analyzer=analyzer, ngram_range=ngram_range,
                    use_hashing=use_hashing)
    fe.ingest(data_dir, file_pattern='.*\d.txt')

    res2 = fe._load_features(uuid)
    assert isinstance(res2,  np.ndarray) or scipy.sparse.issparse(res2), "not an array {}".format(res2)

    assert np.isfinite(res2.data).all()

    assert_allclose(normalize(res2).data, res2.data)  # data is l2 normalized

    fe.delete()


def test_feature_extraction_storage():
    cache_dir = check_cache()

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid = fe.setup()
    fe.ingest(data_dir, file_pattern='.*\d.txt')
    db = pd.read_pickle(os.path.join(cache_dir, 'ediscovery_cache',
                                     uuid, 'db'))
    assert 'file_path' not in db.columns


@pytest.mark.parametrize("weighting, use_hashing",
                         list(itertools.product(("".join(el) for el in itertools.product('nlb', 'ns', 'c')),
                                                ['hashed', ''])))
def test_feature_extraction_weighting(weighting,
                                      use_hashing):
    cache_dir = check_cache()

    use_hashing = (use_hashing == 'hashed')

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid = fe.setup(weighting=weighting, use_hashing=use_hashing)
    fe.ingest(data_dir, file_pattern='.*\d.txt')

    res2 = fe._load_features(uuid)
    assert isinstance(res2,  np.ndarray) or scipy.sparse.issparse(res2), \
        "not an array {}".format(res2)

    assert np.isfinite(res2.data).all()
    assert_allclose(normalize(res2).data, res2.data)  # data is l2 normalized

    fe.delete()


@pytest.mark.parametrize("n_features, weighting, use_hashing",
                         list(itertools.product([None, 4, 1000],
                                                ['nnc', 'nsc'],
                                                ['hashed', ''])))
def test_feature_extraction_nfeatures(n_features, weighting, use_hashing):
    cache_dir = check_cache()

    use_hashing = (use_hashing == 'hashed')

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid = fe.setup(n_features=n_features, weighting=weighting, use_hashing=use_hashing)
    fe.ingest(data_dir, file_pattern='.*\d.txt')

    res2 = fe._load_features(uuid)
    assert isinstance(res2,  np.ndarray) or scipy.sparse.issparse(res2), \
        "not an array {}".format(res2)

    assert np.isfinite(res2.data).all()

    assert res2.shape[1] == fe.n_features_

    fe.delete()


@pytest.mark.parametrize('use_hashing,', [True, False])
def test_search_filenames(use_hashing):
    cache_dir = check_cache()

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid = fe.setup(use_hashing=use_hashing)
    fe.ingest(data_dir, file_pattern='.*\d.txt')

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

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid = fe.setup(min_df=min_df, max_df=max_df, use_hashing=use_hashing)
    fe.ingest(data_dir)

    X = fe._load_features(uuid)

    fe2 = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid2 = fe2.setup(use_hashing=use_hashing)
    fe2.ingest(data_dir)

    X2 = fe2._load_features(uuid2)

    if use_hashing:
        # min/max_df does not affect the number of features
        assert X.shape[1] == X2.shape[1]
    else:
        # min/max_df removes some features
        assert X.shape[1] < X2.shape[1]

    fe.delete()


def test_append_documents():
    cache_dir = check_cache()

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid = fe.setup()
    fe.ingest(data_dir)

    X = fe._load_features(uuid)
    db = fe.db_
    filenames = fe.filenames_
    n_samples = len(fe.filenames_)

    docs = DocumentIndex.from_folder(data_dir).data
    docs['document_id'] += 10
    dataset_definition = docs[['file_path', 'document_id']].to_dict(orient='records')
    for row in dataset_definition:
        row['file_path'] = os.path.join(data_dir, row['file_path'])
    fe.append(dataset_definition)
    X_new = fe._load_features(uuid)
    assert X_new.shape[0] == X.shape[0]*2
    assert fe.db_.data.shape[0] == db.data.shape[0]*2
    assert len(fe.filenames_) == len(filenames)*2

    dbn = fe.db_.data
    assert_equal(dbn.iloc[:n_samples]['document_id'].values,
                 dbn.iloc[n_samples:]['document_id'].values - 10)
    # check that internal id is contiguous
    assert (np.diff(dbn.internal_id.values) == 1).all()

    # check the number of samples is consistent
    del fe._pars
    assert fe.n_samples_ == n_samples * 2

    fe.delete()


def test_remove_documents():
    cache_dir = check_cache()

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid = fe.setup()
    fe.ingest(data_dir)

    X = fe._load_features(uuid)
    db = fe.db_.data
    filenames = fe.filenames_
    n_samples = len(fe.filenames_)

    docs = DocumentIndex.from_folder(data_dir).data
    dataset_definition = docs[['document_id']].to_dict(orient='records')
    fe.remove([dataset_definition[2], dataset_definition[4]])
    X_new = fe._load_features(uuid)
    assert X_new.shape[0] == X.shape[0] - 2
    assert fe.db_.data.shape[0] == db.shape[0] - 2
    assert len(fe.filenames_) == len(filenames) - 2

    dbn = fe.db_.data
    assert_equal(db.iloc[[0, 1, 3, 5]]['document_id'].values,
                 dbn['document_id'].values)
    # check that internal id is contiguous
    assert (np.diff(dbn.internal_id.values) == 1).all()

    # check the number of samples is consistent
    del fe._pars
    assert fe.n_samples_ == n_samples - 2

    fe.delete()


def test_sampling_filenames():
    cache_dir = check_cache()

    fe_pars = {'weighting': 'bnn'}

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    with pytest.warns(UserWarning):
        # there is a warning because we don't use norm='l2'
        uuid = fe.setup(use_hashing=True, **fe_pars)
        fe.ingest(data_dir, file_pattern='.*\d.txt')
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
    data_dir = os.path.join(basename, "..", "..", "data", "ds_002", "raw")
    cache_dir = check_cache()
    use_hashing = (use_hashing == 'hashed')

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid = fe.setup(use_hashing=use_hashing)
    fe.ingest(data_dir, file_pattern='.*\d.txt')

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
    data_dir = os.path.join(basename, "..", "..", "data",
                            "fedora-devel-list-2008-October")
    cache_dir = check_cache()

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid = fe.setup()
    fe.ingest(data_dir)

    email_md = fe.parse_email_headers()
    assert len(fe.filenames_) == len(email_md)

    fe.delete()


def test_ingestion_batches():
    data_dir = os.path.join(basename, "..", "..", "data", "ds_002", "raw")
    cache_dir = check_cache()

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid = fe.setup()
    with pytest.raises(ValueError):
        fe.ingest(vectorize=True)  # no ingested files
    fe.ingest(data_dir, file_pattern='.*\d.txt', vectorize=False)
    fe.ingest(data_dir, file_pattern='.*\d.txt', vectorize=False)
    fe.ingest(data_dir, file_pattern='.*\d.txt', vectorize=False)

    fe.ingest(vectorize=True)

    assert fe.db_.data.shape[0] == len(fe.filenames_)
    assert len(fe.filenames_) == 6*3
    X = fe._load_features()
    assert X.shape[0] == 6*3

    with pytest.raises(ValueError):
        fe.ingest(vectorize=True)  # already vectorized


def test_ingestion_content():
    data_dir = Path(basename, "..", "..", "data", "ds_002", "raw")

    dd = []
    for idx, fname in enumerate(sorted(data_dir.glob('*txt'))):
        with fname.open('rt', encoding='utf-8') as fh:
            dd.append({'document_id': idx + 19,
                       'content': fh.read()})
    cache_dir = check_cache()

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid = fe.setup()
    fe.ingest(dataset_definition=dd, vectorize=True)
    assert len(fe.filenames_) == 6
    assert fe.filenames_[0] == '000000000_0.txt'
    X = fe._load_features()
    assert X.shape[0] == 6
    assert fe.db_.data.shape[0] == len(fe.filenames_)

    fe2 = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    fe2.setup()
    fe2.ingest(data_dir=str(data_dir))

    X2 = fe2._load_features()
    assert X.shape == X2.shape
    assert_array_equal(X.indices, X2.indices)
    assert_array_equal(X.data, X2.data)

def test_non_random_dsid():
    data_dir = os.path.join(basename, "..", "..", "data", "ds_002", "raw")
    cache_dir = check_cache()

    dsid = 'test-dataset'

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w', dsid=dsid)
    uuid = fe.setup()
    assert dsid == uuid
    fe.ingest(data_dir, file_pattern='.*\d.txt', vectorize=False)
    # writing with the same name fails
    with pytest.raises(WrongParameter):
        FeatureVectorizer(cache_dir=cache_dir, mode='w', dsid=dsid)

    FeatureVectorizer(cache_dir=cache_dir, mode='r', dsid=dsid)

    FeatureVectorizer(cache_dir=cache_dir, mode='fw', dsid=dsid)
    # special characters are not allowed
    with pytest.raises(WrongParameter):
        fh = FeatureVectorizer(cache_dir=cache_dir, mode='fw', dsid='?+ds$$')
        uuid = fh.setup()
