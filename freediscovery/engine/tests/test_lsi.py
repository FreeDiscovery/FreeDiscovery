# -*- coding: utf-8 -*-

import os.path

from numpy.testing import assert_allclose, assert_equal

from sklearn.preprocessing import normalize
import pytest


from freediscovery.engine.vectorizer import FeatureVectorizer
from freediscovery.engine.lsi import _LSIWrapper
from freediscovery.engine.ingestion import DocumentIndex
from freediscovery.exceptions import WrongParameter
from freediscovery.tests.run_suite import check_cache

basename = os.path.dirname(__file__)
data_dir = os.path.join(basename, "..", "..", "data", "ds_001", "raw")


def test_lsi():

    cache_dir = check_cache()
    n_components = 2

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid = fe.setup()
    fe.ingest(data_dir, file_pattern='.*\d.txt')

    lsi = _LSIWrapper(cache_dir=cache_dir, parent_id=uuid, mode='w')
    lsi_res, exp_var = lsi.fit_transform(n_components=n_components, alpha=1.0)
    assert lsi_res.components_.shape[0] == 5
    assert lsi_res.components_.shape[1] == fe.n_features_
    assert lsi._load_pars() is not None
    lsi._load_model()
    X_lsi = lsi._load_features()

    assert_allclose(normalize(X_lsi), X_lsi)

    lsi.list_models()
    lsi.delete()


def test_lsi_append_documents():
    cache_dir = check_cache()

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid = fe.setup()
    fe.ingest(data_dir)

    lsi = _LSIWrapper(cache_dir=cache_dir, parent_id=uuid, mode='w')
    lsi_res, exp_var = lsi.fit_transform(n_components=2, alpha=1.0)
    X_lsi = lsi._load_features()
    n_samples = fe.n_samples_

    docs = DocumentIndex.from_folder(data_dir).data
    docs['document_id'] += 10
    dataset_definition = docs[['file_path', 'document_id']].to_dict(orient='records')
    for row in dataset_definition:
        row['file_path'] = os.path.join(data_dir, row['file_path'])
    fe.append(dataset_definition)

    X_lsi_new = lsi._load_features()
    assert X_lsi_new.shape[0] == X_lsi.shape[0]*2
    assert_equal(X_lsi_new[:n_samples], X_lsi_new[:n_samples])


def test_lsi_remove_documents():
    cache_dir = check_cache()

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid = fe.setup()
    fe.ingest(data_dir)

    lsi = _LSIWrapper(cache_dir=cache_dir, parent_id=uuid, mode='w')
    lsi_res, exp_var = lsi.fit_transform(n_components=2, alpha=1.0)
    X_lsi = lsi._load_features()

    docs = DocumentIndex.from_folder(data_dir).data
    dataset_definition = docs[['document_id']].to_dict(orient='records')
    fe.remove([dataset_definition[2], dataset_definition[4]])

    X_lsi_new = lsi._load_features()
    assert X_lsi_new.shape[0] == X_lsi.shape[0] - 2


def test_custom_mid():
    cache_dir = check_cache()

    fe = FeatureVectorizer(cache_dir=cache_dir, mode='w')
    uuid = fe.setup()
    fe.ingest(data_dir)

    mid_orig = "sklds"

    lsi = _LSIWrapper(cache_dir=cache_dir, parent_id=uuid, mid=mid_orig,
                      mode='w')
    lsi_res, exp_var = lsi.fit_transform(n_components=2, alpha=1.0)
    lsi._load_features()

    assert lsi.mid == mid_orig

    with pytest.raises(WrongParameter):
        lsi = _LSIWrapper(cache_dir=cache_dir, parent_id=uuid, mid=mid_orig,
                          mode='w')
        lsi.fit_transform(n_components=2, alpha=1.0)

    lsi = _LSIWrapper(cache_dir=cache_dir, parent_id=uuid, mid=mid_orig,
                      mode='fw')
    lsi.fit_transform(n_components=2, alpha=1.0)

    with pytest.raises(WrongParameter):
        lsi = _LSIWrapper(cache_dir=cache_dir, parent_id=uuid, mid='?',
                          mode='fw')
        lsi.fit_transform(n_components=2, alpha=1.0)
