# -*- coding: utf-8 -*-

import os
import pytest
import pandas as pd
import numpy as np
from numpy.testing import (assert_array_equal, assert_array_less,
                           assert_allclose)
from sklearn.externals import joblib

from ...utils import dict2type

from .base import (parse_res, V01, app, get_features_cached,
                   get_features_lsi_cached, CACHE_DIR)


@pytest.mark.parametrize("method, min_score, max_results",
                         [('regular', -1, None),
                          ('semantic', -1, None),
                          ('semantic', 0.2, None),
                          ('semantic', -1, 1000000),
                          ('semantic', -1, 3),
                          ('semantic', -1, 0)])
def test_search(app, method, min_score, max_results):

    dsid, lsi_id, _, input_ds = get_features_lsi_cached(app)
    if method == 'semantic':
        parent_id = lsi_id
    elif method == 'regular':
        lsi_id = None
        parent_id = dsid
    query = """The American National Standards Institute sells ANSI standards, and also
    ISO (international) standards.  Their sales office is at 1-212-642-4900,
    mailing address is 1430 Broadway, NY NY 10018.  It helps if you have the
    complete name and number.
    """
    query_file_path = "02256.txt"

    pars = dict(parent_id=parent_id,
                min_score=min_score,
                query=query)
    if max_results is not None:
        pars['max_results'] = max_results

    data = app.post_check(V01 + "/search/", json=pars)
    assert sorted(data.keys()) == ['data', 'pagination']
    data = data['data']
    for row in data:
        assert dict2type(row) == {'score': 'float',
                                  'document_id': 'int'}
    scores = np.array([row['score'] for row in data])
    assert (np.diff(scores) <= 0).all()
    assert_array_less(min_score, scores)
    if max_results:
        assert len(data) == min(max_results, len(input_ds['dataset']))
    elif min_score > -1:
        pass
    else:
        assert len(data) == 1967

    data = app.post_check(V01 + "/feature-extraction/{}/id-mapping"
                          .format(dsid), 
                          json={'data': [data[0]]})

    if not max_results:
        assert data['data'][0]['file_path'] == query_file_path


def test_search_retrieve_batch(app):
    dsid, lsi_id, _, input_ds = get_features_lsi_cached(app)
    parent_id = lsi_id

    max_results = -1
    query_document_id = 3844
    total_document_number = 1967 - 1  # removing the query one

    for batch_id, batch_size in [(-1, 1000),
                                 (0, 983),
                                 (1, 983),
                                 (2, 983),
                                 (1, 1000)]:

        pars = dict(parent_id=parent_id,
                    max_results=max_results,
                    sort=True,
                    query_document_id=query_document_id,
                    batch_id=batch_id,
                    batch_size=batch_size)

        data = app.post_check(V01 + "/search/", json=pars)
        assert sorted(data.keys()) == ['data', 'pagination']
        for row in data['data']:
            assert dict2type(row) == {'score': 'float',
                                      'document_id': 'int'}
        assert dict2type(data['pagination']) == {
                                  'total_response_count': 'int',
                                  'current_response_count': 'int',
                                  'batch_id': 'int',
                                  'batch_id_last': 'int'}
        scores = np.array([row['score'] for row in data['data']])
        assert (np.diff(scores) <=  0).all()
        assert len(data['data']) == data['pagination']['current_response_count']
        assert data['pagination']['total_response_count'] == total_document_number
        assert data['pagination']['batch_id'] == batch_id
        if batch_id == -1:
            assert len(data['data']) == total_document_number
            assert data['pagination']['batch_id_last'] == batch_id
        elif batch_id >= 0:
            assert data['pagination']['current_response_count'] <= batch_size
            if batch_id == 1 and batch_size == 983:
                assert data['pagination']['current_response_count'] == batch_size
            elif batch_id == 2:
                assert data['pagination']['current_response_count'] == 0


def test_search_document_id(app):
    dsid, lsi_id, _, input_ds = get_features_lsi_cached(app, hashed=False)
    parent_id = lsi_id

    max_results = 2
    query_document_id = 3844

    pars = dict(parent_id=parent_id,
                max_results=max_results,
                sort=True,
                query_document_id=query_document_id)

    data = app.post_check(V01 + "/search/", json=pars)
    assert sorted(data.keys()) == ['data', 'pagination']
    data = data['data']
    for row in data:
        assert dict2type(row) == {'score': 'float',
                                  'document_id': 'int'}
    scores = np.array([row['score'] for row in data])
    assert (np.diff(scores) <= 0).all()
    assert len(data) == min(max_results, len(input_ds['dataset']))
    # assert data[0]['document_id'] == query_document_id
    # assert data[0]['score'] >= 0.99


def test_search_subset_document_id(app):
    dsid, lsi_id, _, input_ds = get_features_lsi_cached(app, hashed=False)
    parent_id = lsi_id

    query_document_id = 3844

    pars = dict(parent_id=parent_id,
                sort=True,
                query_document_id=query_document_id)

    # last document id is invalid
    pars['subset_document_id'] = [3705625, 1110916, 999999999]
    data = app.post_check(V01 + "/search/", json=pars)
    data = data['data']
    # check that the response only includes documents from the subset,
    # in the correct order
    assert_array_equal([row['document_id'] for row in data],
                       [1110916, 3705625])


def test_search_consistency(app):
    """ A number of consistency checks"""
    import pickle
    from scipy.stats import pearsonr
    from sklearn.metrics.pairwise import cosine_similarity
    dataset_id, lsi_id, ds_pars, input_ds = get_features_lsi_cached(app)
    query_document_id = 1

    input_ds = pd.DataFrame(input_ds['dataset']).set_index('document_id')

    # compute semantic search
    pars = dict(parent_id=lsi_id,
                query_document_id=query_document_id)
    data = app.post_check(V01 + "/search/", json=pars)
    df_s = pd.DataFrame(data['data']).set_index('document_id')
    df_s = df_s.merge(input_ds, how='left', left_index=True, right_index=True)

    # manually compute the similarity for a pair of documents and
    # check that it's the same as the one computed by the system
    with open(os.path.join(CACHE_DIR, 'ediscovery_cache', dataset_id, 'vectorizer'), 'rb') as fh:
        vect = pickle.load(fh)
    lsi_est = joblib.load(os.path.join(CACHE_DIR, 'ediscovery_cache', dataset_id,
                          'lsi', lsi_id, 'model'))
    X_tmp = []
    comp_document_id = 2365444
    for document_id in [query_document_id, comp_document_id]:
        X_tmp.append(os.path.join(ds_pars['data_dir'],
                                  input_ds.loc[document_id].file_path))
    X_tmp = vect.transform(X_tmp)
    X_tmp = lsi_est.transform_lsi_norm(X_tmp)
    assert_allclose(cosine_similarity(X_tmp[[0]], X_tmp[[1]]),
                    df_s.loc[comp_document_id].score)

    # check that providing the query_document_id or the same document as text
    # produces the same results
    with open(os.path.join(ds_pars['data_dir'],
                           input_ds.loc[query_document_id].file_path), 'rt') as fh:
        query_txt = fh.read()
    pars = dict(parent_id=lsi_id,
                query=query_txt)
    data = app.post_check(V01 + "/search/", json=pars)
    df_s_txt = pd.DataFrame(data['data']).set_index('document_id')
    df_s_txt = df_s_txt.merge(input_ds, how='left',
                              left_index=True, right_index=True)
    df_s_txt.loc[query_document_id].score == 1.0
    df_s_txt = df_s_txt[df_s_txt.index != query_document_id]
    assert_allclose(df_s.score.values, df_s_txt.score.values)

    # check that query document is on average closer to the documents of its class
    query_category = input_ds.loc[query_document_id].category
    assert df_s[df_s.category == query_category].score.quantile(q=0.75) \
        > df_s[df_s.category != query_category].score.quantile(q=0.75)


def test_search_eq_categorization(app):
    """Check that NN categorization with a single training document and semantic search
    returns the same results
    """
    dataset_id, lsi_id, ds_pars, input_ds = get_features_lsi_cached(app)
    query_document_id = 1

    input_ds = pd.DataFrame(input_ds['dataset']).set_index('document_id')

    # compute regular search
    pars = dict(parent_id=lsi_id,
                query_document_id=query_document_id)
    data = app.post_check(V01 + "/search/", json=pars)
    df_s = pd.DataFrame(data['data']).set_index('document_id')

    pars = {
          'parent_id': lsi_id,
          'data': [{'document_id': query_document_id, 'category': 'search'}],
          'method': 'NearestNeighbor',
          'subset': 'test'
           }

    method = V01 + "/categorization/"
    data = app.post_check(method, json=pars)

    method = V01 + "/categorization/{}/predict".format(data['id'])
    data = app.get_check(method)
    categories = [el['scores'][0]['category'] for el in data['data']]
    assert categories == ['search']*len(categories)

    df_c = pd.DataFrame([{'document_id': row['document_id'],
                          'score': row['scores'][0]['score']}
                         for row in data['data']]).set_index('document_id')
    # The results do not appear to be exactly matching (probably due to tie-breaking)
    assert (df_c.index == df_s.index).sum() / df_s.shape[0] > 0.995
    assert_allclose(df_c.score.values, df_s.score.values)


def test_ingestion_no_document_id_provided(app):
    """ Test what happens if no document id are provided during ingestion
    We use search merely because it's the simplest processing that can be done
    in a single API call"""
    data_dir = os.path.dirname(__file__)
    data_dir = os.path.join(data_dir, "..", "..", "data", "ds_001", "raw")
    method = V01 + "/feature-extraction/"
    data = app.post_check(method)
    dsid = data['id']
    app.post_check(method + dsid, json={'data_dir': data_dir})

    data = app.post_check(V01 + "/search/", json={'parent_id': dsid,
                                                  'query': 'test'})
    assert dict2type(data['data'][0]) == {'score': 'float',
                                          'document_id': 'int'}
