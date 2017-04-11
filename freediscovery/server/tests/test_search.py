# -*- coding: utf-8 -*-

import os
import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_equal, assert_array_less

from ...utils import dict2type

from .base import (parse_res, V01, app, get_features_cached,
                   get_features_lsi_cached, CACHE_DIR)


@pytest.mark.parametrize("method, min_score, max_results",
                         [('regular', -1, None),
                          ('semantic', -1, None),
                          ('semantic', 0.5, None),
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
    assert_equal(np.diff(scores) <= 0, True)
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
        assert_equal(np.diff(scores) <= 0, True)
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
    assert_equal(np.diff(scores) <= 0, True)
    assert len(data) == min(max_results, len(input_ds['dataset']))
    # assert data[0]['document_id'] == query_document_id
    # assert data[0]['score'] >= 0.99



def test_search_consistency(app):
    import pickle
    from scipy.stats import pearsonr
    from sklearn.metrics.pairwise import cosine_similarity
    dataset_id, lsi_id, ds_pars, input_ds = get_features_lsi_cached(app)
    query_document_id = 3034564

    input_ds = pd.DataFrame(input_ds['dataset']).set_index('document_id')

    # compute regular search
    pars = dict(parent_id=dataset_id,
                query_document_id=query_document_id)
    data = app.post_check(V01 + "/search/", json=pars)
    df_n = pd.DataFrame(data['data']).set_index('document_id')
    df_n = df_n.merge(input_ds, how='left', left_index=True, right_index=True)

    # manually compute the similarity to a few documents
    with open(os.path.join(CACHE_DIR, 'ediscovery_cache', dataset_id, 'vectorizer'), 'rb') as fh:
        vect = pickle.load(fh)
    print(vect)
    X_tmp = []
    comp_document_id = 2365444
    for document_id in [query_document_id, comp_document_id]:
        X_tmp.append(os.path.join(ds_pars['data_dir'],
                                  input_ds.loc[query_document_id].file_path))
    X_tmp = vect.transform(X_tmp)
    print(X_tmp)
    

    # compute semantic search
    pars = dict(parent_id=lsi_id,
                query_document_id=query_document_id)
    data = app.post_check(V01 + "/search/", json=pars)
    df_s = pd.DataFrame(data['data']).set_index('document_id')
    df_s = df_s.merge(input_ds, how='left', left_index=True, right_index=True)

    # check that providing the query_document_id or the same document as text
    # produces the same results
    with open(os.path.join(ds_pars['data_dir'],
                           input_ds.loc[query_document_id].file_path), 'rt') as fh:
        query_txt = fh.read()
    pars = dict(parent_id=lsi_id,
                query=query_txt)
    data = app.post_check(V01 + "/search/", json=pars)
    df_s_txt = pd.DataFrame(data['data']).set_index('document_id')
    df_s_txt = df_s_txt.merge(input_ds, how='left', left_index=True, right_index=True)
    df_s_txt.loc[query_document_id].score == 1.0
    df_s_txt = df_s_txt[df_s_txt.index != query_document_id]
    assert_equal(df_s.score.values, df_s_txt.score.values)

    df_m = df_s.merge(df_n, how='left', left_index=True, right_index=True,
                      suffixes=('_s', '_n'))
    #print(df_m)
    #df_m.to_pickle('/tmp/search_ex.pkl')
    #assert pearsonr(df_m.score_s.values, df_m.score_n.values)[0] > 1.0 

    # check that query document is on average closer to the documents of its class
    query_category = input_ds.loc[query_document_id].category
    print(df_s[df_s.category == query_category].score.mean())
    print(df_s[df_s.category != query_category].score.mean())

    print(ds_pars)
