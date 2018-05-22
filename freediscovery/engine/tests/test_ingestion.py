# -*- coding: utf-8 -*-

import os.path
from numpy.testing import assert_equal, assert_array_equal
from pandas.util.testing import assert_frame_equal
import pytest
import pandas as pd

from freediscovery.engine.ingestion import DocumentIndex, _infer_document_id_from_path
from freediscovery.tests.run_suite import check_cache
from freediscovery.exceptions import (NotFound)

basename = os.path.dirname(__file__)
data_dir = os.path.join(basename, "..", "..", "data", "ds_001", "raw")
cache_dir = check_cache()


fnames_in = ['0.7.47.101442.txt',
             '0.7.47.117435.txt',
             '0.7.6.28635.txt',
             '0.7.6.28636.txt',
             '0.7.6.28637.txt',
             '0.7.6.28638.txt']
fnames_in_abs = [os.path.join(data_dir, el) for el in fnames_in]


def test_ingestion_base_dir():
    dbi = DocumentIndex.from_folder(data_dir)
    dbi._make_relative_paths()
    data_dir_res, filenames, db = dbi.data_dir, dbi.filenames_, dbi.data
    assert data_dir_res == os.path.normpath(data_dir)
    assert_array_equal(db.columns.values, ['file_path', 'internal_id', 'document_id'])
    assert_array_equal(db.file_path.values, fnames_in)
    assert_array_equal([os.path.normpath(os.path.join(data_dir_res, el))
                        for el in filenames],
                       [os.path.join(data_dir_res, el)
                        for el in db.file_path.values])


def test_search_2fields():
    dbi = DocumentIndex.from_folder(data_dir)
    dbi._make_relative_paths()

    query = pd.DataFrame([{'internal_id': 3},
                          {'internal_id': 1},
                          {'internal_id': 2}])
    sres = dbi.search(query)
    assert_equal(sres.internal_id.values, [3, 1, 2])
    assert_array_equal(sorted(sres.columns),
                       sorted(['internal_id', 'file_path', 'document_id']))

    # make sure that if we have some additional field,
    # we still use the internal_id
    query = pd.DataFrame([{'internal_id': 1, 'a': 2},
                          {'internal_id': 2, 'b': 4},
                          {'internal_id': 1, 'a': 3}])
    sres = dbi.search(query)
    assert_equal(sres.internal_id.values, [1, 2, 1])
    assert_array_equal(sorted(sres.columns),
                       sorted(['internal_id', 'file_path', 'document_id']))

    sres = dbi.search(query, drop=False)
    assert_equal(sres.internal_id.values, [1, 2, 1])
    assert_array_equal(sorted(sres.columns),
                       sorted(['internal_id', 'file_path', 'document_id',
                               'a', 'b']))

    query = pd.DataFrame([{'file_path': "0.7.6.28637.txt"},
                          {'file_path': "0.7.47.117435.txt"}])
    del dbi.data['file_path']
    sres = dbi.search(query)
    query_res = [dbi.data.file_path.values.tolist().index(el)
                 for el in query.file_path.values]
    assert_array_equal(query_res, sres.internal_id)


def test_search_not_found():
    dbi = DocumentIndex.from_folder(data_dir)
    query = pd.DataFrame([{'file_path': "DOES_NOT_EXISTS"},
                          {'file_path': "0.7.6.28637.txt"}])
    with pytest.raises(NotFound):
        sres = dbi.search(query)


@pytest.mark.parametrize('return_file_path',
                         ['return_file_path', 'dont_return_file_path'])
def test_ingestion_render(return_file_path):

    def _process_results(rd):
        rd = pd.DataFrame(rd)
        if return_file_path:
            assert 'file_path' in rd.columns
            del rd['file_path']
        return rd

    # make it a binary variable
    return_file_path = (return_file_path == 'return_file_path')

    md = [{'file_path': '/test',  'document_id': 2},
          {'file_path': '/test2', 'document_id': 1},
          {'file_path': '/test3', 'document_id': 7},
          {'file_path': '/test8', 'document_id': 9},
          {'file_path': '/test9', 'document_id': 4}]

    for idx, el in enumerate(md):
        el['internal_id'] = idx

    dbi = DocumentIndex.from_list(md)
    query = pd.DataFrame([{'a': 2, 'internal_id': 3},
                          {'a': 4, 'internal_id': 1}])
    res = pd.DataFrame([{'a': 2, 'internal_id': 3, 'document_id': 9},
                        {'a': 4, 'internal_id': 1, 'document_id': 1}])

    rd = dbi.render_dict(query, return_file_path=return_file_path)
    rd = _process_results(rd)
    assert_frame_equal(rd, res)
    rd = dbi.render_dict(return_file_path=return_file_path)
    rd = _process_results(rd)
    assert_frame_equal(rd.loc[[0]],
                       pd.DataFrame([{'internal_id': 0, 'document_id': 2}]))
    assert len(rd) == len(md)

    rd = dbi.render_list(res, return_file_path=return_file_path)
    rd = _process_results(rd)
    assert sorted(rd.keys()) == sorted(['internal_id', 'document_id', 'a'])
    assert_frame_equal(pd.DataFrame(rd),
                       pd.DataFrame([{'a': 2, 'internal_id': 3, 'document_id': 9},
                                     {'a': 4, 'internal_id': 1, 'document_id': 1}]),
                       check_like=True)

    rd = dbi.render_list()
    assert sorted(rd.keys()) == sorted(['internal_id', 'document_id'])


def test_search_document_id():
    md = [{'file_path': '/test',  'document_id': 2},
          {'file_path': '/test2', 'document_id': 1},
          {'file_path': '/test3', 'document_id': 7},
          {'file_path': '/test8', 'document_id': 9},
          {'file_path': '/test9', 'document_id': 4}]

    for idx, el in enumerate(md):
        el['internal_id'] = idx

    dbi = DocumentIndex.from_list(md)
    dbi._make_relative_paths()
    query = pd.DataFrame([{'internal_id': 1},
                          {'internal_id': 2},
                          {'internal_id': 1}])
    sres = dbi.search(query)
    assert_equal(sres.internal_id.values, [1, 2, 1])
    assert_array_equal(sorted(sres.columns),
                       sorted(['internal_id', 'file_path', 'document_id']))

    # make sure we use internal id first
    query = pd.DataFrame([{'internal_id': 1, 'document_id': 2},
                          {'internal_id': 2, 'document_id': 2},
                          {'internal_id': 1, 'document_id': 2}])
    sres = dbi.search(query)
    assert_equal(sres.internal_id.values, [1, 2, 1])

    query = pd.DataFrame([{'document_id': 4},
                          {'document_id': 9},
                          {'document_id': 2}])
    sres = dbi.search(query)
    assert_equal(sres.internal_id.values, [4, 3, 0])


def test_search_document_rendition_id():
    md = [{'file_path': '/test',  'document_id': 0, 'rendition_id': 0},
          {'file_path': '/test2', 'document_id': 0, 'rendition_id': 1},
          {'file_path': '/test3', 'document_id': 1, 'rendition_id': 0},
          {'file_path': '/test8', 'document_id': 2, 'rendition_id': 0},
          {'file_path': '/test9', 'document_id': 3, 'rendition_id': 0}]

    for idx, el in enumerate(md):
        el['internal_id'] = idx

    # can always index with internal_id
    dbi = DocumentIndex.from_list(md)
    dbi._make_relative_paths()

    query = pd.DataFrame([{'internal_id': 1},
                          {'internal_id': 2},
                          {'internal_id': 1}])
    sres = dbi.search(query)
    assert_equal(sres.internal_id.values, [1, 2, 1])
    assert_array_equal(sorted(sres.columns),
                       sorted(['internal_id', 'file_path',
                               'document_id', 'rendition_id']))

    # the internal id is not sufficient to fully index documents in this case
    query = pd.DataFrame([{'document_id': 0},
                          {'document_id': 1},
                          {'document_id': 2}])
    with pytest.raises(ValueError):
        sres = dbi.search(query)

    query = pd.DataFrame([{'document_id': 0, 'rendition_id': 0},
                          {'document_id': 1, 'rendition_id': 0},
                          {'document_id': 2, 'rendition_id': 0}])

    sres = dbi.search(query)
    assert_equal(sres.internal_id.values, [0, 2, 3])


def test_bad_search_document_rendition_id():
    md = [{'file_path': '/test',  'document_id': 0, 'rendition_id': 0},
          {'file_path': '/test2', 'document_id': 0, 'rendition_id': 1},
          {'file_path': '/test3', 'document_id': 1, 'rendition_id': 0},
          {'file_path': '/test8', 'document_id': 2, 'rendition_id': 0},
          {'file_path': '/test9', 'document_id': 3, 'rendition_id': 0}]
    for idx, el in enumerate(md):
        el['internal_id'] = idx

    # can always index with internal_id
    dbi = DocumentIndex.from_list(md)
    query = pd.DataFrame([{'internal_id': 1},
                          {'internal_id': 2},
                          {'document_id': 1}])
    with pytest.raises(NotFound):
        sres = dbi.search(query)


def test_ingestion_pickling():
    from sklearn.externals import joblib
    db = DocumentIndex.from_folder(data_dir)
    fname = os.path.join(cache_dir, 'document_index')
    # check that db is picklable
    joblib.dump(db, fname)
    db2 = joblib.load(fname)
    os.remove(fname)


@pytest.mark.parametrize('n_fields', [1, 2, 3])
def test_ingestion_metadata(n_fields):
    metadata = []
    for idx, fname in enumerate(fnames_in_abs):
        el = {'file_path': fname}
        if n_fields >= 2:
            el['document_id'] = 'a' + str(idx + 100)
        if n_fields >= 3:
            el['rendition_id'] = 1
        metadata.append(el)

    dbi = DocumentIndex.from_list(metadata)
    dbi._make_relative_paths()
    data_dir_res, filenames, db = dbi.data_dir, dbi.filenames_, dbi.data

    if n_fields in [1, 2]:
        columns_ref = sorted(['file_path', 'document_id', 'internal_id'])
    elif n_fields == 3:
        columns_ref = sorted(['file_path', 'document_id', 'rendition_id',
                              'internal_id'])

    assert_array_equal(sorted(db.columns.values), columns_ref)
    assert_array_equal([os.path.normpath(os.path.join(data_dir_res, el))
                        for el in filenames],
                       [os.path.join(data_dir_res, el)
                        for el in db.file_path.values])


def test_infer_document_id_from_path():
    file_path = ['test/file1.txt', 'test3/file2.txt', 'test4/file4.txt']

    document_id = _infer_document_id_from_path(file_path)
    assert_array_equal(document_id, [1, 2, 4])

    file_path = ['test/file.txt', 'test3/file2.txt', 'test4/file4.txt']

    document_id = _infer_document_id_from_path(file_path)
    assert document_id is None

    file_path = ['test/file1.txt', 'test3/file1.txt', 'test4/file4.txt']

    document_id = _infer_document_id_from_path(file_path)
    assert document_id is None
