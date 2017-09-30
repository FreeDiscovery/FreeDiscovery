# -*- coding: utf-8 -*-

import pytest
from unittest import SkipTest
from numpy.testing import assert_equal, assert_almost_equal

from ...utils import _silent, dict2type, sdict_keys
from .base import parse_res, V01, app, app_notest, get_features_cached, get_features_lsi


#=============================================================================#
#
#                     Exception handling
#
#=============================================================================#


@pytest.mark.parametrize("method", ['feature-extraction', 'categorization', 'lsi', 'clustering'])
def test_get_model_404(app_notest, method):
    method = V01 + "/{}/DOES_NOT_EXISTS".format(method)
    with _silent('stderr'):
        res = app_notest.get(method)

    data = parse_res(res)
    assert res.status_code in [500, 404]  # depends on the url
    #assert '500' in data['message']

    assert sorted(data.keys()) == sorted(['messages'])


@pytest.mark.parametrize("method", ['feature-extraction', 'categorization', 'lsi', 'clustering'])
def test_get_model_train_404(app_notest, method):
    method = V01 + "/{}/DOES_NOT_EXISTS/DOES_NOT_EXIST".format(method)
    with _silent('stderr'):
        res = app_notest.get(method)

    assert res.status_code in [404, 500]


@pytest.mark.parametrize("method", ['feature-extraction', 'categorization', 'lsi', 'clustering'])
def test_get_model_predict_404(app_notest, method):

    method = V01 + "/{0}/DOES_NOT_EXISTS/DOES_NOT_EXIST/predict".format(method)
    with _silent('stderr'):
        res = app_notest.get(method)

    assert res.status_code == 404

    method = V01 + "/{0}/DOES_NOT_EXISTS/DOES_NOT_EXIST/test".format(method)
    with _silent('stderr'):
        res = app_notest.post(method)

    assert res.status_code == 404


def test_exception_handling(app_notest):
    dsid, pars, _ = get_features_cached(app_notest)

    method = V01 + "/categorization/"
    with _silent('stderr'):
        res = app_notest.post(method,
                        json={
                              'parent_id': dsid,
                              'index': [0, 0, 0],       # just something wrong
                              'y': ['ds', 'dsd', 'dsd'],
                              'method': "LogisticRegression",
                              'cv': 0,
                              })
    data = parse_res(res)
    assert res.status_code in [500, 422]
    assert sorted(data.keys()) == ['messages']
