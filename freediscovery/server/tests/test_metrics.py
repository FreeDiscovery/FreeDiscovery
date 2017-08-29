# -*- coding: utf-8 -*-

import pytest
import itertools
from numpy.testing import assert_almost_equal

from .. import fd_app

from .base import parse_res, V01, app, app_notest


@pytest.mark.parametrize('metrics',
                         itertools.combinations(['precision', 'recall', 'f1',
                                                 'roc_auc', 'average_precision'], 3))
def test_categorization_metrics(app, metrics):
    metrics = list(metrics)
    url = V01 + '/metrics/categorization'
    labels = ['negative', 'positive']


    y_true = [{'category': labels[key],
               'document_id': idx} for idx, key in\
                      enumerate([0, 0, 0, 1, 1, 0, 1, 0, 1])]
    y_pred = [{'scores': [{'category': labels[key],
                           'score': 1.}],
               'document_id': idx} for idx, key in\
                      enumerate([0, 0, 1, 1, 1, 0, 1, 1, 1])]

    pars = {'y_true': y_true, 'y_pred': y_pred, 'metrics': metrics}
    data = app.post_check(url, json=pars)
    assert sorted(data.keys()) == sorted(metrics)
    for key in metrics:
        assert data[key] > 0.5
        assert data[key] <= 1.0



@pytest.mark.parametrize('metrics',
                         itertools.combinations(['adjusted_rand', 'adjusted_mutual_info', 'v_measure'], 2))
def test_clustering_metrics(app, metrics):
    metrics = list(metrics)
    url = V01 + '/metrics/clustering'
    labels_true = [0, 0, 1, 2]
    labels_pred = [0, 0, 1, 1]

    pars = {'labels_true': labels_true, 'labels_pred': labels_pred, 'metrics': metrics}
    data = app.post_check(url, json=pars)

    assert sorted(data.keys()) == sorted(metrics)
    if 'adjusted_rand' in metrics:
        assert_almost_equal(data['adjusted_rand'], 0.5714, decimal=4)
    if 'adjusted_mutual_info' in metrics:
        assert_almost_equal(data['adjusted_mutual_info'], 0.4)
    if 'v_measure' in metrics:
        assert_almost_equal(data['v_measure'], 0.8)


@pytest.mark.parametrize('metrics',
                         itertools.combinations(['ratio_duplicates', 'f1_same_duplicates', 'mean_duplicates_count'], 2))
def test_dupdetection_metrics(app, metrics):
    metrics = list(metrics)
    url = V01 + '/metrics/duplicate-detection'
    labels_true = [0, 1, 1, 2, 3, 2]
    labels_pred = [0, 1, 3, 2, 5, 2]

    pars = {'labels_true': labels_true, 'labels_pred': labels_pred, 'metrics': metrics}
    data = app.post_check(url, json=pars)
    assert sorted(data.keys()) == sorted(metrics)
    if 'ratio_duplicates' in metrics:
        assert_almost_equal(data['ratio_duplicates'], 0.5)
    if 'f1_same_duplicates' in metrics:
        assert_almost_equal(data['f1_same_duplicates'], 0.667, decimal=3)
    if 'mean_duplicates_count' in metrics:
        assert_almost_equal(data['mean_duplicates_count'], 0.75)
