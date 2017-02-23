# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from .cluster.utils import _dbscan_unique2noisy

# Information retrieval specific metrics
# See scikit learn metrics for more general ones

## Clustering Metrics

def ratio_duplicates_score(x, y):
    """
    Given cluster labels x and y, compute the relative error
    between the number of duplicates in x vs the one in y.
    """
    if x.shape != y.shape:
        raise ValueError
    N = len(x)
    N_dups_x = N - len(np.unique(x))
    N_dups_y = N - len(np.unique(y))

    return 1 - abs(N_dups_x - N_dups_y)/max(N_dups_x, N_dups_y)


def f1_same_duplicates_score(x, y):
    """
    Given cluster labels x and y, compute the f1 score
    that the same elements are marked as duplicates
    """
    import warnings
    from sklearn.metrics import f1_score
    from sklearn.metrics.base import UndefinedMetricWarning

    if x.shape != y.shape:
        raise ValueError
    x_dup = _dbscan_unique2noisy(x)
    x_dup[x_dup > -1] = 1  # duplicates
    x_dup[x_dup == -1] = 0  # not duplicates
    y_dup = _dbscan_unique2noisy(y)
    y_dup[y_dup > -1] = 1  # duplicates
    y_dup[y_dup == -1] = 0  # not duplicates

    x_dup = np.abs(x_dup)
    y_dup = np.abs(y_dup)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        score = f1_score(x_dup, y_dup)

    return score


def mean_duplicates_count_score(x, y):
    """ Score based on the number of duplicates for sample k,
    averaged over samples."""
    from .utils import _count_duplicates
    x_count = _count_duplicates(x)
    y_count = _count_duplicates(y)
    mask = (x_count > 1) | (y_count > 1) # select only duplicates
    x_count = x_count[mask]
    y_count = y_count[mask]

    score = 1 - np.abs(x_count - y_count)/np.maximum(x_count, y_count)
    return np.mean(score)


def seuclidean_dist2cosine_sim(x):
    """Given an squared euclidean distance on L2 normalized data,
    convert it to the cosine similarity

    Warning: this function would give completely wrong results if,
      * the euclidean distance is not squared
      * the data is initially not L2 normalized
    """
    return 1 - x/2.


def cosine2jaccard_similarity(s_cos):
    """ Given a cosine similarity on L2 normalized data,
    compute the jaccard similarity

    Parameters
    ----------
    s_cos : {float, ndarray}
      the cosine similarity

    Returns
    -------
    s_jac : {float, ndarray}
      the Jaccard similarity
    """
    return s_cos / (2 - s_cos)


def _normalize_similarity(x, metric='cosine'):
    """Given a similarity score, normalize it to the
    [0, 1] range

    Parameters
    ----------
    x : {float, ndarray}
      the similarity score
    metric : str
      the metric used (one of 'cosine', 'jaccard')
    """
    if metric == 'cosine':
        # cosine similarity can be in the [-1, 1] range
        return (x + 1)/2
    elif metric == 'jaccard':
        # jaccard similarity could potenitally be in the [-1/3, 1] range
        # when using the cosine2jaccard_similarity function
        return (3*x + 1)/4.
    else:
        raise ValueError

def _scale_cosine_similarity(x, metric='cosine'):
    """ Given a cosine similarity on L2 normalized data,
    optionally convert it to Jaccard similarity, and/or 
    normalize it to the [0, 1] interval

    Parameters
    ----------
    x : {float, ndarray}
      the cosine similarity value
    metric : str
      the conversion to apply one of ['cosine', 'jaccard', 'cosine_norm',
                                      'jaccard_norm']
    """
    valid_metrics = ['cosine', 'jaccard', 'cosine_norm', 'jaccard_norm']
    if metric not in valid_metrics:
        raise ValueError('metric {} not supported, must be in {}'.format(metric, valid_metrics))
    if metric == 'cosine':
        return x
    if metric.startswith('jaccard'):
        x = cosine2jaccard_similarity(x)

    if metric.endswith('norm'):
        x = _normalize_similarity(x, metric=metric.split('_')[0])

    return x



