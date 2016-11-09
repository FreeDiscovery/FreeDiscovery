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
    x_dup[x_dup > -1] = 1 # not a duplicate
    x_dup[x_dup == -1] = 0 # not a duplicate
    y_dup = _dbscan_unique2noisy(y)
    y_dup[y_dup > -1] = 1 # not a duplicate
    y_dup[y_dup == -1] = 0 # not a duplicate

    x_dup = np.abs(x_dup)
    y_dup = np.abs(y_dup)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        score = f1_score(x_dup, y_dup)

    return score
