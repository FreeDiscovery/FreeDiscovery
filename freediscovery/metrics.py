# Authors: Roman Yurchak
#
# License: BSD 3 clause

from __future__ import division

import numpy as np
from freediscovery.cluster.utils import _dbscan_unique2noisy

from sklearn.exceptions import UndefinedMetricWarning

# Information retrieval specific metrics
# See scikit learn metrics for more general ones


# Categorization Metrics


def recall_at_k_score(y_true, y_pred, k):
    """
    Recall after retrieving k documents from the collections

    Parameters
    ----------
    y_true : ndarray [n_samples]
      array of integer classes
    y_pred : ndarray [n_samples]
      array of float predicted scores
    k : {int, float}
      the threashold either float in [0., 1.] or int in [0, n_samples]

    Returns
    -------
    score : float
      the recall at k score
    """
    N_docs = len(y_true)
    if isinstance(k, float):
        k = int(N_docs*k)
    elif isinstance(k, int):
        pass
    else:
        raise TypeError('Provided k with type {} must be int or float'
                        .format(type(k)))

    if len(y_true) != len(y_pred):
        raise ValueError('len(y_true)={} != len(y_pred)={}'
                         .format(len(y_true) != len(y_pred)))

    index_sorted = np.argsort(y_pred)[::-1]
    recall_curve = y_true[index_sorted].cumsum()/y_true.sum()
    # make sure the element 0 is always 0
    recall_curve = np.hstack((np.zeros(1), recall_curve))
    return recall_curve[k]


# Clustering Metrics
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
    from freediscovery.utils import _count_duplicates
    x_count = _count_duplicates(x)
    y_count = _count_duplicates(y)
    mask = (x_count > 1) | (y_count > 1)  # select only duplicates
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


def jaccard2cosine_similarity(s_jac):
    """ Given a cosine similarity on L2 normalized data,
    compute the jaccard similarity

    Parameters
    ----------
    s_jac : {float, ndarray}
      the cosine similarity

    Returns
    -------
    s_cos : {float, ndarray}
      the Jaccard similarity
    """
    return 2*s_jac / (1 + s_jac)


def _normalize_similarity(x, metric='cosine', inverse=False):
    """Given a similarity score, normalize it to the
    [0, 1] range

    Parameters
    ----------
    x : {float, ndarray}
      the similarity score
    metric : str
      the metric used (one of 'cosine', 'jaccard')
    inverse : bool
      perform the inverse de-normalization operation
    """
    if metric == 'cosine':
        # cosine similarity can be in the [-1, 1] range
        if not inverse:
            return (x + 1)/2
        else:
            return 2*x - 1
    elif metric == 'jaccard':
        # jaccard similarity could potenitally be in the [-1/3, 1] range
        # when using the cosine2jaccard_similarity function
        if not inverse:
            return (3*x + 1)/4.
        else:
            return (4*x - 1)/3.
    else:
        raise ValueError


def _scale_cosine_similarity(x, metric='cosine', inverse=False):
    """ Given a cosine similarity on L2 normalized data,
    appriximately convert it to Jaccard similarity, and/or
    normalize it to the [0, 1] interval

    Parameters
    ----------
    x : {float, ndarray}
      the cosine similarity value
    metric : str
      the conversion to apply one of ['cosine', 'jaccard']
    inverse : bool
      perform the inverse de-normalization operation
    """
    valid_metrics = ['cosine', 'jaccard', 'cosine_norm', 'jaccard_norm',
                     'cosine-positive']
    if metric not in valid_metrics:
        raise ValueError('metric {} not supported, must be in {}'
                         .format(metric, valid_metrics))
    if metric == 'cosine':
        return x
    elif metric == 'cosine-positive':
        if isinstance(x, (int, float)):
            return max(x, 0.0)
        else:
            return np.fmax(x, 0.0)

    if metric.startswith('jaccard'):
        if not inverse:
            x = cosine2jaccard_similarity(x)
        else:
            x = jaccard2cosine_similarity(x)

    if metric.endswith('norm'):
        x = _normalize_similarity(x, metric=metric.split('_')[0],
                                  inverse=inverse)

    return x


def categorization_score(idx_ref, Y_ref, idx, Y):
    """ Calculate the efficiency scores """
    # This function should be deprecated
    # An equivalent functionally should be achieved with a
    # more general freediscovery.metrics module
    import warnings
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.metrics import roc_auc_score, average_precision_score
    threshold = 0.0

    idx = np.asarray(idx, dtype='int')
    idx_ref = np.asarray(idx_ref, dtype='int')
    Y = np.asarray(Y)
    Y_ref = np.asarray(Y_ref)

    idx_out = np.intersect1d(idx_ref, idx)
    if not len(idx_out):
        return {"recall_score": -1, "precision_score": -1, 'f1': -1,
                'auc_roc': -1, 'average_precision': -1}

    # sort values by index
    order_ref = idx_ref.argsort()
    idx_ref = idx_ref[order_ref]
    Y_ref = Y_ref[order_ref]

    order = idx.argsort()
    idx = idx[order]
    Y = Y[order]

    # find indices that are in both the reference and the test dataset
    mask_ref = np.in1d(idx_ref, idx_out)
    mask = np.in1d(idx, idx_out)

    Y_ref = Y_ref[mask_ref]
    Y = Y[mask]
    Y_bin = (Y > threshold)

    n_classes = len(np.unique(Y_ref))

    if n_classes != 2:
        opts = {"average": 'micro'}
    else:
        opts = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)

        m_recall_score = recall_score(Y_ref, Y_bin, **opts)
        m_precision_score = precision_score(Y_ref, Y_bin, **opts)
        m_f1_score = f1_score(Y_ref, Y_bin, **opts)
    if n_classes == 2:
        m_roc_auc = roc_auc_score(Y_ref, Y)
        m_average_precision = average_precision_score(Y_ref, Y)
        m_recall_at_20p = recall_at_k_score(Y_ref, Y, k=0.2)
    else:
        # not defined for non binary categorization
        m_roc_auc = np.nan
        m_average_precision = np.nan
        m_recall_at_20p = np.nan

    return {"recall": m_recall_score, "precision": m_precision_score,
            "f1": m_f1_score, 'roc_auc': m_roc_auc,
            'average_precision': m_average_precision,
            'recall_at_20p': m_recall_at_20p}
