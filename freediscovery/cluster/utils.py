# Authors: Roman Yurchak
#
# License: BSD 3 clause

import numpy as np
from sklearn.utils.validation import check_array


def _binary_linkage2clusters(linkage, n_samples):
    """ Given a list of elements of size n_sample and a linkage matrix
    linking some of those samples, compute the cluster_id of each element

    Parameters
    ----------

    linkage : array (n_pairs, 2)
       arrays indicating binary links between elements
    n_samples : int
       total number of elements

    Returns
    -------
    labels : array (n_samples)
    """

    if (linkage > n_samples).any():
        raise ValueError
    if (linkage < 0).any():
        raise ValueError
    if n_samples < 0:
        raise ValueError

    dmap = {}
    idx = 0
    for a, b in linkage:
        if a in dmap:
            cid = dmap[a]
        elif b in dmap:
            cid = dmap[b]
        else:
            cid = idx
            idx += 1
        dmap[a] = cid
        dmap[b] = cid

    labels = np.zeros(n_samples, dtype=np.int)
    cid = 0
    for idx in range(n_samples):
        if idx in dmap:
            labels[idx] = n_samples + dmap[idx]
        else:
            labels[idx] = cid
            cid += 1
    _, labels_renamed = np.unique(labels, return_inverse=True)
    return labels_renamed


def _merge_clusters(X, rename=False):
    """
    Compute a union of all clusters

    Used to determine which cluster_id a document should belong to
    if at least one of it's lexicons suggest that it's a duplicate

    Approximate time complexity O(n_samples*n_features)

    Parameters
    ----------
     X: array (n_samples, n_features)
       input arrays with the cluster id's to merge
     rename : binary
       make sure the output array is between 0 and len(unique(cluster_id))

    Parameters
    ----------
      cluster_id: array (n_samples)
    """
    X = check_array(X, ensure_2d=True)

    n_samples, n_features = X.shape

    y = np.zeros(n_samples, dtype=X.dtype)

    out = {}

    for (i_idx, X_row) in enumerate(X):
        for j_idx, X_el in enumerate(X_row):
            if (j_idx, X_el) in out:
                res = out[(j_idx, X_el)]
                break
        else:
            res = X_row[0]  # use the 1st columnt index for this cluster id

        for (j_idx, X_el) in enumerate(X_row):
            out[(j_idx, X_el)] = res

        y[i_idx] = res
    if rename:
        _, labels_renamed = np.unique(y, return_inverse=True)
        return labels_renamed
    else:
        return y


def _dbscan_noisy2unique(labels_):
    """
    Take labels_ given by DBSCAN and replace each "noisy"
    point specified by -1, to a unique cluster id
    """
    labels_ = np.asarray(labels_).copy()
    mask = labels_ == -1
    indices = np.arange(mask.sum(), dtype=np.int)
    indices += labels_.max()+1
    labels_[mask] = indices
    return labels_


def _dbscan_unique2noisy(labels_):
    """
    Given an array of cluster_id, for each element that forms a cluster
    by itself, replace the corresponding cluster_id by -1.

    This is the iverse operation to _dbscan_noisy2unique
    """
    from collections import Counter
    labels_ = np.asarray(labels_).copy()
    cnt = Counter()
    for el in labels_:
        cnt[el] += 1

    labels_ndup_ = np.zeros(labels_.shape, dtype=labels_.dtype)
    for idx, el in enumerate(labels_):
        el_cnt = cnt[el]
        if el_cnt > 1:
            labels_ndup_[idx] = el
        elif el_cnt == 1:
            labels_ndup_[idx] = -1
        else:
            raise ValueError('This should not be possible!')
    return labels_ndup_


def centroid_similarity(X, internal_ids, nn_metric='cosine'):
    """ Given a list of documents in a cluster, compute the cluster centroid,
    intertia and individual distances

    Parameters
    ----------
    internal_ids : list
      a list of internal ids
    nn_metric : str
      a rescaling of the metric if needed
    """
    from ..metrics import _scale_cosine_similarity
    from sklearn.metrics.pairwise import pairwise_distances

    X_sl = X[internal_ids, :]
    centroid = X_sl.mean(axis=0)

    if centroid.ndim == 1:
        centroid = centroid[None, :]

    S_cos = 1 - pairwise_distances(X_sl, centroid, metric='cosine')
    S_sim = _scale_cosine_similarity(S_cos, metric=nn_metric)
    S_sim_mean = np.mean(S_sim)
    return float(S_sim_mean), S_sim[:, 0]
