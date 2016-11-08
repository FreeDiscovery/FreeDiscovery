# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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

    Used to determine which cluster_id a document should belong to if at least one of it's
    lexicons suggest that it's a duplicate

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
            res = X_row[0] # use the 1st columnt index for this cluster id

        for (j_idx, X_el) in enumerate(X_row):
            out[(j_idx, X_el)] = res

        y[i_idx] = res
    if rename:
        _, labels_renamed = np.unique(y, return_inverse=True)
        return labels_renamed
    else:
        return y
