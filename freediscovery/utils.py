# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import os.path
from contextlib import contextmanager
import pandas as pd
import numpy as np
import uuid
try:  # sklearn v0.17
    from sklearn.exceptions import UndefinedMetricWarning
except ImportError:  # v0.18
    from sklearn.metrics.base import UndefinedMetricWarning

from .exceptions import (DatasetNotFound, ModelNotFound, InitException,
                            WrongParameter)

@contextmanager
def _silent(stream='stderr'):
    stderr = getattr(sys, stream)
    fh = open(os.devnull, 'w')
    sys.stderr = fh
    yield
    setattr(sys, stream, stderr)


INT_NAN = -99999


def categorization_score(idx_ref, Y_ref, idx, Y):
    """ Calculate the efficiency scores """
    # This function should be deprecated
    # An equivalent functionally should be achieved with a
    # more general freediscovery.metrics module
    import warnings
    from sklearn.metrics import (precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score)
    threshold = 0.0

    idx = np.asarray(idx, dtype='int')
    idx_ref = np.asarray(idx_ref, dtype='int')
    Y = np.asarray(Y)
    Y_ref = np.asarray(Y_ref)

    idx_out = np.intersect1d(idx_ref, idx)
    if not len(idx_out):
        return {"recall_score": -1, "precision_score": -1, 'f1': -1, 'auc_roc': -1,
                'average_precision': -1}

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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)

        m_recall_score = recall_score(Y_ref, Y_bin)
        m_precision_score = precision_score(Y_ref, Y_bin)
        m_f1_score = f1_score(Y_ref, Y_bin)
    if len(np.unique(Y_ref)) == 2:
        m_roc_auc = roc_auc_score(Y_ref, Y)
    else:
        m_roc_auc = np.nan # ROC not defined in this case
    m_average_precision = average_precision_score(Y_ref, Y)

    return {"recall": m_recall_score, "precision": m_precision_score,
            "f1": m_f1_score, 'roc_auc': m_roc_auc,
            'average_precision': m_average_precision }

def _rename_main_thread():
    """
    This aims to address the fact that joblib wrongly detects uWSGI workers
    as running in the non main thread even when they are not
    see https://github.com/joblib/joblib/issues/180
    """
    import threading
    if isinstance(threading.current_thread(), threading._MainThread) and \
                    threading.current_thread().name != 'MainThread':
        print('Warning: joblib: renaming current thread {} to "MainThread".'.format(threading.current_thread().name))
        threading.current_thread().name = 'MainThread'

def _count_duplicates(x):
    """Return y an array of the same shape as x with the number of
    duplicates for each element"""
    _, indices, counts = np.unique(x, return_counts=True, return_inverse=True)
    return counts[indices]

def generate_uuid(size=16):
    """
    Generate a unique id for the model
    """
    sl = slice(size)
    return uuid.uuid4().hex[sl] # a new random id


def setup_model(base_path):
    """
    Generate a unique model id and create the corresponding folder for storing results
    """
    mid = generate_uuid()
    mid_dir = os.path.join(base_path, mid)
    # hash collision; should not happen
    if os.path.exists(mid_dir):
        os.remove(mid_dir)  # removing the old folder nevertheless
    os.mkdir(mid_dir)
    return mid, mid_dir
