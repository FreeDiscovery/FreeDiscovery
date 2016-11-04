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

@contextmanager
def _silent(stream='stderr'):
    stderr = getattr(sys, stream)
    fh = open(os.devnull, 'w')
    sys.stderr = fh
    yield
    setattr(sys, stream, stderr)


def classification_score(X_ref, Y_ref, X, Y):
    """ Calculate the efficiency scores """
    import warnings
    from sklearn.metrics import (precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score)
    from sklearn.metrics.base import UndefinedMetricWarning
    threshold = 0.0

    X_ref = np.asarray(X_ref)
    X = np.asarray(X)

    d_pred = pd.DataFrame({'is_relevant_p': Y > threshold, 'is_relevant_score': Y}, index=X)
    d_ref = pd.DataFrame({'is_relevant': Y_ref}, index=X_ref)
    d_out = pd.merge(d_ref, d_pred, how='inner', left_index=True, right_index=True)
    if d_out.shape[0] == 0:
        return {"recall_score": -1, "precision_score": -1, 'f1': -1, 'auc_roc': -1,
                'average_precision': -1}

    is_relevant = d_out.is_relevant.values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)

        m_recall_score = recall_score(is_relevant,
                                      d_out.is_relevant_p.values)
        m_precision_score = precision_score(d_out.is_relevant.values,
                                            d_out.is_relevant_p.values)
        m_f1_score = f1_score(is_relevant,
                              d_out.is_relevant_p.values)
    if len(np.unique(is_relevant)) == 2:
        m_roc_auc = roc_auc_score(is_relevant,
                                  d_out.is_relevant_score.values)
    else:
        m_roc_auc = np.nan # ROC not defined in this case
    m_average_precision = average_precision_score(
                              is_relevant,
                              d_out.is_relevant_score.values)

    return {"recall": m_recall_score, "precision": m_precision_score,
            "f1": m_f1_score, 'roc_auc': m_roc_auc,
            'average_precision': m_average_precision }


def filter_rel_nrel(self, relevant_filenames, non_relevant_filenames):
    filenames_all, fset_all = self.fe.load(self.dsid)  #, mmap_mode='r')
    idx_rel = self.fe.search(relevant_filenames)
    idx_nrel = self.fe.search(non_relevant_filenames)
    if idx_rel is None:
        raise ValueError('No relevant files found with the input provided: {} ...!'.format(relevant_filenames[:20]))
    if idx_nrel is None:
        raise ValueError('No not-relevant files found with the input provided!')
    d_rel = fset_all[idx_rel,:]
    #print(idx_rel)
    #print(type(fset_all.shape))
    d_nrel = fset_all[idx_nrel,:]
    return fset_all, idx_rel, idx_nrel, d_rel, d_nrel


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


def generate_uuid():
    """
    Generate a unique id for the model
    """
    return uuid.uuid4().hex # a new random id


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
