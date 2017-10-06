# Authors: Roman Yurchak
#
# License: BSD 3 clause

import os
import sys
import shutil
from contextlib import contextmanager
import numpy as np
import uuid
from pkg_resources import parse_version
import sklearn
from .exceptions import WrongParameter

# this is wrong, and should be eventually replaced
INT_NAN = -99999

sklearn_version = parse_version(sklearn.__version__)


@contextmanager
def _silent(stream='stderr'):
    stderr = getattr(sys, stream)
    fh = open(os.devnull, 'w')
    sys.stderr = fh
    yield
    setattr(sys, stream, stderr)


def _rename_main_thread():
    """
    This aims to address the fact that joblib wrongly detects uWSGI workers
    as running in the non main thread even when they are not
    see https://github.com/joblib/joblib/issues/180
    """
    import threading
    if isinstance(threading.current_thread(), threading._MainThread) and \
       threading.current_thread().name != 'MainThread':
        print('Warning: joblib: renaming current thread {} to "MainThread".'
              .format(threading.current_thread().name))
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
    return uuid.uuid4().hex[sl]  # a new random id


def setup_model(base_path, mid=None, mode='w'):
    """
    Generate a unique model id and create the corresponding folder
    for storing results
    """
    if mode not in ['w', 'fw']:
        raise WrongParameter(('Cannot setup model {} in mode={} '
                              'must be one of "w", "fw"!')
                             .format(base_path, mode))
    if mid is None:
        mid = generate_uuid()

    if not base_path.exists():
        base_path.mkdir()

    mid_dir = base_path / mid
    if mid_dir.exists():
        if mode == 'w':
            raise WrongParameter(('model mid={} already exists in {} .'
                                  'Use mode="fw" to overwrite!')
                                 .format(mid, base_path))
        shutil.rmtree(str(mid_dir))  # removing the old folder nevertheless
    mid_dir.mkdir()
    return mid, mid_dir


def _docstring_description(docstring):
    """ Given a function docstring, return only the text prior
    to the "Parameters" section"""

    res = []
    for line in docstring.splitlines():
        if line.strip() == 'Parameters':
            break
        res.append(line)
    return '\n'.join(res)


def _query_features(vect, X, indices, n_top_words=10, remove_stop_words=False):
    """ Query the features with most weight

    Parameters
    ----------
    vect : TfidfVectorizer
       the vectorizer object
    X : ndarray
       the document term tfidf array
    indices : list or ndarray
      indices for the subcluster
    n_top_words : int
      the number of workds to return
    remove_stop_words : bool
      remove stop words
    """
    from .cluster.base import select_top_words

    # this should raise a warning when used with wrong weights
    X = X[indices]

    centroid = X.sum(axis=0).view(type=np.ndarray)[0] / len(indices)
    order_centroid = centroid.argsort()[::-1]
    terms = vect.get_feature_names()

    out = []
    for ridx, idx in enumerate(order_centroid):
        if len(out) >= n_top_words:
            break
        if remove_stop_words:
            out += select_top_words([terms[idx]])
        else:
            out.append(terms[idx])
    return out


def _type_mapper(mtype):
    mapper = {'unicode': 'str', 'long': 'str'}
    if mtype in mapper:
        return mapper[mtype]
    else:
        return mtype


def dict2type(d, collapse_lists=False, max_depth=10):
    """Recursively walk though the object
    and replace all dict values by their type

    Parameters
    ----------
    collapse_lists : bool
      collapse a list to a single element
    max_depth : bool
      maximum depth on which the typing would be computed
    """
    if max_depth == 0:
        return _type_mapper(type(d).__name__)

    if isinstance(d, dict):
        res = {}
        for key, val in d.items():
            res[key] = dict2type(val, collapse_lists, max_depth - 1)
        return res
    elif isinstance(d, list):
        res = [dict2type(el, collapse_lists, max_depth - 1) for el in d]
        if collapse_lists:
            res = list(set(res))
        return res
    else:
        return _type_mapper(type(d).__name__)


def sdict_keys(x):
    """Sorted dictionary keys of x"""
    return list(sorted(x.keys()))


def _mean_csr_nonzero_axis1(X):
    """ Compute the mean of non zero elements a csr array along axis=1
    """

    sums = X.sum(axis=1).A1
    counts = np.diff(X.indptr)
    # prevent 0 / 0 division (which should yield 0 here)
    counts[counts == 0] = 1
    return sums / counts


def _paginate(df, batch_id, batch_size):
    """ Given a dataframe df, return a batch of rows
    specified by `batch_id` and `batch_size`

    Parameters
    ----------
    df : pd.DataFrame
       the input dataframe
    batch_id : int
       the batch_id element
    batch_size : int
       the batch size

    Returns
    -------
    df_out : pd.DataFrame
       sliced dataframe
    md : dict
       a dictionationary with pagination information
    """
    n_samples = df.shape[0]
    if batch_id >= 0:
        mslice = slice(batch_id*batch_size,
                       min((batch_id + 1)*batch_size, n_samples))
        df_out = df.iloc[mslice, :]
    else:
        df_out = df

    pagination = {'batch_id': batch_id,
                  'current_response_count': df_out.shape[0],
                  'total_response_count': n_samples}
    if batch_id < 0:
        pagination['batch_id_last'] = batch_id
    else:
        pagination['batch_id_last'] = n_samples // batch_size

    return df_out, pagination


def _split_path(path):
    """ A helper function that splits the path into a list

    Parameters
    ----------
    path : str
      path to split
    """
    path = os.path.normpath(path)

    head, tail = os.path.split(path)

    if not head:
        return [tail]
    elif head == path:
        if tail:
            return [head, tail]
        else:
            return [head]
    else:
        if not tail:
            return _split_path(head)
        else:
            return _split_path(head) + [tail]
