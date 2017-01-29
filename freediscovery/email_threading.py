# -*- coding  utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import scipy
from scipy.special import logit
from sklearn.externals import joblib

from .text import FeatureVectorizer
from .base import _BaseWrapper
from .parsers import EmailParser
from .utils import setup_model, INT_NAN
from .exceptions import (ModelNotFound, WrongParameter,
             NotImplementedFD, OptionalDependencyMissing)

from .externals import jwzthreading as jwzt


class _EmailThreadingWrapper(_BaseWrapper):
    """ JWZ Email threading class


    Parameters
    ----------
    cache_dir : str
      folder where the model will be saved
    parent_id : str, optional
      dataset id
    mid : str, optional
      model id
    """

    _wrapper_type = "threading"

    def __init__(self, cache_dir='/tmp/',  parent_id=None, mid=None,
                 decode_header=False):

        super(_EmailThreadingWrapper, self).__init__(cache_dir=cache_dir,
                                          parent_id=parent_id, mid=mid,
                                          dataset_definition=EmailParser,
                                          load_model=True)


    def thread(self, index=None, group_by_subject=False,
            sort_by_key='message_idx', sort_missing=-1):
        """
        Thread the emails

        Parameters
        ----------
        index : array-like, shape (n_samples)
           document indices of the training set

        Returns
        -------
        cmod : sklearn.BaseEstimator
           the scikit learn classifier object
        Y_train : array-like, shape (n_samples)
           training predictions
        group_by_subject : boolean, default=True
           group emails by subject

        Returns
        -------

        tree : array (N_samples)
           the id of the parent element in the tree
        root_idx : array (N_samples)
           the id of the root element in the tree
        """
        if index is None:
            index = np.arange(self.fe.n_samples_)

        _, d_all = self.fe.load()  #, mmap_mode='r')

        threads = jwzt.thread(d_all, group_by_subject)

        threads = [el.collapse_empty() for el in threads]

        threads = jwzt.sort_threads(threads, key=sort_by_key,
                                             missing=sort_missing)

        cmod = threads

        mid, mid_dir = setup_model(self.model_dir)


        pars = {
            'group_by_subject': group_by_subject
        }
        self._pars = pars
        joblib.dump(pars, os.path.join(mid_dir, 'pars'), compress=9)
        joblib.dump(cmod, os.path.join(mid_dir, 'model'), compress=9)


        self.mid = mid
        self.cmod = cmod
        return cmod

