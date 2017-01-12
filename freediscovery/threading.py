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

from jwzthreading import jwzthreading as jwzt


class EmailThreading(_BaseWrapper):
    """ JWZ Email threading class


    Parameters
    ----------
    cache_dir : str
      folder where the model will be saved
    dsid : str, optional
      dataset id
    mid : str, optional
      model id
    """

    _DIRREF = "threading"

    def __init__(self, cache_dir='/tmp/',  dsid=None, mid=None,
                 decode_header=False):

        if dsid is None and mid is not None:
            self.dsid = dsid =  self.get_dsid(cache_dir, mid)
            self.mid = mid
        elif dsid is not None:
            self.dsid  = dsid
            self.mid = None
        elif dsid is None and mid is None:
            raise WrongParameter('dsid and mid')

        self.fe = EmailParser(cache_dir=cache_dir, dsid=dsid)

        self.model_dir = os.path.join(self.fe.cache_dir, dsid, self._DIRREF)

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        if self.mid is not None:
            pars, cmod = self._load_pars()
        else:
            pars = None
            cmod = None
        self._pars = pars
        self.cmod = cmod


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

        _, d_all = self.fe.load(self.dsid)  #, mmap_mode='r')

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

    def _load_pars(self):
        """ Load the parameters specified by a mid """
        mid = self.mid
        mid_dir = os.path.join(self.model_dir, mid)
        if not os.path.exists(mid_dir):
            raise ModelNotFound('Model id {} not found in the cache!'.format(mid))
        pars = joblib.load(os.path.join(mid_dir, 'pars'))
        cmod = joblib.load(os.path.join(mid_dir, 'model'))
        return pars, cmod
