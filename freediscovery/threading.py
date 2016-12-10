# -*- coding: utf-8 -*-

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
from .base import BaseEstimator
from .parsers import EmailParser
from .utils import setup_model, _rename_main_thread
from .exceptions import (ModelNotFound, WrongParameter, NotImplementedFD, OptionalDependencyMissing)

from jwzthreading import jwzthreading as jwzt


class EmailThreading(BaseEstimator):
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
               sort_by='message_idx', sort_missing=-1):
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
        sort_by_subject : str, default='message_idx'
           key used for sorting threads
        sort_missing : object, default=-1
           value used for sorting when the `sort_by_subject` key is missing
        """
        if index is None:
            index = np.arange(self.fe.n_samples_)

        _, d_all = self.fe.load(self.dsid)  #, mmap_mode='r')

        cmod = jwzt.thread(d_all, group_by_subject)

        cmod = jwzt.sort_threads(cmod, key=sort_by, missing=sort_missing)

        mid, mid_dir = setup_model(self.model_dir)

        joblib.dump(cmod, os.path.join(mid_dir, 'model'), compress=9)

        pars = {
            'index': index,
        }
        self._pars = pars
        joblib.dump(pars, os.path.join(mid_dir, 'pars'), compress=9)

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
        pars['options'] = cmod.get_params()
        return pars, cmod
