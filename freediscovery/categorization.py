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
from .utils import setup_model, _rename_main_thread
from .exceptions import (ModelNotFound, WrongParameter, NotImplementedFD, OptionalDependencyMissing)


def _zip_relevant(relevant_id, non_relevant_id):
    """ Take a list of relevant and non relevant documents id and return
    an array of indices and prediction values """
    idx_id = np.hstack((np.asarray(relevant_id), np.asarray(non_relevant_id)))
    y = np.concatenate((np.ones((len(relevant_id))),
                        np.zeros((len(non_relevant_id))))).astype(np.int)
    return idx_id, y

def _unzip_relevant(idx_id, y):
    """ Take an array of indices and prediction values and return
    a list of relevant and non relevant documents id
    """
    mask = np.asarray(y) > 0.5
    idx_id = np.asarray(idx_id, dtype='int')
    return idx_id[mask], idx_id[~mask]


class Categorizer(BaseEstimator):
    """ Document categorization model

    The option `use_hashing=True` must be set for the feature extraction.
    Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

    Parameters
    ----------
    cache_dir : str
      folder where the model will be saved
    dsid : str, optional
      dataset id
    mid : str, optional
      model id
    cv_scoring : str, optional, default='roc_auc'
      score that is used for Cross Validation, cf. sklearn
    cv_n_folds : str, optional
      number of K-folds used for Cross Validation
    """

    _DIRREF = "models"

    def __init__(self, cache_dir='/tmp/',  dsid=None, mid=None,
            cv_scoring='roc_auc', cv_n_folds=3):

        if dsid is None and mid is not None:
            self.dsid = dsid =  self.get_dsid(cache_dir, mid)
            self.mid = mid
        elif dsid is not None:
            self.dsid  = dsid
            self.mid = None
        elif dsid is None and mid is None:
            raise WrongParameter('dsid and mid')

        self.fe = FeatureVectorizer(cache_dir=cache_dir, dsid=dsid)

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

        self.cv_scoring = cv_scoring
        self.cv_n_folds = cv_n_folds


    @staticmethod
    def _build_estimator(Y_train, method, cv, cv_scoring, cv_n_folds, **options):
        if cv:
            #from sklearn.cross_validation import StratifiedKFold
            #cv_obj = StratifiedKFold(n_splits=cv_n_folds, shuffle=False)
            cv_obj = cv_n_folds  # temporary hack (due to piclking issues otherwise, this needs to be fixed)
        else:
            cv_obj = None

        _rename_main_thread()

        if method == 'LinearSVC':
            from sklearn.svm import LinearSVC
            if cv is None:
                cmod = LinearSVC(**options)
            else:
                try:
                    from freediscovery_extra import make_linearsvc_cv_model
                except ImportError:
                    raise OptionalDependencyMissing('freediscovery_extra')
                cmod = make_linearsvc_cv_model(cv_obj, cv_scoring, **options)
        elif method == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            if cv is None:
                cmod = LogisticRegression(**options)
            else:
                try:
                    from freediscovery_extra import make_logregr_cv_model
                except ImportError:
                    raise OptionalDependencyMissing('freediscovery_extra')
                cmod = make_logregr_cv_model(cv_obj, cv_scoring, **options)
        elif method == 'xgboost':
            try:
                import xgboost as xgb
            except ImportError:
                raise OptionalDependencyMissing('xgboost')
            if cv is None:
                try:
                    from freediscovery_extra import make_xgboost_model
                except ImportError:
                    raise OptionalDependencyMissing('freediscovery_extra')
                cmod = make_xgboost_model(cv_obj, cv_scoring, **options)
            else:
                try:
                    from freediscovery_extra import make_xgboost_cv_model
                except ImportError:
                    raise OptionalDependencyMissing('freediscovery_extra')
                cmod = make_xgboost_cv_model(cv, cv_obj, cv_scoring, **options)
        elif method == 'MLPClassifier':
            if cv is not None:
                raise NotImplementedFD('CV not supported with MLPClassifier')
            from sklearn.neural_network import MLPClassifier
            cmod = MLPClassifier(solver='adam', hidden_layer_sizes=10,
                                 max_iter=200, activation='identity', verbose=0)
        else:
            raise WrongParameter('Method {} not implemented!'.format(method))
        return cmod

    def train(self, index, y, method='LinearSVC', cv=None):
        """
        Train the categorization model

        Parameters
        ----------
        index : array-like, shape (n_samples)
           document indices of the training set
        y : array-like, shape (n_samples)
           target binary class relative to index
        method : str
           the ML algorithm to use (one of "LogisticRegression", "LinearSVC", 'xgboost')
        cv : str
           use cross-validation
        Returns
        -------
        cmod : sklearn.BaseEstimator
           the scikit learn classifier object
        Y_train : array-like, shape (n_samples)
           training predictions
        """

        valid_methods = ["LinearSVC", "LogisticRegression", "xgboost"]

        if method in ['ensemble-stacking', 'MLPClassifier']:
            raise WrongParameter('method={} is implemented but not production ready. It was disabled for now.'.format(method))

        if method not in valid_methods:
            raise WrongParameter('method={} is not supported, should be one of {}'.format(
                method, valid_methods)) 

        if cv not in [None, 'fast', 'full']:
            raise WrongParameter('cv')

        if method == 'ensemble-stacking':
            if cv is not None:
                raise WrongParameter('CV with ensemble stacking is not supported!')

        _, d_all = self.fe.load(self.dsid)  #, mmap_mode='r')

        X_train = d_all[index, :]

        Y_train = y

        if method != 'ensemble-stacking':
            cmod = self._build_estimator(Y_train, method, cv, self.cv_scoring, self.cv_n_folds)
        else:
            from freediscovery.private import _EnsembleStacking

            cmod_logregr = self._build_estimator(Y_train, 'LogisticRegression', 'full',
                                             self.cv_scoring, self.cv_n_folds)
            cmod_svm = self._build_estimator(Y_train, 'LinearSVC', 'full',
                                             self.cv_scoring, self.cv_n_folds)
            cmod_xgboost = self._build_estimator(Y_train, 'xgboost', None,
                                             self.cv_scoring, self.cv_n_folds)
            cmod_xgboost.transform = cmod_xgboost.predict
            cmod = _EnsembleStacking([('logregr', cmod_logregr),
                                      ('svm', cmod_svm),
                                      ('xgboost', cmod_xgboost)
                                      ])

        mid, mid_dir = setup_model(self.model_dir)

        if method == 'xgboost' and not cv:
            cmod.fit(X_train, Y_train, eval_metric='auc')
        else:
            cmod.fit(X_train, Y_train)

        joblib.dump(cmod, os.path.join(mid_dir, 'model'), compress=9)

        pars = {
            'method': method,
            'index': index,
            'y': y
            }
        pars['options'] = cmod.get_params()
        self._pars = pars
        joblib.dump(pars, os.path.join(mid_dir, 'pars'), compress=9)

        self.mid = mid
        self.cmod = cmod
        return cmod, Y_train

    def predict(self, chunk_size=5000):
        """
        Predict the relevance using a previously trained model

        Parameters
        ----------
        chunck_size : int
           chunck size
        """

        if self.cmod is not None:
            cmod = self.cmod
        else:
            raise WrongParameter('The model must be trained first, or sid must be provided to load\
                    a previously trained model!')
        #else:
        #    mid_dir = os.path.join(self.model_dir, mid)
        #    if not os.path.exists(mid_dir):
        #        raise ModelNotFound('Model id {} not found in the cache!'.format(mid))

        #    cmod = joblib.load(os.path.join(mid_dir, 'model'))

        ds = joblib.load(os.path.join(self.fe.dsid_dir, 'features'))  #, mmap_mode='r')
        n_samples = ds.shape[0]

        def _predict_chunk(cmod, ds, k, chunk_size):
            n_samples = ds.shape[0]
            mslice = slice(k*chunk_size, min((k+1)*chunk_size, n_samples))
            ds_sl = ds[mslice, :]
            if hasattr(cmod, 'decision_function'):
                res = cmod.decision_function(ds_sl)
            else:  # gradient boosting define the decision function by analogy
                tmp = cmod.predict_proba(ds_sl)[:, 1]
                res = logit(tmp)
            return res

        res = []
        for k in range(n_samples//chunk_size + 1):
            pred = _predict_chunk(cmod, ds, k, chunk_size)
            res.append(pred)
        res = np.concatenate(res, axis=0)
        return res

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
