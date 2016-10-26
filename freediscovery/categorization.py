# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path

import numpy as np
import scipy
from scipy.special import logit

from sklearn.externals import joblib

from .text import FeatureVectorizer
from .base import BaseEstimator
from .utils import filter_rel_nrel, setup_model, _rename_main_thread
from .exceptions import (DatasetNotFound, ModelNotFound, InitException,
                            WrongParameter, NotImplementedFD, OptionalDependencyMissing)



class Categorizer(BaseEstimator):

    _DIRREF = "models"

    def __init__(self, cache_dir='/tmp/',  dsid=None, mid=None,
            cv_scoring='roc_auc', cv_n_folds=3):
        """ Document categrorization model

        Parameters
        ----------
          cache_dir : str
             folder where the model will be saved
          dsid : str, optional
             dataset id
          mid : str, optional
             model id
          cv_scoring: str, optional, default='roc_auc'
             score that is used for Cross Validation, cf. sklearn
          cv_n_folds: str, optional
             number of K-folds used for Cross Validation

        """

        if dsid is None and mid is not None:
            self.dsid = dsid =  self.get_dsid(cache_dir, mid)
            self.mid = mid
        elif dsid is not None:
            self.dsid  = dsid
            self.mid = None
        elif dsid is None and mid is None:
            raise WrongParameter('dsid and mid')

        self.fe = FeatureVectorizer(cache_dir=cache_dir, dsid=dsid)
        if not self.fe._pars['use_hashing']:
            raise NotImplementedFD('Using categorisation without hashed features is not supported by FreeDiscovery at the moment!')

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
            cv_obj = cv_n_folds # temporary hack (due to piclking issues otherwise, this needs to be fixed)
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



    def train(self, relevant_filenames, non_relevant_filenames, method='LinearSVC', cv=None, **options):
        """
        Train the categorization model

        Parameters
        ----------
           relevant_filenames: list
              a list of relevant documents filenames
           non_relevant_filenames: list
              a list of not relevant documents filenames
           method: str
              the ML algorithm to use (one of "LogisticRegression", "LinearSVC", 'xgboost')
           cv: str
              use cross-validation
           options: dict
               method specific ML agorithm options
        Returns
        -------
        dict
           a dictionary with the results
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

        d_all, _, _, d_rel, d_nrel = filter_rel_nrel(self, relevant_filenames,
                                                non_relevant_filenames)

        X_train = scipy.sparse.vstack((d_rel, d_nrel))
        X_train_str = np.hstack((np.asarray(relevant_filenames),
                                    np.asarray(non_relevant_filenames)))
        Y_train = np.concatenate((np.ones((d_rel.shape[0])),
                                np.zeros((d_nrel.shape[0]))), axis=0).astype(np.int)

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
            cmod = _EnsembleStacking([  ('logregr', cmod_logregr),
                                       ('svm', cmod_svm),
                                       ('xgboost', cmod_xgboost)
                                    ])


        mid, mid_dir = setup_model(self.model_dir)

        if method == 'xgboost' and not cv:
            cmod.fit(X_train, Y_train, eval_metric='auc')
        else:
            cmod.fit(X_train, Y_train)

        joblib.dump(cmod, os.path.join(mid_dir, 'model'), compress=9)

        pars = {'method': method, 'relevant_filenames': relevant_filenames,
                'non_relevant_filenames': non_relevant_filenames}
        pars['options'] = cmod.get_params()
        self._pars = pars
        joblib.dump(pars, os.path.join(mid_dir, 'pars'), compress=9)

        self.mid = mid
        self.cmod = cmod
        return cmod, X_train_str, Y_train


    def predict(self, chunk_size=5000):
        """
        Predict the relevance using a previously trained model

        Parameters
        ----------
          chunck_size: int
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

        ds = joblib.load(os.path.join(self.fe.dsid_dir, 'features'))#, mmap_mode='r')

        n_samples = ds.shape[0]

        def _predict_chunk(cmod, ds, k, chunk_size):
            n_samples = ds.shape[0]

            mslice = slice(k*chunk_size, min((k+1)*chunk_size, n_samples))

            ds_sl = ds[mslice, :]

            if hasattr(cmod, 'decision_function'):
                res = cmod.decision_function(ds_sl)
            else: # gradient boosting define the decision function by analogy

                tmp = cmod.predict_proba(ds_sl)[:,1]
                res = logit(tmp)
            return res

        res = []
        for k in range(n_samples//chunk_size + 1):
            pred = _predict_chunk(cmod, ds, k, chunk_size)
            res.append(pred)

        res = np.concatenate(res, axis=0)
        return res


    def get_params(self):
        """ Get model parameters """
        return self._pars


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
