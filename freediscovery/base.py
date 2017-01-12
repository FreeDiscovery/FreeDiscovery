# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
from sklearn.externals import joblib

from .text import FeatureVectorizer
from .exceptions import ModelNotFound
from .exceptions import (DatasetNotFound, InitException, NotFound, WrongParameter)


class RankerMixin(object):
    """Mixin class for all ranking estimators in FreeDiscovery.
    A ranker is a binary classifier without a decision threshold.
    """
    _estimator_type = "ranker"  # so that thing would still work with scikit learn

    def score(self, X, y, sample_weight=None):
        """Returns the ROC score of the prediction.
        Best possible score is 1.0 and the worst in 0.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples)
            True values for X.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.
        Returns
        -------
        score : float
            ROC score of self.decision_function(X) wrt. y.
        """

        from .metrics import roc_auc_score
        return roc_auc_score(y, self.decision_function(X), sample_weight=sample_weight,)


class _BaseWrapper(object):
    """Base class for wrappers in FreeDiscovery

    Parameters
    ----------
    cache_dir : str
      folder where the model will be saved
    dsid : str, optional
      dataset id
    mid : str, optional
      model id
    """
    def __init__(self, cache_dir='/tmp/', dsid=None, mid=None,
                 dataset_definition=FeatureVectorizer):

        if dsid is None and mid is not None:
            self.dsid = dsid = self.get_dsid(cache_dir, mid)
            self.mid = mid
        elif dsid is not None:
            self.dsid  = dsid
            self.mid = None
        elif dsid is None and mid is None:
            raise WrongParameter('dsid and mid')

        self.fe = dataset_definition(cache_dir=cache_dir, dsid=dsid)

        self.model_dir = os.path.join(self.fe.cache_dir, dsid, self._wrapper_type)

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        if self.mid is not None:
            pars = self._load_pars()
        else:
            pars = None
        self._pars = pars


    def get_path(self, mid):
        dsid = self.get_dsid(self.fe.cache_dir, mid)
        return os.path.join(self.fe.cache_dir, dsid, self._wrapper_type, mid)

    def get_dsid(self, cache_dir, mid):
        if 'ediscovery_cache' not in cache_dir:  # not very pretty
            cache_dir = os.path.join(cache_dir, "ediscovery_cache")
        for dsid in os.listdir(cache_dir):
            mid_path = os.path.join(cache_dir, dsid, self._wrapper_type)
            if not os.path.exists(mid_path):
                continue
            for mid_el in os.listdir(mid_path):
                if mid_el == mid:
                    return dsid
        raise ModelNotFound('Model id {} not found in {}/*/{}!'.format(mid, cache_dir, self._wrapper_type))

    def delete(self):
        """ Delete a trained model"""
        import shutil
        mid_dir = os.path.join(self.model_dir, self.mid)
        shutil.rmtree(mid_dir, ignore_errors=True)

    def __contains__(self, mid):
        mid_dir = os.path.join(self.model_dir, mid)
        return os.path.exists(mid_dir)

    def get_params(self):
        """ Get model parameters """
        return self._pars

    def _load_pars(self, mid=None):
        """Load model parameters from disk"""
        if mid is None:
            mid = self.mid
        model_path = os.path.join(self.model_dir, mid)
        if not os.path.exists(model_path):
            raise ValueError('Model id {} ({}) not found in the cache {}!'.format(
                             mid, self._wrapper_type, model_path))
        pars = joblib.load(os.path.join(model_path, 'pars'))
        pars['id'] = mid
        return pars

    def list_models(self):
        """ List existing models of this type """
        out = []
        if not os.path.exists(self.model_dir):
            return out
        for mid in os.listdir(self.model_dir):
            try:
                pars = self._load_pars(mid)
                pars['id'] = mid
                out.append(pars)
            except:
                raise
        return out


