# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
from sklearn.externals import joblib

from .exceptions import ModelNotFound
from .exceptions import (DatasetNotFound, InitException, NotFound, WrongParameter)


class RankerMixin(object):
    """Mixin class for all ranking estimators in FreeDiscovery.
    A ranker is a binary classifier without a decision threshold.
    """
    _estimator_type = "classifier"  # so that thing would still work with scikit learn

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


class _BaseTextTransformer(object):
    """Base text transformer

    Parameters
    ----------
    cache_dir : str, default='/tmp/'
        directory where to save temporary and regression files
    dsid : str
        load an exising dataset
    verbose : bool
        pring progress messages
    """

    def __init__(self, cache_dir='/tmp/', dsid=None, verbose=False):
        self.data_dir = None
        self.verbose = verbose
        if cache_dir is not None:
            self.cache_dir = cache_dir = os.path.join(cache_dir, 'ediscovery_cache')
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
        else:
            self.cache_dir = None
        self.dsid = dsid
        if dsid is not None:
            dsid_dir = os.path.join(self.cache_dir, dsid)
            if not os.path.exists(dsid_dir):
                raise DatasetNotFound()
            pars = self._load_pars()
        else:
            dsid_dir = None
            pars = None
        self.dsid_dir = dsid_dir
        self._pars = pars

    @staticmethod
    def _list_filenames(data_dir, dir_pattern, file_pattern):
        import re
        # parse all files in the folder
        filenames = []
        for root, subdirs, files in os.walk(data_dir):
            #print(root, dir_pattern)
            if re.match(dir_pattern, root):
                for fname in files:
                    if re.match(file_pattern, fname):
                        filenames.append(os.path.normpath(os.path.join(root, fname)))

        # make sure that sorting order is deterministic
        return sorted(filenames)

    def delete(self):
        """ Delete the current dataset """
        import shutil
        shutil.rmtree(self.dsid_dir, ignore_errors=True)

    def __contains__(self, dsid):
        """ This is a somewhat non standard call that checks if a dsid
        exist on disk (in general)"""
        dsid_dir = os.path.join(self.cache_dir, dsid)
        return os.path.exists(dsid_dir)

    def __getitem__(self, index):
        return np.asarray(self._pars['filenames'])[index]

    def load(self, dsid):
        """ Load a computed features from disk"""
        if self.cache_dir is None:
            raise InitException('cache_dir is None: cannot load from cache!')
        dsid_dir = os.path.join(self.cache_dir, dsid)
        if not os.path.exists(dsid_dir):
            raise DatasetNotFound('dsid not found!')
        pars = joblib.load(os.path.join(dsid_dir, 'pars'))
        fset_new = joblib.load(os.path.join(dsid_dir, 'features'))
        return pars['filenames'], fset_new

    def get_params(self):
        """ Get the vectorizer parameters """
        return self._pars

    def _load_pars(self):
        """ Load parameters from disk"""
        dsid = self.dsid
        if self.cache_dir is None:
            raise InitException('cache_dir is None: cannot load from cache!')
        dsid_dir = os.path.join(self.cache_dir, dsid)
        if not os.path.exists(dsid_dir):
            raise DatasetNotFound('dsid {} not found!'.format(dsid))
        pars = joblib.load(os.path.join(dsid_dir, 'pars'))
        return pars

    def search(self, filenames):
        """ Return the document ids that correspond to the provided filenames.

        Parameters
        ----------
        filenames : list[str]
            list of filenames (relatives to the data_dir)

        Returns
        -------
        indices : array[int]
            corresponding list of document id (order is not preserved)
        """
        filenames_all = self._pars['filenames']
        # calculate the indices of the intersection of filenames with filenames_all
        ind_dict = dict((k,i) for i,k in enumerate(filenames_all))
        indices = [ ind_dict[x] for x in filenames]
        return np.array(indices)

    @property
    def n_samples_(self):
        """ Number of documents in the dataset """
        return len(self._pars['filenames'])

    def list_datasets(self):
        """ List all datasets in the working directory """
        import traceback
        out = []
        for dsid in os.listdir(self.cache_dir):
            row = {"id": dsid}
            self.dsid = dsid
            try:
                pars = self._load_pars()
            except:
                print(pars.keys())
                traceback.print_exc()
                continue

            if pars['type'] != type(self).__name__:
                continue

            try:
                for key in self._PARS_SHORT:
                    row[key] = pars[key]
                out.append(row)
            except Exception:
                print(pars.keys())
                traceback.print_exc()

        return out


class BaseEstimator(object):
    def get_path(self, mid):
        dsid = self.get_dsid(self.fe.cache_dir, mid)
        return os.path.join(self.fe.cache_dir, dsid, self._DIRREF, mid)

    def get_dsid(self, cache_dir, mid):
        if 'ediscovery_cache' not in cache_dir:  # not very pretty
            cache_dir = os.path.join(cache_dir, "ediscovery_cache")
        for dsid in os.listdir(cache_dir):
            mid_path = os.path.join(cache_dir, dsid, self._DIRREF)
            if not os.path.exists(mid_path):
                continue
            for mid_el in os.listdir(mid_path):
                if mid_el == mid:
                    return dsid
        raise ModelNotFound('Model id {} not found in {}/*/{}!'.format(mid, cache_dir, self._DIRREF))

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

    def list_models(self):
        out = []
        for mid in os.listdir(self.model_dir):
            try:
                pars = self.load_pars(mid)
                pars['id'] = mid
                out.append(pars)
            except:
                pass
        return out


