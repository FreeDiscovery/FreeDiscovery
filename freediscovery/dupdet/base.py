# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from sklearn.externals import joblib
import numpy as np

from ..base import BaseEstimator
from ..text import FeatureVectorizer
from ..utils import setup_model, _rename_main_thread
from ..exceptions import (DatasetNotFound, ModelNotFound, InitException,
                            WrongParameter)

class DuplicateDetection(BaseEstimator):

    _DIRREF = "dupdet"

    def __init__(self, cache_dir='/tmp/', dsid=None, mid=None):
        """
        Find near duplicates in a document collection.

        Currently supported backends are:
          - simhash-py
          - i-match

        Parameters
        ----------
          cache_dir : str
             directory where to save temporary and regression files
          dsid : str, optional
             dataset id
          mid : str, optional
             model id
        """

        if dsid is None and mid is not None:
            self.dsid = dsid =  self.get_dsid(cache_dir, mid)
            self.mid = mid
        elif dsid is not None:
            self.dsid  = dsid
            self.mid = None
        elif dsid is None and mid is None:
            raise ValueError

        self.fe = FeatureVectorizer(cache_dir=cache_dir, dsid=dsid)
        #if self.fe._pars['use_hashing']:
        #    raise NotImplementedError('Using dup detection with non-hashed features is not supported by FreeDiscovery!')

        self.model_dir = os.path.join(self.fe.cache_dir, dsid, self._DIRREF)

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        if mid is not None:
            self.model = self.load(mid)
            self._pars = self._load_pars()
        else:
            self.model = None
            self._pars = None

    def fit(self, method='simhash'):
        """
        Precompute all the required values for duplicate detection
        """

        pars = {'method': method}
        if method not in ['simhash', 'i-match']:
            raise WrongParameter('Dup. detection method {} not implemented!'.format(method))
        if method == 'simhash':
            from .simhash import SimhashDuplicates
            self.model = shash = SimhashDuplicates()
        else:
            self.model = None
        self._pars = pars
        mid, mid_dir = setup_model(self.model_dir)

        self.mid = mid


        X = joblib.load(os.path.join(self.fe.dsid_dir, 'features'))
        if method == 'simhash':
            shash.fit(X)

        self._fit_X = X
        
        joblib.dump(self.model, os.path.join(self.model_dir, mid,  'model'), compress=9)
        joblib.dump(pars, os.path.join(self.model_dir, mid,  'pars'), compress=9)


    def query(self, **args):
        """ Find all the nearests neighbours for the dataset

        Parameters
        ----------
        distance : int, default=2
            Maximum number of differnet bits in the simhash
        blocks : int or 'auto', default='auto'
                number of blocks into which the simhash is split
                when searching for duplicates,
                see  https://github.com/seomoz/simhash-py
        Returns
        -------
        cluster_id : array
            the exact duplicates (documents with the same simhash)
            are grouped by in cluster_id
        """


        if self._pars['method'] == 'simhash':
            from simhash import find_all  # TODO resolve reference
            from ..clustering.utils import (_binary_linkage2clusters, 
                                    _merge_clusters)

            shash = self.model

            _fit_shash, cluster_id_exactdup, matches = shash.query(**args)

            if matches.shape[0] > 0:
                matches_idx = np.zeros(matches.shape, dtype=np.int)
                for idx, val in np.nditer(matches):
                    matches_idx[idx] = shash.get_index_by_hash(val)
            else:
                matches_idx = matches
            # compute cluster_id for near duplicates
            cluster_id_dnup = _binary_linkage2clusters(matches_idx, len(_fit_shash))
            # merge near duplicates and exact duplicates clusters
            cluster_id = _merge_clusters(
                     np.concatenate((cluster_id_exactdup[:, None],
                                     cluster_id_dnup[:, None]), axis=1),
                     rename=True)
        elif self._pars['method'] == 'i-match':
            from .imatch import IMatchDuplicates
            if not hasattr(self, '_fit_X'):
                self._fit_X = joblib.load(os.path.join(self.fe.dsid_dir, 'features'))
            model = IMatchDuplicates(**args)
            model.fit(self._fit_X)
            cluster_id = model.labels_

        else:
            raise ValueError


        return cluster_id


    def _load_pars(self):
        """ Load parameters from cache specified by a mid """

        mid = self.mid

        mid_dir = os.path.join(self.model_dir, mid)
        if not os.path.exists(mid_dir):
            raise ValueError('Model id {} not found in the cache!'.format(mid))

        pars = joblib.load(os.path.join(mid_dir, 'pars'))

        return pars


    def load(self, mid):
        """ Load results from cache specified by a mid """

        mid_dir = os.path.join(self.model_dir, mid)
        if not os.path.exists(mid_dir):
            raise ValueError('Model id {} not found in the cache!'.format(mid))

        shash = joblib.load(os.path.join(mid_dir, 'model'))
        return shash
