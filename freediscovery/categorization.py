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
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array


from .text import FeatureVectorizer
from .base import BaseEstimator, RankerMixin
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
    """Take an array of indices and prediction values and return
    a list of relevant and non relevant documents id

    Parameters
    ----------
    idx_id : ndarray[int] (n_samples)
        array of indices
    y : ndarray[float] (n_samples)
        target array
    """
    mask = np.asarray(y) > 0.5
    idx_id = np.asarray(idx_id, dtype='int')
    return idx_id[mask], idx_id[~mask]


class NearestNeighborRanker(BaseEstimator, RankerMixin):
    """A nearest neighbor ranker, behaves like
        * KNeigborsClassifier (supervised) when trained on both positive and negative samples
        * NearestNeighbors  (unsupervised) when trained on positive samples only

    Parameters
    ----------
    radius : float, optional (default = 1.0)
        Range of parameter space to use by default for :meth:`radius_neighbors`
        queries.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDtree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    """

    def __init__(self, n_neighbors=1, radius=1.0,
                 algorithm='ball_tree', leaf_size=30, n_jobs=1, **kwargs):

        # define nearest neighbors search objects for positive and negative samples
        self._mod_p = NearestNeighbors(n_neighbors=1,
                                       leaf_size=leaf_size,
                                       algorithm=algorithm,
                                       n_jobs=n_jobs,
                                       metric='euclidean',  # euclidean metric by default
                                       **kwargs)
        self._mod_n = NearestNeighbors(n_neighbors=1,
                                       leaf_size=leaf_size,
                                       algorithm=algorithm,
                                       n_jobs=n_jobs,
                                       metric='euclidean',  # euclidean metric by default
                                       **kwargs)

    @staticmethod
    def _ranking_score(d_p, d_n=None):
        """ Compute the ranking score from the positive an negative
        distances on L2 normalized data

        Parameters
        ----------
        d_p : array (n_samples,)
           distance to the positive samples
        d_n : array (n_samples,)
           (optional) distance to the negative samples

        Returns
        -------
        score : array (n_samples,)
           the ranking score in the range [-1, 1]
           For positive items score = 1 - cosine distance / 2
        """
        # convert from eucledian distance in L2 norm space to cosine similarity
        S_p = 1 - d_p/2
        if d_n is not None:
            S_n = 1 - d_n/2
            return np.where(S_p > S_n,
                            S_p + 1,
                            -1 - S_n) / 2
        else:
            return (S_p + 1) / 2

    def fit(self, X, y):
        """Fit the model using X as training data
        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training data, shape [n_samples, n_features],

        """
        X = check_array(X, accept_sparse='csr')

        index = np.arange(X.shape[0], dtype='int')

        self._index_p, self._index_n = _unzip_relevant(index, y)


        if self._index_p.shape[0] > 0:
            self._mod_p.fit(X[self._index_p])
        else:
            raise ValueError('Training sets with no positive labels are not supported!')
        if self._index_n.shape[0] > 0:
            self._mod_n.fit(X[self._index_n])
        else:
            pass

    def kneighbors(self, X=None):
        """Finds the K-neighbors of a point.
        Returns indices of and distances to the neighbors of each point.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            the input array
        Returns
        -------
        score : array
            ranking score (based on cosine similarity)
        ind : array
            Indices of the nearest points in the population matrix.
        md : dict
            Additional result data
              * ind_p : Indices of the nearest positive points
              * ind_n : Indices of the nearest negate points
              * dist_p : distance to the nearest positive points
              * dist_n : distance to the nearest negate points
        --------
        """
        X = check_array(X, accept_sparse='csr')

        D_p, idx_p_loc = self._mod_p.kneighbors(X)

        # only NearestNeighbor-1 (only one column in the kneighbors output)
        D_p = D_p[:,0]
        # map local index within _index_p, _index_n to global index
        ind_p = self._index_p[idx_p_loc[:,0]]

        md = {'dist_p': D_p,
              'ind_p': ind_p,
             }

        if self._mod_n._fit_method is not None:
            D_n, idx_n_loc = self._mod_n.kneighbors(X)
            D_n = D_n[:,0]
            ind_n = self._index_n[idx_n_loc[:,0]]
            md['ind_n'] = ind_n
            md['dist_n'] = D_n
            ind = np.where(D_p <= D_n, ind_p, ind_n)
        else:
            D_n = None
            ind = ind_p

        score = self._ranking_score(D_p, D_n)

        return  score, ind , md



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
