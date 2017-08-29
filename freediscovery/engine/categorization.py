# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import logit, expit
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

from freediscovery.engine.base import _BaseWrapper
from freediscovery.utils import setup_model, _rename_main_thread
from freediscovery.neighbors import NearestCentroidRanker, NearestNeighborRanker
from freediscovery.metrics import _scale_cosine_similarity
from freediscovery.exceptions import (WrongParameter, NotImplementedFD,
                                      OptionalDependencyMissing)


class _CategorizerWrapper(_BaseWrapper):
    """ Document categorization model

    The option `use_hashing=True` must be set for the feature extraction.
    Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

    Parameters
    ----------
    cache_dir : str
      folder where the model will be saved
    parent_id : str, optional
      dataset id
    mid : str, optional
      model id
    cv_scoring : str, optional, default='roc_auc'
      score that is used for Cross Validation, cf. sklearn
    cv_n_folds : str, optional
      number of K-folds used for Cross Validation
    """

    _wrapper_type = "categorizer"

    def __init__(self, cache_dir='/tmp/',  parent_id=None, mid=None,
                 cv_scoring='roc_auc', cv_n_folds=3, random_state=None):

        super(_CategorizerWrapper, self).__init__(cache_dir=cache_dir,
                                          parent_id=parent_id,
                                          mid=mid, load_model=True)

        if mid is not None:
            self.le = joblib.load(str(self.model_dir / mid / 'label_encoder'))
        self.cv_scoring = cv_scoring
        self.cv_n_folds = cv_n_folds
        self.random_state = random_state

    @staticmethod
    def _build_estimator(Y_train, method, cv, cv_scoring, cv_n_folds, random_state=None, **options):
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
                cmod = LinearSVC(random_state=random_state, **options)
            else:
                try:
                    from freediscovery_extra import make_linearsvc_cv_model
                except ImportError:
                    raise OptionalDependencyMissing('freediscovery_extra')
                cmod = make_linearsvc_cv_model(cv_obj, cv_scoring, **options)
        elif method == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            if cv is None:
                cmod = LogisticRegression(random_state=random_state, **options)
            else:
                try:
                    from freediscovery_extra import make_logregr_cv_model
                except ImportError:
                    raise OptionalDependencyMissing('freediscovery_extra')
                cmod = make_logregr_cv_model(cv_obj, cv_scoring, **options)
        elif method == 'NearestCentroid':
            cmod = NearestCentroidRanker()
        elif method == 'NearestNeighbor':
            cmod = NearestNeighborRanker()
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
                                 max_iter=200, activation='identity',
                                 verbose=0,
                                 random_state=random_state)
        else:
            raise WrongParameter('Method {} not implemented!'.format(method))
        return cmod

    def fit(self, index, y, method='LinearSVC', cv=None):
        """
        Train the categorization model

        Parameters
        ----------
        index : array-like, shape (n_samples)
           document indices of the training set
        y : array-like, shape (n_samples)
           target class relative to index (string or int)
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

        valid_methods = ["LinearSVC", "LogisticRegression", "xgboost",
                         "NearestCentroid", "NearestNeighbor"]

        if method in ['MLPClassifier']:
            raise WrongParameter('method={} is implemented but not production ready. It was disabled for now.'.format(method))

        if method not in valid_methods:
            raise WrongParameter('method={} is not supported, should be one of {}'.format(
                method, valid_methods)) 
        if cv is not None and method in ['NearestNeighbor', 'NearestCentroid']:
            raise WrongParameter('Cross validation (cv={}) not supported with {}'.format(
                                        cv, method))

        if cv not in [None, 'fast', 'full']:
            raise WrongParameter('cv')

        d_all = self.pipeline.data

        X_train = d_all[index, :]

        Y_labels = y

        self.le = LabelEncoder()
        Y_train = self.le.fit_transform(Y_labels)

        cmod = self._build_estimator(Y_train, method, cv, self.cv_scoring, self.cv_n_folds)

        mid, mid_dir = setup_model(self.model_dir)

        if method == 'xgboost' and not cv:
            cmod.fit(X_train, Y_train, eval_metric='auc')
        else:
            cmod.fit(X_train, Y_train)

        joblib.dump(self.le, str(mid_dir / 'label_encoder'))
        joblib.dump(cmod, str(mid_dir / 'model'))

        pars = {
            'method': method,
            'index': index,
            'y': y,
            'categories': self.le.classes_
            }
        pars['options'] = cmod.get_params()
        self._pars = pars
        joblib.dump(pars, str(mid_dir / 'pars'))

        self.mid = mid
        self.cmod = cmod
        return cmod, Y_train

    def predict(self, chunk_size=5000, ml_output='probability', metric='cosine'):
        """
        Predict the relevance using a previously trained model

        Parameters
        ----------
        chunck_size : int
           chunk size
        ml_output : str
           type of the output in ['decision_function', 'probability'],
           only affects ML methods. default: 'probability'
        metric : str   
            The similarity returned by nearest neighbor classifier in
            ['cosine', 'jaccard', 'cosine_norm', 'jaccard_norm'].
            default: 'cosine'

        Returns
        -------
        res : ndarray [n_samples, n_classes]
           the score for each class
        nn_ind : {ndarray [n_samples, n_classes], None}
           the index of the nearest neighbor for each class
           (when the NearestNeighborRanker is used)
        """
        if ml_output not in ['probability', 'decision_function']:
            raise ValueError(("Wrong input value ml_output={}, must be one of "
                              "['probability', 'decision_function']")
                             .format(ml_output))

        if ml_output == 'probability':
            ml_output = 'predict_proba'

        if self.cmod is not None:
            cmod = self.cmod
        else:
            raise WrongParameter('The model must be trained first, or sid must be provided to load\
                    a previously trained model!')

        ds = self.pipeline.data

        nn_ind = None
        if isinstance(cmod, NearestNeighborRanker):
            res, nn_ind_orig = cmod.kneighbors(ds)
            res = _scale_cosine_similarity(res, metric=metric)
            nn_ind = self._pars['index'][nn_ind_orig]
        elif hasattr(cmod, ml_output):
            res = getattr(cmod, ml_output)(ds)
        elif hasattr(cmod, 'decision_function'):
            # and we need predict_proba
            res = cmod.decision_function(ds)
            res = expit(res)
        elif hasattr(cmod, 'predict_proba'):
            # and we need decision_function
            res = cmod.predict_proba(ds)
            res = logit(res)
        else:
            raise ValueError('Model {} has neither decision_function nor predict_proba methods!'.format(cmod))

        # handle the case of binary categorization
        # as two classes categorization
        if res.ndim == 1:
            if ml_output == 'decision_function':
                res_p = res
                res_n = - res
            else:
                res_p = res
                res_n = 1 - res
            res = np.hstack((res_n[:, None], res_p[:, None]))
        return res, nn_ind

    def _load_pars(self, mid=None):
        """Load model parameters from disk"""
        if mid is None:
            mid = self.mid
        mid_dir = self.model_dir / mid
        pars = super(_CategorizerWrapper, self)._load_pars(mid)
        cmod = joblib.load(str(mid_dir / 'model'))
        pars['options'] = cmod.get_params()
        return pars
