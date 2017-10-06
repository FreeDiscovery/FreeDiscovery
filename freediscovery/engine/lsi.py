# -*- coding: utf-8 -*-

import numpy as np
from sklearn.externals import joblib

from freediscovery.engine.base import _BaseWrapper
from freediscovery.lsi import _compute_lsi_dimensionality, _TruncatedSVD_LSI
from freediscovery.utils import setup_model
from freediscovery.exceptions import WrongParameter


class _LSIWrapper(_BaseWrapper):
    """Document categorization using Latent Semantic Indexing (LSI)

    Parameters
    ----------
    cache_dir : str
       folder where the model will be saved
    parent_id : str
       dataset id
    mid : str
       LSI model id (the dataset id will be inferred)
    verbose : bool, optional
       print progress messages
    """

    _wrapper_type = "lsi"

    def __init__(self, cache_dir='/tmp/', parent_id=None,
                 mid=None, verbose=False, random_state=None, mode='r'):

        if mode not in ['r', 'w', 'fw']:
            raise WrongParameter('mode={} must be one of "r", "w", "fw"'
                                 .format(mode))
        self.mode = mode

        super(_LSIWrapper, self).__init__(cache_dir=cache_dir,
                                          parent_id=parent_id, mid=mid,
                                          mode=mode)
        self.random_state = random_state

    def _load_features(self):
        mid_dir = self.fe.dsid_dir / self._wrapper_type / self.mid
        return joblib.load(str(mid_dir / 'data'))

    def fit_transform(self, n_components=150, n_iter=5, alpha=0.33):
        """
        Perform the SVD decomposition

        Parameters
        ----------
        n_components : int
           number of selected singular values (number of LSI dimensions)
        n_iter : int
           number of iterations for the stochastic SVD algorithm

        Returns
        -------
        mid : str
           model id
        lsi : _BaseWrapper
           the TruncatedSVD object
        exp_var : float
           the explained variance of the SVD decomposition
        """
        parent_id = self.pipeline.mid

        dsid_dir = self.fe.dsid_dir
        if not dsid_dir.exists():
            raise IOError

        pars = {'parent_id': parent_id, 'n_components': n_components}

        mid_dir_base = dsid_dir / self._wrapper_type

        mid, mid_dir = setup_model(mid_dir_base, mid=self.mid, mode=self.mode)

        ds = self.pipeline.data
        n_components_opt = _compute_lsi_dimensionality(n_components, *ds.shape,
                                                       alpha=alpha)
        svd = _TruncatedSVD_LSI(n_components=n_components_opt,
                                n_iter=n_iter, random_state=self.random_state)
        lsi = svd
        lsi.fit(ds)

        ds_p = lsi.transform_lsi_norm(ds)

        joblib.dump(pars, str(mid_dir / 'pars'))
        joblib.dump(lsi, str(mid_dir / 'model'))
        joblib.dump(ds_p, str(mid_dir / 'data'))

        exp_var = lsi.explained_variance_ratio_.sum()
        self.mid = mid

        return lsi, exp_var

    def append(self, X):
        """ Add new documents to an existing LSI model

        Parameters
        ----------
        X : scipy.sparse CSR array
          the additional data to add to the index

        """
        mid_dir = self.model_dir / self.mid
        lsi = joblib.load(str(mid_dir / 'model'))
        Y_old = joblib.load(str(mid_dir / 'data'))
        Y_new = lsi.transform_lsi_norm(X)
        Y = np.vstack((Y_old, Y_new))
        joblib.dump(Y, str(mid_dir / 'data'))
