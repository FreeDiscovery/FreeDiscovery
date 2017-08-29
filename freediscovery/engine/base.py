# -*- coding: utf-8 -*-

import os

from sklearn.externals import joblib

from freediscovery.engine.vectorizer import FeatureVectorizer
from freediscovery.engine.pipeline import PipelineFinder
from freediscovery.exceptions import WrongParameter


class _BaseWrapper(object):
    """Base class for wrappers in FreeDiscovery

    Parameters
    ----------
    cache_dir : str
      folder where the model will be saved
    parent_id : str, optional
      dataset id
    mid : str, optional
      model id
    dataset_definition : a dataset defintion class
      one of FeatureVectorizer, EmailParser
    load_model : bool
      whether the model should be loaded from disk on class
      initialization
    """
    def __init__(self, cache_dir='/tmp/', parent_id=None, mid=None,
                 load_model=False):
        if parent_id is None and mid is None:
            raise WrongParameter('At least one of parent_id or mid should be provided!')

        if parent_id is None and mid is not None:
            self.pipeline = PipelineFinder.by_id(mid, cache_dir).parent
            self.mid = mid
        elif parent_id is not None:
            self.pipeline = PipelineFinder.by_id(parent_id, cache_dir)
            self.mid = None

        # this is an alias that should be deprecated
        self.fe = FeatureVectorizer(cache_dir=cache_dir,
                                    dsid=self.pipeline['vectorizer'])

        self.model_dir = self.pipeline.get_path() / self._wrapper_type

        if self._wrapper_type == 'search':
            # no data need to be stored on disk
            return

        if not self.model_dir.exists():
            self.model_dir.mkdir()

        if self.mid is not None:
            self._pars = self._load_pars()
        else:
            self._pars = None

        if load_model:
            if self.mid is not None:
                self.cmod = self._load_model()
            else:
                self.cmod = None

    def delete(self):
        """ Delete a trained model"""
        import shutil
        mid_dir = self.model_dir / self.mid
        shutil.rmtree(str(mid_dir), ignore_errors=True)

    def __contains__(self, mid):
        return (self.model_dir / mid).exists()

    def get_params(self):
        """ Get model parameters """
        return self._pars

    def _load_model(self):
        """ Load model from disk """
        mid = self.mid
        mid_dir = self.model_dir / mid
        if not mid_dir.exists():
            raise ValueError('Model id {} ({}) not found in the cache {}!'
                             .format(mid, self._wrapper_type, mid_dir))
        cmod = joblib.load(str(mid_dir / 'model'))
        return cmod

    def _load_pars(self, mid=None):
        """Load model parameters from disk"""
        if mid is None:
            mid = self.mid
        model_path = self.model_dir / mid
        if not model_path.exists():
            raise ValueError('Model id {} ({}) not found in the cache {}!'
                             .format(mid, self._wrapper_type, model_path))
        pars = joblib.load(str(model_path / 'pars'))
        pars['id'] = mid
        return pars

    def list_models(self):
        """ List existing models of this type """
        out = []
        if not self.model_dir.exists():
            return out
        for mid in os.listdir(str(self.model_dir)):
            try:
                pars = self._load_pars(mid)
                pars['id'] = mid
                out.append(pars)
            except:
                raise
        return out
