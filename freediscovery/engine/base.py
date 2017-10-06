# -*- coding: utf-8 -*-

import os

from sklearn.externals import joblib

from freediscovery.engine.vectorizer import FeatureVectorizer
from freediscovery.engine.pipeline import PipelineFinder
from freediscovery.engine.utils import validate_mid
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
    mode : bool
      read/write mode. One of 'r', 'w', 'fw' (write, overwrite if exists)
    """
    def __init__(self, cache_dir='/tmp/', parent_id=None, mid=None,
                 load_model=False, mode='r'):
        if parent_id is None and mid is None:
            raise WrongParameter('At least one of parent_id or mid '
                                 'should be provided!')

        if self._wrapper_type == 'lsi' and self.mode in ['w', 'fw']:
            # lsi supports explicitly providing mid at creation
            if parent_id is None:
                raise WrongParameter(('parent_id={} must be provided for '
                                      'model creation!')
                                     .format(parent_id))
            else:
                validate_mid(parent_id)
                self.pipeline = PipelineFinder.by_id(parent_id, cache_dir)
                if mid is not None:
                    validate_mid(mid)
                self.mid = mid
        else:
            if parent_id is None and mid is not None:
                validate_mid(mid)
                self.pipeline = PipelineFinder.by_id(mid, cache_dir).parent
                self.mid = mid
            elif parent_id is not None:
                validate_mid(parent_id)
                self.pipeline = PipelineFinder.by_id(parent_id, cache_dir)
                self.mid = None

        # this only affects LSI
        if mode not in ['r', 'w', 'fw']:
            raise WrongParameter('mode={} must be one of "r", "w", "fw"'
                                 .format(mode))
        self.mode = mode

        # this is an alias that should be deprecated
        self.fe = FeatureVectorizer(cache_dir=cache_dir,
                                    dsid=self.pipeline['vectorizer'])

        self.model_dir = self.pipeline.get_path() / self._wrapper_type

        if self._wrapper_type == 'search':
            # no data need to be stored on disk
            return

        if not self.model_dir.exists():
            self.model_dir.mkdir()

        if self.mid is not None and self.mode == 'r':
            self._pars = self._load_pars()
        else:
            self._pars = None

        if load_model:
            if self.mid is not None and self.mode == 'r':
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
