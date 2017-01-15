# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from collections import OrderedDict

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

def _split_path(path, force_os=False):
    """ A helper function that splits the path into a list

    Parameters
    ----------
    path : str
      path to split
    force_os : bool, default=False
      replace \\ with / even on Linux 
    """
    if (force_os or os.name == 'nt') and "\\" in path: # windows
        # make all paths Linux like
        path = path.replace('\\', '/')

    head, tail = os.path.split(path)

    if not head:
        return [tail]
    elif head == path:
        if tail:
            return [head, tail]
        else:
            return [head]
    else:
        if not tail:
            return _split_path(head)
        else:
            return _split_path(head) + [tail]


class PipelineFinder(OrderedDict):
    """Walk through the hierarchy of existing models
    and find the processing pipeline that terminate with the
    model having the given uid

    Parameters
    ----------
    uid : str
      a unique model id
    cache_dir : str
      folder where models are saved
    ingestion_method : str, default='vectorizer'
      default ingestion method (one of ['vectorizer', 'parser'])
      unless email threading is used, this whould be set to 'vectorizer'

    Returns
    -------
    result : OrderedDict
      the prior processing pipeline with as keys the processing step type
      and as values the model ids
    """

    def __init__(self, mid=None, cache_dir="/tmp/", ingestion_method='vectorizer', **args):
        self.ingestion_method = ingestion_method
        self._loaded_models = {}

        self.mid = mid

        cache_dir = os.path.normpath(cache_dir)
        if 'ediscovery_cache' not in cache_dir:  # not very pretty
            cache_dir = os.path.join(cache_dir, "ediscovery_cache")
        self.cache_dir = cache_dir

        super(PipelineFinder, self).__init__(**args)


    @classmethod
    def by_id(cls, mid, cache_dir="/tmp/", ingestion_method='vectorizer'):
        """ Find a pipeline by id

        Parameters
        ----------
        mid : str
          a unique model id
        cache_dir : str
          folder where models are saved
        ingestion_method : str, default='vectorizer'
          default ingestion method (one of ['vectorizer', 'parser'])
          unless email threading is used, this whould be set to 'vectorizer'

        Returns
        -------
        result : OrderedDict
          the prior processing pipeline with as keys the processing step type
          and as values the model ids
        """

        pipeline = cls(mid=mid, cache_dir=cache_dir,
                       ingestion_method=ingestion_method)

        cache_dir_base = os.path.dirname(cache_dir)
        _break_flag = False
        for root, subdirs, files in os.walk(cache_dir):
            root = os.path.relpath(root, cache_dir_base)
            for sdir in subdirs:
                path = os.path.join(root, sdir)
                path_hierarchy = _split_path(path)
                if len(path_hierarchy) % 2 == 1:
                    # the path is of the form
                    # ['ediscovery_cache']
                    # or ['ediscovery_cache', 'ce196de4c7de4e57', 'cluster']
                    # ignore it
                    continue

                if path_hierarchy[-1] == mid:
                    # found the model
                    _break_flag = True
                    break
            if _break_flag:
                break
        else:
            raise ModelNotFound('Model id {} not found in {}!'.format(mid, cache_dir))

        if path_hierarchy[0] == 'ediscovery_cache':
            path_hierarchy[0] = ingestion_method
        else:
            raise ValueError('path_hierarchy should start with ediscovery_cache',
                             'this indicates a bug in the code')

        for idx in range(len(path_hierarchy)//2):
            key, val = path_hierarchy[2*idx], path_hierarchy[2*idx+1]
            if key in pipeline:
                raise NotImplementedError('The current PipelineFinder class does not support'
                                          'multiple identical processing steps'
                                          'duplicates of {} found!'.format(key))
            pipeline[key] = val
        return pipeline

    @property
    def parent(self):
        """ Make a new pipeline without the latest node """

        if len(self.keys()) <= 1:
            raise ValueError("Can't take the parent of a root node!")

        # get all the steps except the last one
        steps = self.copy()
        steps.popitem(last=True)

        return PipelineFinder(mid=list(steps.values())[-1],
                              cache_dir=self.cache_dir,
                              ingestion_method=self.ingestion_method,
                              **steps)

    def get_path(self, mid=None):
        """ Find the path to the model specified by mid """
        import itertools

        if mid is None:
            mid = self.mid

        if mid not in self.values():
            raise ValueError('{} is not a processing step current pipeline,\n {}'.format(mid, self)) 
        idx = list(self.values()).index(mid)
        valid_keys = list(self.keys())[:idx]
        path = list(itertools.chain.from_iterable(
                            [[key, self[key]] for key in valid_keys]))
        path += [list(self.keys())[idx]]
        path[0] = 'ediscovery_cache'
        return os.path.join(*path)




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
                 dataset_definition=FeatureVectorizer,
                 load_model=False):
        if parent_id is None and mid is None:
            raise WrongParameter('At least one of parent_id or mid should be provided!')

        if parent_id is None and mid is not None:
            self.pipeline = PipelineFinder.by_id(mid, cache_dir).parent
            self.mid = mid
        elif parent_id is not None:
            self.pipeline = PipelineFinder.by_id(parent_id, cache_dir)
            self.mid = None

        self.fe = dataset_definition(cache_dir=cache_dir,
                                     dsid=self.pipeline['vectorizer'])

        self.model_dir = os.path.join(self.pipeline.get_path(), self._wrapper_type)

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        if self.mid is not None:
            self._pars = self._load_pars()
        else:
            self._pars = None

        if load_model:
            if self.mid is not None:
                self.cmod = self._load_model()
            else:
                self.cmod = None


    def get_path(self, mid):
        parent_id = self.get_dsid(self.fe.cache_dir, mid)
        return os.path.join(self.fe.cache_dir, parent_id, self._wrapper_type, mid)


    def get_dsid(self, cache_dir, mid):
        if 'ediscovery_cache' not in cache_dir:  # not very pretty
            cache_dir = os.path.join(cache_dir, "ediscovery_cache")
        for parent_id in os.listdir(cache_dir):
            mid_path = os.path.join(cache_dir, parent_id, self._wrapper_type)
            if not os.path.exists(mid_path):
                continue
            for mid_el in os.listdir(mid_path):
                if mid_el == mid:
                    return parent_id
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

    def _load_model(self):
        mid = self.mid
        mid_dir = os.path.join(self.model_dir, mid)
        if not os.path.exists(mid_dir):
            raise ValueError('Model id {} ({}) not found in the cache {}!'.format(
                             mid, self._wrapper_type, mid_dir))
        cmod = joblib.load(os.path.join(mid_dir, 'model'))
        return cmod


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


