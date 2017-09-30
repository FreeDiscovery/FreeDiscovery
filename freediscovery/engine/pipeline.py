# -*- coding: utf-8 -*-

import os
from collections import OrderedDict

from freediscovery.exceptions import ModelNotFound
from freediscovery.utils import _split_path
from pathlib import Path

from sklearn.externals import joblib


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

    Returns
    -------
    result : OrderedDict
      the prior processing pipeline with as keys the processing step type
      and as values the model ids
    """

    def __init__(self, mid=None, cache_dir="/tmp/", steps=None):
        self._loaded_models = {}

        self.mid = mid
        if steps is None:
            steps = OrderedDict()

        self.cache_dir = self._normalize_cachedir(cache_dir)

        super(PipelineFinder, self).__init__(steps)

    @staticmethod
    def _normalize_cachedir(cache_dir):
        """ Normalize the cachedir path. This ensures that the cache_dir
        ends with "ediscovery_cache"
        """
        cache_dir = os.path.normpath(str(cache_dir))
        if 'ediscovery_cache' not in cache_dir:  # not very pretty
            cache_dir = os.path.join(cache_dir, "ediscovery_cache")
        return Path(cache_dir)

    @classmethod
    def by_id(cls, mid, cache_dir="/tmp/"):
        """ Find a pipeline by id

        Parameters
        ----------
        mid : str
          a unique model id
        cache_dir : str
          folder where models are saved

        Returns
        -------
        result : OrderedDict
          the prior processing pipeline with as keys the processing step type
          and as values the model ids
        """
        cache_dir = cls._normalize_cachedir(cache_dir)

        pipeline = cls(mid=mid, cache_dir=cache_dir)

        cache_dir_base = cache_dir.parent
        _break_flag = False
        for root, subdirs, files in os.walk(str(cache_dir)):
            root = os.path.relpath(root, str(cache_dir_base))
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
            path_hierarchy[0] = 'vectorizer'
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

        # create a copy
        steps = OrderedDict(self)
        # get all the steps except the last one
        steps.popitem(last=True)

        return PipelineFinder(mid=list(steps.values())[-1],
                              cache_dir=self.cache_dir, steps=steps)

    @property
    def data(self):
        """ Load the data provided by the last node of the pipeline """
        last_node = list(self.keys())[-1]
        ds_path = self.get_path(self[last_node])

        if last_node == "vectorizer":
            full_path = ds_path / 'features'
        elif last_node == 'lsi':
            full_path = ds_path / 'data'
        return joblib.load(str(full_path))

    def get_path(self, mid=None, absolute=True):
        """ Find the path to the model specified by mid """
        import itertools

        if mid is None:
            mid = self.mid

        if mid not in self.values():
            raise ValueError('{} is not a processing step current pipeline,\n {}'.format(mid, self)) 
        idx = list(self.values()).index(mid)
        valid_keys = list(self.keys())[:idx+1]
        path = list(itertools.chain.from_iterable(
                            [[key, self[key]] for key in valid_keys]))
        if absolute:
            del path[0]  # "ediscovery_cache" is already present in cache_dir
            rel_path = os.path.join(*path)
            return self.cache_dir / rel_path
        else:
            path[0] = 'ediscovery_cache'
            return Path(os.path.join(*path))
