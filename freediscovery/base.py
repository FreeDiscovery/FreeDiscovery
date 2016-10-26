# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from .exceptions import (DatasetNotFound, ModelNotFound, InitException,
                            WrongParameter)


class BaseEstimator(object):
    def get_path(self, mid):
        dsid = self.get_dsid(self.fe.cache_dir, mid)
        return os.path.join(self.fe.cache_dir, dsid, self._DIRREF, mid)

    def get_dsid(self, cache_dir, mid):
        if not 'ediscovery_cache' in cache_dir: # not very pretty
            cache_dir = os.path.join(cache_dir, "ediscovery_cache")
        for dsid in os.listdir(cache_dir):
            mid_path = os.path.join(cache_dir, dsid, self._DIRREF)
            if not os.path.exists(mid_path):
                continue
            for mid_el in os.listdir(mid_path):
                if mid_el == mid:
                    return dsid
        raise ModelNotFound('Model id {} not found in {}/*/{}!'.format(mid,
                                    cache_dir, self._DIRREF))

    def delete(self):
        """ Delete a trained model"""
        import shutil
        mid_dir = os.path.join(self.model_dir, self.mid)
        shutil.rmtree(mid_dir, ignore_errors=True)


    def __contains__(self, mid):
        mid_dir = os.path.join(self.model_dir, mid)
        return os.path.exists(mid_dir)


    def list_models(self):
        out = []
        for mid in os.listdir(self.model_dir):
            try:
                pars = self.load_pars(mid)
                pars['id'] =  mid
                out.append(pars)
            except:
                pass
        return out


