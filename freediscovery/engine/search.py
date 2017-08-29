# -*- coding: utf-8 -*-

import warnings

from sklearn.externals import joblib

from freediscovery.engine.base import _BaseWrapper
from freediscovery.search import Search


class _SearchWrapper(_BaseWrapper):
    """ Document search wrapper

    Parameters
    ----------
    cache_dir : str
      folder where the model will be saved
    parent_id : str, optional
      dataset id
    mid : str, optional
      model id
    """

    _wrapper_type = "search"

    def __init__(self, cache_dir='/tmp/',  parent_id=None, mid=None):

        super(_SearchWrapper, self).__init__(cache_dir=cache_dir,
                                             parent_id=parent_id,
                                             mid=mid)

    def search(self, text, internal_id=None, metric='cosine'):
        """
        Search given some text query

        Parameters
        ----------
        text : str
          the query string. This will be ignored if an
          internal_id parameter is provided.
        internal_id : int
          the document id used as a search query.
          If provided the text input will be ignored.
        metric : str
          the output metric to use
        """
        if internal_id is not None:
            if 'lsi' not in self.pipeline:
                warnings.warn('Search using a document_id as a query'
                              ' should not be applied in the space of '
                              ' raw document term vectors due'
                              ' to the curse of dimensionality.'
                              ' Please add an LSI processing'
                              ' step (i.e. use `parent_id=lsi_id`)')
            vect = None
            tfidf = None
        else:
            vect = self.fe.vect_
            vect.set_params(input='content')
            tfidf = self.fe.tfidf_

        X = self.pipeline.data

        if "lsi" in self.pipeline:
            lsi = joblib.load(str(
                              self.pipeline.get_path(self.pipeline['lsi']) /
                              'model'))
        else:
            lsi = None

        s = Search(vect, tfidf, lsi)
        s.fit(X)

        if internal_id is not None:
            dist = s.search_id(internal_id, metric=metric)
        else:
            dist = s.search(text, metric=metric)

        return dist
