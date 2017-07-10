# -*- coding: utf-8 -*-

import warnings

from sklearn.metrics import pairwise_distances
from sklearn.externals import joblib

from .base import _BaseWrapper


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
        else:
            vect = self.fe.vect_
            vect.set_params(input='content')

        X = self.pipeline.data

        if "lsi" in self.pipeline:
            lsi = joblib.load(str(
                              self.pipeline.get_path(self.pipeline['lsi']) /
                              'model'))
        else:
            lsi = None

        s = Search(vect, lsi)
        s.fit(X)

        if internal_id is not None:
            dist = s.search_id(internal_id, metric=metric)
        else:
            dist = s.search(text, metric=metric)

        return dist


class Search(object):
    """ Perform (semantic) search in a document collection

    Parameters
    ----------
    vectorizer : {TfidfVectorizer, HashingVectorizer}
      the (fitted) vectorizer that was used on the document collection
    lsi : TruncatedSVD
      (optional) an LSI model fitted on the vectorised document-term matrix
      If provided this corresponds to a semantic search, default=None
    """
    def __init__(self, vectorizer, lsi=None):
        self.vectorizer = vectorizer
        self.lsi = lsi
        self._fit_X = None

    def fit(self, X):
        """
        Fit using a document term matrix (optionally in the LSI space)

        Parameters
        ----------
        X : ndarray
          the sparse document-terms arrays (if lsi was not used) or
          dense documents / lsi terms array (if lsi was provided)
        """
        self._fit_X = X

    def search(self, text, metric='cosine'):
        """
        Perform the search operation

        Parameters
        ----------
        text : str
          the search query text
        metric : str
          the output metric to use
        """
        from .lsi import _TruncatedSVD_LSI

        if self._fit_X is None:
            raise ValueError('Estomator must be fitted before '
                             'using the search method!')
        q_vect = self.vectorizer.transform([text])

        if self.lsi is not None:
            # this is a hack need to be rewritten
            if isinstance(self.lsi, _TruncatedSVD_LSI):
                q_lsi = self.lsi.transform_lsi_norm(q_vect)
            else:  # a regular TruncatedSVD object
                q_lsi = self.lsi.transform(q_vect)
            q = q_lsi
        else:
            q = q_vect

        scores = self._compute_score(q, self._fit_X, metric)

        return scores

    def search_id(self, internal_id, metric='jaccard_norm'):
        """
        Perform the search operation

        Parameters
        ----------
        internal_id : int
          the internal_id of the document used as a search query
        metric : str
          the output metric to use
        """
        if self._fit_X is None:
            raise ValueError('Estomator must be fitted before '
                             'using the search method!')

        X = self._fit_X

        q = X[[internal_id], :]

        scores = self._compute_score(q, self._fit_X, metric)

        return scores

    @staticmethod
    def _compute_score(q, X, metric):
        """ Internal method to compute the scores """

        from .metrics import _scale_cosine_similarity

        dist = pairwise_distances(q, X, 'cosine')
        dist = dist[0]

        scores = 1 - dist

        scores = _scale_cosine_similarity(scores, metric=metric)

        return scores
