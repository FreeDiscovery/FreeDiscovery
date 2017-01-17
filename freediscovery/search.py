# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from sklearn.metrics import pairwise_distances


class Search(object):
    """ Perform (semantic) search in a document collection

    Parameters
    ----------
    vectorizer : {TfidfVectorizer, HashingVectorizer}
      the (fitted) vectorizer that was used on the document collection
    lsi : TruncatedSVD
      (optional) an LSI model fitted on the vectorised document-term matrix
      If provided this corresponds to a semantic search, default=None
    n_jobs : int
      The number of parallel jobs to run for neighbors search.
      If -1, then the number of jobs is set to the number of CPU cores.
    """
    def __init__(self, vectorizer, lsi=None, n_jobs=1):
        self.vectorizer = vectorizer
        self.lsi = lsi
        self.n_jobs = n_jobs
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

    def search(self, text):
        """
        Perform the search operation

        Parameters
        ----------
        text : str
          the search query text
        """
        from .lsi import _TruncatedSVD_LSI

        if self._fit_X is None:
            raise ValueError('Estomator must be fitted before using the search method!')
        q_vect = self.vectorizer.transform([text])


        if self.lsi is not None:
            if isinstance(self.lsi, _TruncatedSVD_LSI): # this is a hack need to be rewritten
                q_lsi = self.lsi.transform_lsi_norm(q_vect)
            else:  # a regular TruncatedSVD object
                q_lsi = self.lsi.transform(q_vect)
            q = q_lsi
        else:
            q = q_vect

        dist = pairwise_distances(q, self._fit_X,
                                  'euclidean',
                                  n_jobs=self.n_jobs, squared=True)

        return dist[0]
