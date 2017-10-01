# Authors: Roman Yurchak
#
# License: BSD 3 clause

from sklearn.metrics import pairwise_distances


class Search(object):
    """ (Semantic) search in a document collection

    Parameters
    ----------
    vectorizer : {CountVectorizer, HashingVectorizer}
      the (fitted) vectorizer that was used extract tokens from the
      document collection
    tfidf : {TfidfTransformer, SmartTfidfTransfomer}
      the (fitted) IDF transformer used to weight and normalize the
      bag of word/n-gram features
    lsi : TruncatedSVD
      (optional) an LSI model fitted on the vectorised document-term matrix
      If provided this corresponds to a semantic search, default=None
    """
    def __init__(self, vectorizer, tfidf, lsi=None):
        self.vectorizer = vectorizer
        self.tfidf = tfidf
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
        q_vect = self.tfidf.transform(q_vect)

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

    def search_id(self, internal_id, metric='cosine'):
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
