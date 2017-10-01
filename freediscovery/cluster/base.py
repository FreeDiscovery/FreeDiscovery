# Authors: Roman Yurchak
#
# License: BSD 3 clause

from freediscovery.stop_words import COMMON_FIRST_NAMES, CUSTOM_STOP_WORDS


# Clustering methods for FreeDiscovery
# This is highly inspired from the scikit-learn text clustering example

MAX_N_TOP_WORDS = 1000


def select_top_words(word_list, n=10):
    """ Filter out cluster term names"""
    import re
    from nltk.stem.porter import PorterStemmer
    from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
    st = PorterStemmer()
    out_st = []
    out = []
    for word in word_list:
        word_st = st.stem(word)
        if len(word_st) <= 2 or\
                re.match('\d+', word_st) or \
                re.match('[^a-zA-Z0-9]', word_st) or\
                word in COMMON_FIRST_NAMES or \
                word in CUSTOM_STOP_WORDS or\
                word in ENGLISH_STOP_WORDS or \
                word_st in out_st:  # ignore stemming duplicate
            continue
        out_st.append(word_st)
        out.append(word)
        if len(out) >= n:
            break
    return out


class _BirchDummy(object):
    """ A dummy class for Birch """
    pass


class ClusterLabels(object):
    """Calculate the cluster labels.


    Parameters
    ----------
    vect : VectorizerMixin object
       a scikit-learn's text vectorizer
    model : ClusterMixin object
       the cluster object
    lsi_components: TruncatedSVD object or None
       LSA object if it was used for clustering
    method: str, optional, default='centroid-frequency'
       the method used to compute the centroid labels
       Only 'centroid-frequency' is supported at the moment.
    n_top_words: int, default=10
       keep only most relevant n_top_words words
    """
    def __init__(self, vect, model, lsi=None,
                 method='centroid-frequency', n_top_words=6):
        self.model = model
        self.vect = vect
        self.lsi = lsi
        self.method = method
        self.n_top_words = n_top_words

    def _to_original_space(self, centroids):
        """If LSI is used recover the data points positions in the original space
        """
        if self.lsi is not None:
            return self.lsi.inverse_transform(centroids)
        else:
            return centroids

    def _get_model_centroids(self):
        method_name = type(self.model).__name__
        if method_name not in ['MiniBatchKMeans', 'AgglomerativeClustering',
                               'Birch', '_BirchDummy', 'DBSCAN']:
            raise NotImplementedError('Method name: '
                                      '{} not implented!'.format(method_name))
        # centroids were previously computed
        return self.model.cluster_centers_

    def predict(self, centroids=None):
        """ Compute the cluster labels

        Parameters
        ----------
        centroids : list, default=None
           if not None, ignore clustering given by the clustering model and
           compute labels for the given cluster centroids

        Returns
        -------
        cluster_labels: array [n_samples]
        """
        if centroids is None:
            centroids = self._get_model_centroids()

        if self.n_top_words > MAX_N_TOP_WORDS:
            raise ValueError
        if self.method == 'centroid-frequency':
            return self._predict_centroid_freq(centroids)
        else:
            raise ValueError

    def _predict_centroid_freq(self, centroids):
        """ Return cluster labels based on
        the most frequent words (tfidf) at cluster centroids """

        centroids = self._to_original_space(centroids)

        n_clusters = centroids.shape[0]

        centroids_ordered = centroids.argsort()[:, ::-1]

        terms = self.vect.get_feature_names()
        cluster_terms = []
        for i in range(n_clusters):
            terms_i = [terms[ind]
                       for ind in centroids_ordered[i, :MAX_N_TOP_WORDS]]
            terms_i = select_top_words(terms_i, self.n_top_words)
            cluster_terms.append(terms_i)
        return cluster_terms
