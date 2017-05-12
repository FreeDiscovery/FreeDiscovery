# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
from sklearn.externals import joblib
import scipy.sparse
import pandas as pd

from ..base import _BaseWrapper
from ..utils import setup_model
from ..stop_words import COMMON_FIRST_NAMES, CUSTOM_STOP_WORDS
from .utils import _dbscan_noisy2unique
from .birch import _BirchHierarchy


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

    This is an internal class that is called by Clustering

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


class _BaseClusteringWrapper(object):

    def _merge_response(self, cluster_id):
        res_scores = pd.DataFrame({'internal_id': np.arange(self.fe.n_samples_,
                                                            dtype='int'),
                                   'cluster_id': cluster_id})

        res_scores = res_scores.set_index('internal_id', verify_integrity=True)
        fdb = self.fe.db_.data.set_index('internal_id', verify_integrity=True)

        y = fdb.merge(res_scores,
                      how='inner',
                      left_index=True,
                      right_index=True,
                      suffixes=('_db', '_cluster'))
        return y


class _ClusteringWrapper(_BaseWrapper, _BaseClusteringWrapper):
    """Document clustering

    The algorithms are adapted from scikit learn.

    The option `use_hashing=False` must be set for the feature extraction.
    Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

    Parameters
    ----------
    cache_dir : str
       directory where to save temporary and regression files
    parent_id : str, optional
       dataset id
    mid : str, optional
       model id
    """

    _wrapper_type = "cluster"

    def __init__(self, cache_dir='/tmp/', parent_id=None, mid=None, metric='cosine'):

        super(_ClusteringWrapper, self).__init__(cache_dir=cache_dir,
                                                 parent_id=parent_id,
                                                 mid=mid, load_model=True)

        if self.fe.pars_['use_hashing']:
            raise NotImplementedError('Using clustering with hashed features '
                                      'is not supported by FreeDiscovery!')

        if mid is None:
            self.metric = metric
        else:
            self.metric = self._pars['metric']
        self.km = self.cmod
        del self.cmod
        self._fit_X = None

    def _load_htree(self):
        return joblib.load(str(self.model_dir / self.mid / 'htree'))

    def _cluster_func(self, n_clusters, km, pars=None):
        """ A helper function for clustering, includes base method used by
        all clustering implementations """
        import warnings
        from sklearn.neighbors import NearestCentroid

        new_pars = km.get_params(deep=True)
        new_pars.pop('metric', None)  # dbscan always uses the euclidean metric
        pars.update(new_pars)
        if self._fit_X is None:
            self._fit_X = X = self.pipeline.data
        else:
            X = self._fit_X

        mid, mid_dir = setup_model(self.model_dir)

        with warnings.catch_warnings():
            if type(km).__name__ != "DBSCAN":
                warnings.filterwarnings("ignore", category=DeprecationWarning)
            km.fit(X)

        self.mid = mid
        self.mid_dir = mid_dir

        if type(km).__name__ in ['Birch', '_BirchDummy'] and n_clusters is None:
            # hierarcical clustering, centroids are computed at a later time..
            labels_ = None

        else:
            if type(km).__name__ == "DBSCAN":
                labels_ = _dbscan_noisy2unique(km.labels_)
                n_clusters = len(np.unique(labels_))
                km.labels_ = labels_
            else:
                labels_ = km.labels_

            # i.e. model is not MiniBatchKMeans => compute centroids
            if not hasattr(km, 'cluster_centers_'):
                km.cluster_centers_ = NearestCentroid().fit(X,
                                                            labels_).centroids_
        if type(km).__name__ in ['Birch', '_BirchDummy']:
            if pars['n_clusters'] is None:
                hmod = _BirchHierarchy(km, metric=pars['metric'])
                hmod.fit(X)
                htree = hmod.htree
                km = _BirchDummy()
            else:
                del km.root_
                del km.dummy_leaf_
                htree = None
        else:
            htree = None

        pars['n_clusters'] = n_clusters

        joblib.dump(km, str(self.model_dir / mid / 'model'))
        joblib.dump(htree, str(self.model_dir / mid / 'htree'))
        joblib.dump(pars, str(self.model_dir / mid / 'pars'))

        self.km = km
        self.htree = htree
        self._pars = pars

        return labels_

    def _get_htree(self, X=None, metric='cosine'):
        km = self.km
        method_name = type(km).__name__
        if method_name == 'AgglomerativeClustering':
            htree = {'n_leaves': km.n_leaves_,
                     'n_components': km.n_components_,
                     'children': km.children_.tolist()}
        elif method_name in ['Birch', '_BirchDummy']\
                and self._pars['n_clusters'] is None:
            hmod = _BirchHierarchy(km, metric=metric)
            hmod.fit(X)
            htree = hmod.htree
        else:
            htree = {}
        return htree

    def compute_labels(self, label_method='centroid-frequency', n_top_words=6,
                       cluster_indices=None):
        """ Compute the cluster labels

        Parameters
        ----------
        label_method : str, default='centroid-frequency'
            the method used for computing the cluster labels
        n_top_words : int, default=10
           keep only most relevant n_top_words words
        cluster_indices : list of lists, default=None
           if not None, ignore clustering given by the clustering model
           and compute terms for the cluster provided by the given indices

        Returns
        -------
        cluster_labels : array [n_samples]
        """
        vect = self.fe.vect_

        if 'lsi' in self.pipeline:
            lsi = joblib.load(str(
                              self.pipeline.get_path(self.pipeline['lsi']) /
                              'model'))
        else:
            lsi = None

        lb = ClusterLabels(vect, self.km, lsi=lsi,
                           method=label_method, n_top_words=n_top_words)

        if cluster_indices is not None:
            if self._fit_X is None:
                X = self.pipeline.data
            else:
                X = self._fit_X
            centroids = []
            for indices in cluster_indices:
                X_sl = X[indices]
                centroids.append(X_sl.mean(axis=0))
            if lsi is None:
                centroids = scipy.sparse.csr_matrix(centroids)
            else:
                centroids = np.array(centroids)
        else:
            centroids = None

        terms = lb.predict(centroids=centroids)
        return terms

    def k_means(self, n_clusters, batch_size=1000):
        """
        Perform K-mean clustering

        Parameters
        ----------
        n_clusters : int
           number of clusters
        batch_size : int
           the bath size for the MiniBatchKMeans algorithm
        """
        from sklearn.cluster import MiniBatchKMeans
        pars = {"batch_size": batch_size, 'is_hierarchical': False,
                "metric": self.metric}
        km = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++',
                             n_init=10,
                             init_size=batch_size, batch_size=batch_size)
        return self._cluster_func(n_clusters, km, pars)

    def birch(self, n_clusters=None, threshold=0.5, branching_factor=50,
              max_tree_depth=None):
        """
        Perform Birch clustering

        Parameters
        ----------
        n_clusters : int
            number of clusters
        lsi_components : int
            apply LSA before the clustering algorithm
        threshold : float
            birch threshold
        max_tree_depth : {int, None}
            maximum depth of the hierarchical tree
        """
        from freediscovery.externals.birch import Birch
        pars = {'threshold': threshold, 'is_hierarchical': n_clusters is None,
                'max_tree_depth': max_tree_depth, "metric": self.metric}
        if 'lsi' not in self.pipeline:
            raise ValueError("you must use lsi with birch clustering "
                             "for scaling reasons.")

        if n_clusters is None:
            compute_labels = False
        else:
            compute_labels = True

        km = Birch(n_clusters=n_clusters, threshold=threshold,
                   branching_factor=branching_factor,
                   compute_labels=compute_labels)

        return self._cluster_func(n_clusters, km, pars)

    def ward_hc(self, n_clusters, n_neighbors=10):
        """
        Perform Ward hierarchical clustering

        Parameters
        ----------
        n_clusters : int
            number of clusters
        lsi_components : int
            apply LSA before the clustering algorithm
        n_neighbors : int
            N nearest neighbors used for computing the connectivity matrix
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.neighbors import kneighbors_graph
        pars = {'n_neighbors': n_neighbors, 'is_hierarchical': True,
                "metric": self.metric}
        if 'lsi' not in self.pipeline:
            raise ValueError("you must use lsi with birch clustering "
                             "for scaling reasons.")

        # This is really not efficient as
        # it's done a second time in _cluster_func
        X = self.pipeline.data
        connectivity = kneighbors_graph(X, n_neighbors=n_neighbors,
                                        include_self=False)

        km = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',
                                     connectivity=connectivity)

        return self._cluster_func(n_clusters, km, pars)

    def dbscan(self, n_clusters=None, eps=0.5, min_samples=10,
               algorithm='auto', leaf_size=30):
        """
        Perform DBSCAN clustering

        This can also be used for Duplicate Detection (when ep

        Parameters
        ----------
        n_clusters : int
            number of clusters # not used just present for compatibility
        lsi_components : int
            apply LSA before the clustering algorithm
        eps : float
            The maximum distance between two samples for them to be considered
             as in the same neighborhood.
        min_samples : int
            The number of samples (or total weight) in a neighborhood
            for a point to be considered as a core point.
            This includes the point itself.
        """
        from sklearn.cluster import DBSCAN
        pars = {'is_hierarchical': False, "metric": self.metric}

        km = DBSCAN(eps=eps, min_samples=min_samples, algorithm=algorithm,
                    leaf_size=leaf_size)

        return self._cluster_func(n_clusters, km, pars)

    def scores(self, ref_labels, labels):
        """
        Parameters
        ----------
        ref_labels : list,
            reference labels
        labels : list,
            computed labels
        """
        from sklearn.metrics import (v_measure_score, adjusted_rand_score)
        out = {}
        out['adjusted_rand_score'] = adjusted_rand_score(ref_labels, labels)
        out['v_measure_score'] = v_measure_score(ref_labels, labels)
        return out
