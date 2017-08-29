# -*- coding: utf-8 -*-

from sklearn.externals import joblib
import numpy as np

from freediscovery.engine.base import _BaseWrapper
from freediscovery.utils import setup_model
from freediscovery.exceptions import WrongParameter
from freediscovery.engine.cluster import _BaseClusteringWrapper


class _DuplicateDetectionWrapper(_BaseWrapper, _BaseClusteringWrapper):
    """Find near duplicates in a document collection.

    Currently supported backends are simhash-py and i-match.

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

    _wrapper_type = "dupdet"

    def __init__(self, cache_dir='/tmp/', parent_id=None, mid=None):

        super(_DuplicateDetectionWrapper, self).__init__(cache_dir=cache_dir,
                                                 parent_id=parent_id, mid=mid,
                                                 load_model=True)

        self.model = self.cmod
        del self.cmod

    def fit(self, method='simhash'):
        """
        Precompute all the required values for duplicate detection
        """

        pars = {'method': method}
        if method not in ['simhash', 'i-match']:
            raise WrongParameter('Dup. detection method {} not implemented!'.format(method))
        if method == 'simhash':
            from freediscovery.near_duplicates import SimhashNearDuplicates
            self.model = shash = SimhashNearDuplicates()
        else:
            self.model = None
        self._pars = pars
        mid, mid_dir = setup_model(self.model_dir)

        self.mid = mid

        X = self.pipeline.data
        if method == 'simhash':
            shash.fit(X)

        self._fit_X = X

        joblib.dump(self.model, str(self.model_dir / mid / 'model'))
        joblib.dump(pars, str(self.model_dir / mid / 'pars'))

    def query(self, **args):
        """ Find all the nearests neighbours for the dataset

        Parameters
        ----------
        distance : int, default=1
            Maximum number of differnet bits in the simhash
        blocks : int or 'auto', default='auto'
            number of blocks into which the simhash is split
            when searching for duplicates,
            see  https://github.com/seomoz/simhash-py
        Returns
        -------
        cluster_id : array
            the exact duplicates (documents with the same simhash)
            are grouped by in cluster_id
        """

        if self._pars['method'] == 'simhash':
            from freediscovery.cluster.utils import (_binary_linkage2clusters,
                                                     _merge_clusters)

            shash = self.model

            if 'distance' not in args:
                args['distance'] = 1

            _fit_shash, cluster_id_exactdup, matches = shash.query(**args)

            if matches.shape[0] > 0:
                # found some near duplicates
                matches_idx = np.zeros(matches.shape, dtype=np.int)
                # match the hash value to the document index
                for (i, j), value in np.ndenumerate(matches):
                    matches_idx[i, j] = shash.get_index_by_hash(value)
            else:
                matches_idx = matches
            # compute cluster_id for near duplicates
            cluster_id_dnup = _binary_linkage2clusters(matches_idx, len(_fit_shash))
            # merge near duplicates and exact duplicates clusters
            cluster_id = _merge_clusters(
                     np.concatenate((cluster_id_exactdup[:, None],
                                     cluster_id_dnup[:, None]), axis=1),
                     rename=True)
        elif self._pars['method'] == 'i-match':
            from freediscovery.near_duplicates import IMatchNearDuplicates
            if not hasattr(self, '_fit_X'):
                self._fit_X = joblib.load(str(self.fe.dsid_dir / 'features'))
            model = IMatchNearDuplicates(**args)
            model.fit(self._fit_X)
            cluster_id = model.labels_

        else:
            raise ValueError

        return cluster_id
