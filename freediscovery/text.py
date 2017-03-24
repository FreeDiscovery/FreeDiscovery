# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import shutil
import numpy as np

from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.pipeline import make_pipeline

from .pipeline import PipelineFinder
from .utils import generate_uuid, _rename_main_thread
from .ingestion import DocumentIndex
from .stop_words import _StopWordsWrapper
from .exceptions import (DatasetNotFound, InitException, NotFound, WrongParameter)


def _touch(filename):
    open(filename, 'ab').close()


def _vectorize_chunk(dsid_dir, k, pars, pretend=False):
    """ Extract features on a chunk of files """
    import os.path
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.externals import joblib

    filenames = pars['filenames_abs']
    chunk_size = pars['chunk_size']
    n_samples = pars['n_samples']

    mslice = slice(k*chunk_size, min((k+1)*chunk_size, n_samples))

    if pars['use_idf']:
        pars['binary'] = False # need to apply TFIDF weights first 

    hash_opts = {key: vals for key, vals in pars.items() \
            if key in ['stop_words', 'n_features', 'binary', 'analyser', 'ngram_range']}
    fe = HashingVectorizer(input='filename', norm=None, decode_error='ignore',
           non_negative=True, **hash_opts) 
    if pretend:
        return fe
    fset_new = fe.transform(filenames[mslice])

    fset_new.eliminate_zeros()

    joblib.dump(fset_new, os.path.join(dsid_dir, 'features-{:05}'.format(k)),
            compress=0)

class _BaseTextTransformer(object):
    """Base text transformer

    Parameters
    ----------
    cache_dir : str, default='/tmp/'
        directory where to save temporary and regression files
    dsid : str
        load an exising dataset
    verbose : bool
        pring progress messages
    """

    def __init__(self, cache_dir='/tmp/', dsid=None, verbose=False):
        self.data_dir = None
        self.verbose = verbose


        self.cache_dir = cache_dir = PipelineFinder._normalize_cachedir(cache_dir)
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        self.dsid = dsid
        if dsid is not None:
            dsid_dir = os.path.join(self.cache_dir, dsid)
            if not os.path.exists(dsid_dir):
                raise DatasetNotFound('Dataset {} ({}) not found in {}!'.format(
                                            dsid, type(self).__name__, cache_dir))
            pars = self._load_pars()
            if hasattr(self, '_load_db'):
                self.db = self._load_db()
        else:
            dsid_dir = None
            pars = None
            if hasattr(self, '_load_db'):
                self.db = None
        self.dsid_dir = dsid_dir
        self._pars = pars

    def delete(self):
        """ Delete the current dataset """
        import shutil
        shutil.rmtree(self.dsid_dir, ignore_errors=True)

    def __contains__(self, dsid):
        """ This is a somewhat non standard call that checks if a dsid
        exist on disk (in general)"""
        dsid_dir = os.path.join(self.cache_dir, dsid)
        return os.path.exists(dsid_dir)

    def __getitem__(self, index):
        return np.asarray(self._pars['filenames'])[index]

    def load(self, dsid=None):
        """ Load a computed features from disk"""
        if self.cache_dir is None:
            raise InitException('cache_dir is None: cannot load from cache!')
        if dsid is None:
            dsid = self.dsid
        dsid_dir = os.path.join(self.cache_dir, dsid)
        if not os.path.exists(dsid_dir):
            raise DatasetNotFound('dsid not found!')
        pars = joblib.load(os.path.join(dsid_dir, 'pars'))
        fset_new = joblib.load(os.path.join(dsid_dir, 'features'))
        return pars['filenames'], fset_new

    def get_params(self):
        """ Get the vectorizer parameters """
        return self._pars

    def _load_pars(self):
        """ Load parameters from disk"""
        dsid = self.dsid
        if self.cache_dir is None:
            raise InitException('cache_dir is None: cannot load from cache!')
        dsid_dir = os.path.join(self.cache_dir, dsid)
        if not os.path.exists(dsid_dir):
            raise DatasetNotFound('dsid {} not found!'.format(dsid))
        pars = joblib.load(os.path.join(dsid_dir, 'pars'))
        return pars

    def _load_model(self):
        mid = self.dsid
        mid_dir = os.path.join(self.cache_dir, mid)
        if not os.path.exists(mid_dir):
            raise ValueError('Vectorizer model id {} ({}) not found in the cache {}!'.format(
                             mid, mid_dir))
        cmod = joblib.load(os.path.join(mid_dir, 'vectorizer'))
        return cmod


    @property
    def n_samples_(self):
        """ Number of documents in the dataset """
        return len(self._pars['filenames'])


    def list_datasets(self):
        """ List all datasets in the working directory """
        import traceback
        out = []
        for dsid in os.listdir(self.cache_dir):
            if dsid == 'stop_words':
                continue
            row = {"id": dsid}
            self.dsid = dsid
            try:
                pars = self._load_pars()
            except:
                #traceback.print_exc()
                continue

            if pars['type'] != type(self).__name__:
                continue

            try:
                for key in self._PARS_SHORT:
                    row[key] = pars[key]
                out.append(row)
            except Exception:
                print(pars.keys())
                traceback.print_exc()

        return out



class FeatureVectorizer(_BaseTextTransformer):
    """Extract features from text documents

    Parameters
    ----------
    cache_dir : str, default='/tmp/'
        directory where to save temporary and regression files
    dsid : str
        load an exising dataset
    verbose : bool
        pring progress messages
    sampling_index : list of filenames or None, default=None
        a list of filenames used for upsampling / downsampling

    """

    _PARS_SHORT = ['data_dir', 'n_samples', 'n_features',
                   'n_jobs', 'chunk_size', 'norm',
                   'analyzer', 'ngram_range', 'stop_words',
                   'use_idf', 'sublinear_tf', 'binary', 'use_hashing']

    _wrapper_type = "vectorizer"

    def preprocess(self, data_dir=None, file_pattern='.*', dir_pattern='.*',
            dataset_definition=None, n_features=None,
            chunk_size=5000, analyzer='word', ngram_range=(1, 1), stop_words=None,
            n_jobs=1, use_idf=False, sublinear_tf=True, binary=False, use_hashing=False,
            norm='l2', min_df=0.0, max_df=1.0):
        """Initalize the features extraction.

        See sklearn.feature_extraction.text for a detailed description of the input parameters

        Parameters
        ----------
        data_dir : str
            path to the data directory (used only if metadata not provided), default: None
        dataset_defintion : list of dicts
            a list of dictionaries with keys ['file_path', 'document_id', 'rendition_id']
            describing the data ingestion (this overwrites data_dir)
        analyzer : string, {'word', 'char'} or callable
            Whether the feature should be made of word or character n-grams.
            If a callable is passed it is used to extract the sequence of features
            out of the raw, unprocessed input.
        ngram_range : tuple (min_n, max_n)
            The lower and upper boundary of the range of n-values for different
            n-grams to be extracted. All values of n such that min_n <= n <= max_n
            will be used.
        stop_words : string {'english'}, list, or None (default)
            If a string, it is passed to _check_stop_list and the appropriate stop
            list is returned. 'english' is currently the only supported string
            value.
            If None, no stop words will be used. max_df can be set to a value
            in the range [0.7, 1.0) to automatically detect and filter stop
            words based on intra corpus document frequency of terms.
        max_df : float in range [0.0, 1.0] or int, default=1.0
            When building the vocabulary ignore terms that have a document
            frequency strictly higher than the given threshold (corpus-specific
            stop words).
            If float, the parameter represents a proportion of documents, integer
            absolute counts.
            This parameter is ignored if vocabulary is not None.
        min_df : float in range [0.0, 1.0] or int, default=1
            When building the vocabulary ignore terms that have a document
            frequency strictly lower than the given threshold. This value is also
            called cut-off in the literature.
            If float, the parameter represents a proportion of documents, integer
            absolute counts.
            This parameter is ignored if vocabulary is not None.
        max_features : int or None, default=None or 100001
            If not None, build a vocabulary that only consider the top
            max_features ordered by term frequency across the corpus.
        binary : boolean, default=False
            If True, all non-zero term counts are set to 1. This does not mean
            outputs will have only 0/1 values, only that the tf term in tf-idf
            is binary. (Set idf and normalization to False to get 0/1 outputs.)
        norm : 'l1', 'l2' or None, optional
            Norm used to normalize term vectors. None for no normalization.
        use_idf : boolean, default=True
            Enable inverse-document-frequency reweighting.

        """

        if dataset_definition is not None:
            db = DocumentIndex.from_list(dataset_definition)
        elif data_dir is not None:
            db = DocumentIndex.from_folder(data_dir, file_pattern, dir_pattern)
        else:
            raise ValueError('At least one of data_dir, dataset_definition must be provided')
        data_dir = db.data_dir

        self.data_dir = data_dir
        if analyzer not in ['word', 'char', 'char_wb']:
            raise WrongParameter('analyzer={} not supported!'.format(analyzer))

        if not isinstance(ngram_range, tuple) and not isinstance(ngram_range, list):
            raise WrongParameter('not a valid input ngram_range={}: should be a list or a typle!'.format(ngram_range))

        if not len(ngram_range) == 2:
            raise WrongParameter('len(gram_range=={}!=2'.format(len(ngram_range)))

        if stop_words in [None, 'english', 'english_alphanumeric']:
            pass
        elif stop_words in _StopWordsWrapper(cache_dir=self.cache_dir):
            pass
        else:
            raise WrongParameter('stop_words = {}'.format(stop_words))

        if n_features is None and use_hashing:
            n_features = 100001 # default size of the hashing table

        filenames_rel = db.data.file_path.values.tolist()
        self.dsid = dsid = generate_uuid()
        self.dsid_dir = dsid_dir = os.path.join(self.cache_dir, dsid)

        # hash collision, should not happen
        if os.path.exists(dsid_dir):
            shutil.rmtree(dsid_dir)

        os.mkdir(dsid_dir)
        pars = {'filenames': filenames_rel, 'data_dir': data_dir,
                'n_samples': len(filenames_rel), "n_features": n_features,
                'chunk_size': chunk_size, 'stop_words': stop_words,
                'analyzer': analyzer, 'ngram_range': ngram_range,
                'n_jobs': n_jobs, 'use_idf': use_idf, 'sublinear_tf': sublinear_tf,
                'binary': binary, 'use_hashing': use_hashing,
                'norm': norm, 'min_df': min_df, 'max_df': max_df,
                'type': type(self).__name__
               }
        self._pars = pars
        joblib.dump(pars, os.path.join(dsid_dir, 'pars'), compress=9)
        joblib.dump(db, os.path.join(dsid_dir, 'db'), compress=9)
        self.db = db
        return dsid

    @staticmethod
    def _generate_stop_words(stop_words):
        from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
        import string
        from itertools import product
        if stop_words in [None]:
            return None
        elif stop_words == 'english':
            return stop_words
        #elif stop_words == 'english_alphanumeric':
        #    stop_words_list = list(ENGLISH_STOP_WORDS)
        #    stop_words_list += [''.join(i) for i in product(
        #                                string.ascii_lowercase + string.digits, repeat=2)]
        #    return stop_words_list
        else:
            raise ValueError

    def transform(self):
        """
        Run the feature extraction
        """
        from glob import glob

        dsid = self.dsid
        dsid_dir = self.dsid_dir
        if not os.path.exists(dsid_dir):
            raise DatasetNotFound()

        pars = self._pars
        filenames_base = pars['filenames']
        data_dir = pars['data_dir']
        pars['filenames_abs'] = [os.path.join(data_dir, el) for el in filenames_base]
        chunk_size = pars['chunk_size']
        n_samples = pars['n_samples']
        use_hashing = pars['use_hashing']

        if use_hashing:
            # just make sure that we can initialize the vectorizer
            # (easier outside of the paralel loop
            vect = _vectorize_chunk(dsid_dir, 0, pars, pretend=True)

        processing_lock =  os.path.join(dsid_dir, 'processing')
        _touch(processing_lock)
        custom_sw = _StopWordsWrapper(cache_dir=self.cache_dir)
        if pars['stop_words'] in custom_sw:
            pars['stop_words'] = custom_sw.load(pars['stop_words'])
        else:
            pars['stop_words'] = self._generate_stop_words(pars['stop_words'])

        try:
            if use_hashing:
                _rename_main_thread() # fixed in https://github.com/joblib/joblib/pull/414
                Parallel(n_jobs=pars['n_jobs'])(delayed(_vectorize_chunk)(dsid_dir, k, pars)\
                            for k in range(n_samples//chunk_size + 1))

                res = self._aggregate_features()

                if pars['use_idf']:
                    tfidf = TfidfTransformer(norm=pars['norm'], use_idf=True,
                                              sublinear_tf=pars['sublinear_tf'])
                    res = tfidf.fit_transform(res)
                    vect = make_pipeline(vect, tfidf)
                self.vect = vect
            else:
                opts_tfidf = {key: val for key, val in pars.items() \
                        if key in ['stop_words', 'use_idf', 'ngram_range', 'analyzer',
                                   'min_df', 'max_df']}

                tfidf = TfidfVectorizer(input='filename',
                            max_features=pars['n_features'],
                            norm=pars['norm'],
                            decode_error='ignore', **opts_tfidf)
                res = tfidf.fit_transform(pars['filenames_abs'])
                self.vect = tfidf
            joblib.dump(self.vect, os.path.join(dsid_dir, 'vectorizer'))

            if pars['norm'] is not None:
                res = normalize(res, norm=pars['norm'], copy=False)
            else:
                # scale feature to [0, 1]
                # this is necessary e.g. by SVM
                # and does not hurt anyway
                res /= res.max()

            joblib.dump(res, os.path.join(dsid_dir, 'features'))
            # remove all identical files
            if use_hashing:
                for filename in glob(os.path.join(dsid_dir, 'features-*[0-9]*')):
                    os.remove(filename)
        except:
            if os.path.exists(processing_lock):
                os.remove(processing_lock)
            raise
        # remove processing lock if finished or if error
        if os.path.exists(processing_lock):
            os.remove(processing_lock)
        _touch(os.path.join(dsid_dir, 'processing_finished'))
        return dsid, filenames_base


    @property
    def n_features_(self):
        """ Number of features of the vecotorizer"""
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.pipeline import Pipeline
        vect = self._load_model()
        if hasattr(vect, 'vocabulary_'):
            return len(vect.vocabulary_)
        elif isinstance(vect, HashingVectorizer):
            return vect.get_params()['n_features']
        elif isinstance(vect, Pipeline):
            return vect.named_steps['hashingvectorizer'].get_params()['n_features']
        else:
            raise ValueError

    def query_features(self, indices, n_top_words=10, remove_stop_words=False):
        """ Query the features with most weight

        Parameters
        ----------
        indices : list or ndarray
          indices for the subcluster
        n_top_words : int
          the number of workds to return
        remove_stop_words : bool
          remove stop words
        """
        from .utils import _query_features

        X = joblib.load(os.path.join(self.dsid_dir, 'features'))
        return _query_features(self.vect, X, indices, n_top_words, remove_stop_words)


    def _aggregate_features(self):
        """ Agregate features loaded as separate files features-<number>
        into a single file features"""
        from glob import glob
        import scipy.sparse
        out = []
        for filename in sorted(glob(os.path.join(self.dsid_dir, 'features-*[0-9]'))):
            ds = joblib.load(filename)
            out.append(ds)
        res = scipy.sparse.vstack(out)
        return res

    def _load_db(self):
        """ Load DatasetIndex from disk"""
        dsid = self.dsid
        if self.cache_dir is None:
            raise InitException('cache_dir is None: cannot load from cache!')
        dsid_dir = os.path.join(self.cache_dir, dsid)
        if not os.path.exists(dsid_dir):
            raise DatasetNotFound('dsid {} not found!'.format(dsid))
        db = joblib.load(os.path.join(dsid_dir, 'db'))
        return db



class _FeatureVectorizerSampled(FeatureVectorizer):
    """Extract features from text documents, with additional sampling

    This is mostly a helper class for debugging and experimenting

    Parameters
    ----------
    cache_dir : str, default='/tmp/'
        directory where to save temporary and regression files
    dsid : str
        load an exising dataset
    verbose : bool
        pring progress messages
    sampling_index : list of filenames or None, default=None
        a list of filenames in self._pars['filenames'] (with possible duplicates)
        used to downsample / upsample the dataset

    """
    def __init__(self, cache_dir='/tmp/', dsid=None, verbose=False, sampling_filenames=None):
        super(_FeatureVectorizerSampled, self).__init__(cache_dir, dsid, verbose)

        if sampling_filenames is not None:
            self.sampling_filenames = sampling_filenames
            if dsid is None:
                raise ValueError('sampling_index must be applied onto an existing dataset\n'
                                 'specified by dsid')
            if not isinstance(sampling_filenames, list):
                raise TypeError('Wrong type {} for sampling_index, must be list'.format(
                            type(sampling_filenames).__name__))
            self.sampling_index = self.db._search_filenames(self.sampling_filenames)
        else:
            self.sampling_filenames = None
            self.sampling_index = None

    def _load_pars_sampled(self):
        pars = super(_FeatureVectorizerSampled , self)._load_pars()
        if self.sampling_filenames is not None:
            pars['filenames'] = self.sampling_filenames
            pars['n_samples'] = self.n_samples_
            pars['additional_sampling'] = True
        else:
            pars['additional_sampling'] = False
        return pars

    def load(self, dsid):
        fnames, X = super(_FeatureVectorizerSampled , self).load(dsid)
        if self.sampling_filenames is not None:
            return (np.array(fnames)[self.sampling_index].tolist(),
                    X[self.sampling_index, :])
        else:
            return fnames, X

    @property
    def n_samples_(self):
        if self.sampling_filenames is not None:
            return len(self.sampling_filenames)
        else:
            return super(_FeatureVectorizerSampled , self).n_samples_
