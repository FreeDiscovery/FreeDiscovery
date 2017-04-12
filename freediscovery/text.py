# -*- coding: utf-8 -*-

import os.path
import shutil
import pickle
import warnings

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.pipeline import make_pipeline

from ._version import __version__
from .pipeline import PipelineFinder
from .utils import generate_uuid, _rename_main_thread
from .ingestion import DocumentIndex
from .stop_words import _StopWordsWrapper
from .exceptions import (DatasetNotFound, InitException, WrongParameter)


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
        pars['binary'] = False  # need to apply TFIDF weights first

    hash_opts = {key: vals for key, vals in pars.items()
                 if key in ['stop_words', 'n_features',
                            'binary', 'analyser', 'ngram_range']}
    fe = HashingVectorizer(input='filename', norm=None, decode_error='ignore',
                           non_negative=True, **hash_opts)
    if pretend:
        return fe
    fset_new = fe.transform(filenames[mslice])

    fset_new.eliminate_zeros()

    joblib.dump(fset_new, os.path.join(dsid_dir, 'features-{:05}'.format(k)))


class FeatureVectorizer(object):
    """Extract features from text documents

    Parameters
    ----------
    cache_dir : str, default='/tmp/'
        directory where to save temporary and regression files
    dsid : str
        load an exising dataset
    verbose : bool
        pring progress messages

    """

    _PARS_SHORT = ['data_dir', 'n_samples', 'n_features',
                   'n_jobs', 'chunk_size', 'norm',
                   'analyzer', 'ngram_range', 'stop_words',
                   'use_idf', 'sublinear_tf', 'binary', 'use_hashing']

    _wrapper_type = "vectorizer"

    def __init__(self, cache_dir='/tmp/', dsid=None, verbose=False):
        self.verbose = verbose

        self._filenames = None
        self._vect = None
        self._db = None
        self._pars = None

        self.cache_dir = cache_dir = PipelineFinder._normalize_cachedir(cache_dir)
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        self.dsid = dsid
        if dsid is not None:
            dsid_dir = os.path.join(self.cache_dir, dsid)
            if not os.path.exists(dsid_dir):
                raise DatasetNotFound('Dataset '
                                      '{} ({}) not found in {}!'.format(
                                       dsid, type(self).__name__, cache_dir))
        else:
            dsid_dir = None
        self.dsid_dir = dsid_dir

    @property
    def n_samples_(self):
        """ Number of documents in the dataset """
        return self.pars_['n_samples']

    @property
    def filenames_(self):
        """ Lazily load the list of filenames if needed """
        if not hasattr(self, '_filenames') or self._filenames is None:
            with open(os.path.join(self.dsid_dir, 'filenames'), 'rb') as fh:
                self._filenames = pickle.load(fh)
        return self._filenames

    @property
    def pars_(self):
        if not hasattr(self, '_pars') or self._pars is None:
            # Load parameters from disk
            dsid = self.dsid
            if self.cache_dir is None:
                raise InitException('cache_dir is None: cannot load from cache!')
            dsid_dir = os.path.join(self.cache_dir, dsid)
            if not os.path.exists(dsid_dir):
                raise DatasetNotFound('dsid {} not found!'.format(dsid))
            with open(os.path.join(dsid_dir, 'pars'), 'rb') as fh:
                self._pars = pickle.load(fh)
        return self._pars

    @property
    def vect_(self):
        if not hasattr(self, '_vect') or self._vect is None:
            mid = self.dsid
            mid_dir = os.path.join(self.cache_dir, mid)
            if not os.path.exists(mid_dir):
                raise ValueError('Vectorizer model id {} ({}) not found in the cache {}!'.format(
                                 mid, mid_dir))
            fname = os.path.join(mid_dir, 'vectorizer')
            if self.pars_['use_hashing']:
                self._vect = joblib.load(fname)
            else:
                # this is much faster in python 3 as cpickle is used
                # (only works if no numpy arrays are used)
                with open(fname, 'rb') as fh:
                    self._vect = pickle.load(fh)
        return self._vect

    def _load_features(self, dsid=None):
        """ Load a computed features from disk"""
        if self.cache_dir is None:
            raise InitException('cache_dir is None: cannot load from cache!')
        if dsid is None:
            dsid = self.dsid
        dsid_dir = os.path.join(self.cache_dir, dsid)
        if not os.path.exists(dsid_dir):
            raise DatasetNotFound('dsid not found!')
        fset_new = joblib.load(os.path.join(dsid_dir, 'features'))
        return fset_new

    def preprocess(self, data_dir=None, file_pattern='.*', dir_pattern='.*',
                   dataset_definition=None, n_features=None,
                   chunk_size=5000, analyzer='word', ngram_range=(1, 1),
                   stop_words=None, n_jobs=1, use_idf=False, sublinear_tf=True,
                   binary=False, use_hashing=False,
                   norm='l2', min_df=0.0, max_df=1.0,
                   parse_email_headers=False):
        """Initalize the features extraction.

        See sklearn.feature_extraction.text for a detailed description
        of the input parameters

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
            raise ValueError('At least one of data_dir, dataset_definition '
                             'must be provided')
        data_dir = db.data_dir

        if analyzer not in ['word', 'char', 'char_wb']:
            raise WrongParameter('analyzer={} not supported!'.format(analyzer))

        if not isinstance(ngram_range, tuple) \
           and not isinstance(ngram_range, list):
            raise WrongParameter('not a valid input ngram_range='
                                 '{}: should be a list or a typle!'.format(ngram_range))

        if not len(ngram_range) == 2:
            raise WrongParameter('len(gram_range=={}!=2'.format(len(ngram_range)))

        if norm != 'l2':
            warnings.warn("the use of 'l2' norm is stronly advised;"
                          "distance calculations"
                          " may not be correct with other normalizations."
                          " Currently norm={}".format(norm))

        if stop_words in [None, 'english', 'english_alphanumeric']:
            pass
        elif stop_words in _StopWordsWrapper(cache_dir=self.cache_dir):
            pass
        else:
            raise WrongParameter('stop_words = {}'.format(stop_words))

        if n_features is None and use_hashing:
            n_features = 100001  # default size of the hashing table

        self._filenames = db.data.file_path.values.tolist()
        del db.data['file_path']
        self.dsid = dsid = generate_uuid()
        self.dsid_dir = dsid_dir = os.path.join(self.cache_dir, dsid)

        # hash collision, should not happen
        if os.path.exists(dsid_dir):
            shutil.rmtree(dsid_dir)

        os.mkdir(dsid_dir)
        pars = {'data_dir': data_dir,
                'n_samples': len(self._filenames), "n_features": n_features,
                'chunk_size': chunk_size, 'stop_words': stop_words,
                'analyzer': analyzer, 'ngram_range': ngram_range,
                'n_jobs': n_jobs, 'use_idf': use_idf,
                'sublinear_tf': sublinear_tf,
                'binary': binary, 'use_hashing': use_hashing,
                'norm': norm, 'min_df': min_df, 'max_df': max_df,
                'parse_email_headers': parse_email_headers,
                'type': type(self).__name__,
                'freediscovery_version': __version__}
        self._pars = pars
        with open(os.path.join(dsid_dir, 'pars'), 'wb') as fh:
            pickle.dump(self._pars, fh)
        if 'file_path' in db.data.columns:
            del db.data['file_path']
        db.data.to_pickle(os.path.join(dsid_dir, 'db'))
        with open(os.path.join(self.dsid_dir, 'filenames'), 'wb') as fh:
            pickle.dump(self._filenames, fh)
        self._db = db
        return dsid

    @staticmethod
    def _generate_stop_words(stop_words):
        # from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
        # import string
        # from itertools import product
        if stop_words in [None]:
            return None
        elif stop_words == 'english':
            return stop_words
        # elif stop_words == 'english_alphanumeric':
        #    stop_words_list = list(ENGLISH_STOP_WORDS)
        #    stop_words_list += [''.join(i) for i in product(
        #                        string.ascii_lowercase + string.digits,
        #                         repeat=2)]
        #    return stop_words_list
        else:
            raise ValueError

    def parse_email_headers(self):
        from email.parser import Parser
        from .externals.jwzthreading import Message
        features = []
        for idx, fname in enumerate(self.filenames_abs_):
            with open(fname, 'rt') as fh:
                txt = fh.read()
                message = Parser().parsestr(txt, headersonly=True)

                msg_obj = Message(message, message_idx=idx)

                features.append(msg_obj)
        joblib.dump(features, os.path.join(self.dsid_dir, 'email_metadata'))
        return features

    @property
    def filenames_abs_(self):
        """ Return the absolute path to filenames """
        data_dir = self.pars_['data_dir']
        return [os.path.join(data_dir, el) for el in self.filenames_]


    def transform(self):
        """
        Run the feature extraction
        """
        from glob import glob

        dsid_dir = self.dsid_dir
        if not os.path.exists(dsid_dir):
            raise DatasetNotFound()

        pars = self.pars_
        pars['filenames_abs'] = self.filenames_abs_
        chunk_size = pars['chunk_size']
        n_samples = pars['n_samples']
        use_hashing = pars['use_hashing']

        if use_hashing:
            # just make sure that we can initialize the vectorizer
            # (easier outside of the paralel loop
            vect = _vectorize_chunk(dsid_dir, 0, pars, pretend=True)

        processing_lock = os.path.join(dsid_dir, 'processing')
        _touch(processing_lock)
        custom_sw = _StopWordsWrapper(cache_dir=self.cache_dir)
        if pars['stop_words'] in custom_sw:
            pars['stop_words'] = custom_sw.load(pars['stop_words'])
        else:
            pars['stop_words'] = self._generate_stop_words(pars['stop_words'])

        try:
            if use_hashing:
                # fixed in https://github.com/joblib/joblib/pull/414
                _rename_main_thread()
                Parallel(n_jobs=pars['n_jobs'])(
                            delayed(_vectorize_chunk)(dsid_dir, k, pars)
                            for k in range(n_samples//chunk_size + 1))

                res = self._aggregate_features()

                if pars['use_idf']:
                    tfidf = TfidfTransformer(norm=pars['norm'], use_idf=True,
                                             sublinear_tf=pars['sublinear_tf'])
                    res = tfidf.fit_transform(res)
                    vect = make_pipeline(vect, tfidf)
                self._vect = vect
            else:
                opts_tfidf = {key: val for key, val in pars.items()
                              if key in ['stop_words', 'use_idf',
                                         'ngram_range', 'analyzer',
                                         'min_df', 'max_df']}

                tfidf = TfidfVectorizer(input='filename',
                                        max_features=pars['n_features'],
                                        norm=pars['norm'],
                                        decode_error='ignore', **opts_tfidf)
                res = tfidf.fit_transform(pars['filenames_abs'])
                self._vect = tfidf
            fname = os.path.join(dsid_dir, 'vectorizer')
            if self._pars['use_hashing']:
                joblib.dump(self._vect, fname)
            else:
                # faster for pure python objects
                with open(fname, 'wb') as fh:
                    pickle.dump(self._vect, fh)

            if pars['norm'] is not None:
                res = normalize(res, norm=pars['norm'], copy=False)
            else:
                # scale feature to [0, 1]
                # this is necessary e.g. by SVM
                # and does not hurt anyway
                res /= res.max()

            del self.pars_['filenames_abs']

            joblib.dump(res, os.path.join(dsid_dir, 'features'))
            # remove all identical files
            if use_hashing:
                for filename in glob(os.path.join(dsid_dir,
                                                  'features-*[0-9]*')):
                    os.remove(filename)
        except:
            if os.path.exists(processing_lock):
                os.remove(processing_lock)
            raise
        # remove processing lock if finished or if error
        if os.path.exists(processing_lock):
            os.remove(processing_lock)
        _touch(os.path.join(dsid_dir, 'processing_finished'))

    def append(self, dataset_definition):
        """ Add some files to the dataset

        This is by no mean an efficient operation, processing all the files
        at once might be more suitable in most occastions.
        """
        from .lsi import _LSIWrapper
        dsid_dir = self.dsid_dir
        db_old = self.db_.data
        internal_id_offset = db_old.internal_id.max()
        data_dir = self.pars_['data_dir']
        db_extra = DocumentIndex.from_list(dataset_definition, data_dir,
                                           internal_id_offset + 1)
        db_new = db_extra.data
        vect = self.vect_

        filenames_abs = [os.path.join(data_dir, el)
                         for el in db_new.file_path.values]

        # write down the new features file
        X_new = vect.transform(filenames_abs)
        X_old = self._load_features()
        X = scipy.sparse.vstack((X_new, X_old))
        joblib.dump(X, os.path.join(dsid_dir, 'features'))

        # write down the new filenames file
        filenames_old = list(self.filenames_)
        filenames_new = list(db_new.file_path.values)
        filenames = filenames_old + filenames_new
        with open(os.path.join(dsid_dir, 'filenames'), 'wb') as fh:
            pickle.dump(filenames, fh)
        del self._filenames
        del db_new['file_path']

        # write down the new database file
        db = pd.concat((db_old, db_new))
        db.to_pickle(os.path.join(dsid_dir, 'db'))
        del self._db

        # write down the new pars file
        self._pars = self.pars_
        self._pars['n_samples'] = len(filenames)
        with open(os.path.join(dsid_dir, 'pars'), 'wb') as fh:
            pickle.dump(self._pars, fh)

        # find all exisisting LSI models and update them as well
        if os.path.exists(os.path.join(dsid_dir, 'lsi')):
            for lsi_id in os.listdir(os.path.join(dsid_dir, 'lsi')):
                lsi_obj = _LSIWrapper(cache_dir=self.cache_dir,
                                      mid=lsi_id)
                lsi_obj.append(X_new)

    @property
    def n_features_(self):
        """ Number of features of the vecotorizer"""
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.pipeline import Pipeline
        vect = self.vect_
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

        X = self._load_features()
        return _query_features(self.vect_, X, indices, n_top_words,
                               remove_stop_words)

    def _aggregate_features(self):
        """ Agregate features loaded as separate files features-<number>
        into a single file features"""
        from glob import glob
        out = []
        for filename in sorted(glob(os.path.join(self.dsid_dir, 'features-*[0-9]'))):
            ds = joblib.load(filename)
            out.append(ds)
        res = scipy.sparse.vstack(out)
        return res

    @property
    def db_(self):
        """ DatasetIndex """
        if not hasattr(self, '_db') or self._db is None:
            dsid = self.dsid
            if self.cache_dir is None:
                raise InitException('cache_dir is None: cannot load from cache!')
            dsid_dir = os.path.join(self.cache_dir, dsid)
            if not os.path.exists(dsid_dir):
                raise DatasetNotFound('dsid {} not found!'.format(dsid))
            data = pd.read_pickle(os.path.join(dsid_dir, 'db'))
            self._db = DocumentIndex(self.pars_['data_dir'], data)
        return self._db

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
        return np.asarray(self.filenames_)[index]

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
                pars = self.pars_
            except:
                # traceback.print_exc()
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
        a list of filenames in self._pars['filenames']
        (with possible duplicates) used to downsample / upsample the dataset

    """
    def __init__(self, cache_dir='/tmp/', dsid=None, verbose=False,
                 sampling_filenames=None):
        super(_FeatureVectorizerSampled, self).__init__(cache_dir, dsid,
                                                        verbose)

        if sampling_filenames is not None:
            self.sampling_filenames = sampling_filenames
            if dsid is None:
                raise ValueError('sampling_index must be applied '
                                 'onto an existing dataset\n'
                                 'specified by dsid')
            if not isinstance(sampling_filenames, list):
                raise TypeError('Wrong type {} for sampling_index,'
                                ' must be list'.format(
                                    type(sampling_filenames).__name__))

            self.db_.filenames_ = super(_FeatureVectorizerSampled, self).filenames_
            self.sampling_index = self.db_._search_filenames(self.sampling_filenames)
        else:
            self.sampling_filenames = None
            self.sampling_index = None

    @property
    def pars_(self):
        pars = super(_FeatureVectorizerSampled, self).pars_
        if self.sampling_filenames is not None:
            pars['n_samples'] = self.n_samples_
            pars['additional_sampling'] = True
        else:
            pars['additional_sampling'] = False
        return pars

    def _load_features(self, dsid):
        X = super(_FeatureVectorizerSampled, self)._load_features(dsid)
        if self.sampling_filenames is not None:
            return X[self.sampling_index, :]
        else:
            return X

    @property
    def filenames_(self):
        fnames = super(_FeatureVectorizerSampled, self).filenames_
        fnames = np.array(fnames)
        if self.sampling_filenames is not None:
            fnames = fnames[self.sampling_index]
        return fnames

    @property
    def n_samples_(self):
        if self.sampling_filenames is not None:
            return len(self.sampling_filenames)
        else:
            return super(_FeatureVectorizerSampled, self).n_samples_
