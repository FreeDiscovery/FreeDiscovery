# -*- coding: utf-8 -*-

import os.path
import shutil
import pickle
import warnings
import time

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer

from freediscovery._version import __version__
from freediscovery.engine.pipeline import PipelineFinder
from freediscovery.engine.stop_words import _StopWordsWrapper
from freediscovery.engine.ingestion import DocumentIndex
from freediscovery.utils import generate_uuid, _rename_main_thread
from freediscovery.feature_weighting import SmartTfidfTransformer, _validate_smart_notation
from freediscovery.preprocessing import processing_filters
from freediscovery.exceptions import (DatasetNotFound, InitException, WrongParameter)
from freediscovery.engine.utils import validate_mid


def _touch(filename):
    filename.open('ab').close()


def _read_file(file_path):
    with open(file_path, 'rb') as fh:
        doc = fh.read()
    doc = doc.decode('utf-8', 'ignore')
    return doc


def _preprocess_stream(doc, steps=None):
    """ Apply pre-processing steps """

    if steps:
        for key in steps:
            func = processing_filters[key]
            doc = func(doc)

    return doc


def _vectorize_chunk(dsid_dir, k, pars, pretend=False):
    """ Extract features on a chunk of files """
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.externals import joblib

    filenames = pars['filenames_abs']
    chunk_size = pars['chunk_size']
    n_samples = pars['n_samples']

    mslice = slice(k*chunk_size, min((k+1)*chunk_size, n_samples))

    hash_opts = {key: vals for key, vals in pars.items()
                 if key in ['stop_words', 'n_features',
                            'analyser', 'ngram_range']}
    hash_opts['alternate_sign'] = False
    fe = HashingVectorizer(input='content', norm=None, **hash_opts)
    if pretend:
        return fe
    fset_new = fe.transform(_read_file(fname) for fname in filenames[mslice])

    fset_new.eliminate_zeros()

    joblib.dump(fset_new, str(dsid_dir / 'features-{:05}'.format(k)))


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
    mode : str
        write or read mode for the FeatureVectorizer
    """

    _PARS_SHORT = ['data_dir', 'n_samples', 'n_features',
                   'n_jobs', 'chunk_size',
                   'analyzer', 'ngram_range', 'stop_words',
                   'weighting', 'norm_alpha', 'use_hashing',
                   'creation_date']

    _wrapper_type = "vectorizer"

    def __init__(self, cache_dir='/tmp/', dsid=None, verbose=False, mode='r'):
        self.verbose = verbose

        self._filenames = None
        self._vect = None
        self._tfidf = None
        self._db = None
        self._pars = None

        self.cache_dir = cache_dir = PipelineFinder._normalize_cachedir(cache_dir)
        if not cache_dir.exists():
            cache_dir.mkdir()
        self.dsid = dsid
        if mode not in ['r', 'w', 'fw']:
            raise WrongParameter('mode={} must be one of "r", "w", "fw"'
                                 .format(mode))
        self.mode = mode
        if dsid is not None:
            validate_mid(dsid)
            dsid_dir = self.cache_dir / dsid
            if mode == 'r':
                if not dsid_dir.exists():
                    raise DatasetNotFound('Dataset '
                                          '{} ({}) not found in {}!'.format(
                                           dsid, type(self).__name__, cache_dir))
            else:
                if dsid_dir.exists():
                    if mode == 'w':
                        raise WrongParameter(('dataset identified by dsid={} '
                                              'already exists. Use mode="fw" '
                                              'to overwrite.')
                                             .format(dsid))
                    elif mode == 'fw':
                        shutil.rmtree(dsid_dir)
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
            with (self.dsid_dir / 'filenames').open('rb') as fh:
                self._filenames = pickle.load(fh)
        return self._filenames

    @property
    def pars_(self):
        if not hasattr(self, '_pars') or self._pars is None:
            # Load parameters from disk
            dsid = self.dsid
            if self.cache_dir is None:
                raise InitException('cache_dir is None: '
                                    'cannot load from cache!')
            dsid_dir = self.cache_dir / dsid
            if not dsid_dir.exists():
                raise DatasetNotFound('dsid {} not found!'.format(dsid))
            with (dsid_dir / 'pars').open('rb') as fh:
                self._pars = pickle.load(fh)
        return self._pars

    @property
    def vect_(self):
        if not hasattr(self, '_vect') or self._vect is None:
            mid = self.dsid
            mid_dir = self.cache_dir / mid
            if not mid_dir.exists():
                raise ValueError(('Vectorizer model id {} ({}) '
                                  'not found in the cache {}!')
                                 .format(mid, mid_dir))
            fname = mid_dir / 'vectorizer'
            if self.pars_['use_hashing']:
                self._vect = joblib.load(str(fname))
            else:
                # this is much faster in python 3 as cpickle is used
                # (only works if no numpy arrays are used)
                with fname.open('rb') as fh:
                    self._vect = pickle.load(fh)
        return self._vect

    @property
    def tfidf_(self):
        if not hasattr(self, '_tfidf') or self._tfidf is None:
            mid = self.dsid
            mid_dir = self.cache_dir / mid
            if not mid_dir.exists():
                raise ValueError(('Vectorizer model id {} ({}) '
                                  'not found in the cache {}!')
                                 .format(mid, mid_dir))
            fname = mid_dir / 'tfidf_transformer'
            self._tfidf = joblib.load(str(fname))
        return self._tfidf

    def _load_features(self, dsid=None):
        """ Load a computed features from disk"""
        if self.cache_dir is None:
            raise InitException('cache_dir is None: cannot load from cache!')
        if dsid is None:
            dsid = self.dsid
        dsid_dir = self.cache_dir / dsid
        if not dsid_dir.exists():
            raise DatasetNotFound('dsid not found!')
        fset_new = joblib.load(str(dsid_dir / 'features'))
        return fset_new

    def setup(self, n_features=None, chunk_size=5000, analyzer='word',
              ngram_range=(1, 1), stop_words=None, n_jobs=1,
              use_hashing=False,
              weighting='nnc', norm_alpha=0.75, min_df=0.0, max_df=1.0,
              parse_email_headers=False,
              preprocess=[]):
        """Initalize the features extraction.

        See sklearn.feature_extraction.text for a detailed description
        of the input parameters

        Parameters
        ----------
        analyzer : string, {'word', 'char'} or callable
            Whether the feature should be made of word or character n-grams.
            If a callable is passed it is used to extract the sequence of
            features out of the raw, unprocessed input.
        ngram_range : tuple (min_n, max_n)
            The lower and upper boundary of the range of n-values for different
            n-grams to be extracted. All values of n such that
            min_n <= n <= max_n will be used.
        stop_words : string {'english'}, list, or None (default)
            If a string, it is passed to _check_stop_list and the appropriate
            stop list is returned. 'english' is currently the only supported
            string value.
            If None, no stop words will be used. max_df can be set to a value
            in the range [0.7, 1.0) to automatically detect and filter stop
            words based on intra corpus document frequency of terms.
        max_df : float in range [0.0, 1.0] or int, default=1.0
            When building the vocabulary ignore terms that have a document
            frequency strictly higher than the given threshold (corpus-specific
            stop words).
            If float, the parameter represents a proportion of documents,
            integer absolute counts.
            This parameter is ignored if vocabulary is not None.
        min_df : float in range [0.0, 1.0] or int, default=1
            When building the vocabulary ignore terms that have a document
            frequency strictly lower than the given threshold. This value is
            also called cut-off in the literature.
            If float, the parameter represents a proportion of documents,
            integer absolute counts.
            This parameter is ignored if vocabulary is not None.
        max_features : int or None, default=None or 100001
            If not None, build a vocabulary that only consider the top
            max_features ordered by term frequency across the corpus.
        weighting : str
            SMART weighting type
        preprocess : list of strings, default: []
            A list of pre-processing steps, including 'emails_ingore_header'
        """
        if self.mode not in ['w', 'fw']:
            raise WrongParameter('The vectorizer can be setup only with '
                                 'mode in ["w", "fw"]')

        if analyzer not in ['word', 'char', 'char_wb']:
            raise WrongParameter('analyzer={} not supported!'.format(analyzer))

        if not isinstance(ngram_range, tuple) \
           and not isinstance(ngram_range, list):
            raise WrongParameter(('not a valid input ngram_range='
                                  '{}: should be a list or a typle!')
                                 .format(ngram_range))

        if not len(ngram_range) == 2:
            raise WrongParameter('len(gram_range=={}!=2'
                                 .format(len(ngram_range)))

        if not 0 <= norm_alpha <= 1:
            raise WrongParameter('norm_alpha={} not in [0, 1]'
                                 .format(norm_alpha))

        _, _, weighting_n = _validate_smart_notation(weighting)
        if weighting_n == 'n':
            warnings.warn('You should use either cosine or pivoted normalization '
                          'i.e. weighting should be "**[cp]"',
                          UserWarning)

        for key in preprocess:
            if key not in processing_filters:
                raise WrongParameter(('Unknown preprocessing step {} '
                                      ' must of be of {}')
                                     .format(key, ', '.join(list(processing_filters.keys()))))

        if stop_words in [None, 'english', 'english_alphanumeric']:
            pass
        elif stop_words in _StopWordsWrapper(cache_dir=self.cache_dir):
            pass
        else:
            raise WrongParameter('stop_words = {}'.format(stop_words))

        if n_features is None and use_hashing:
            n_features = 100001  # default size of the hashing table

        if self.dsid is None:
            self.dsid = dsid = generate_uuid()
        else:
            dsid = self.dsid
        self.dsid_dir = dsid_dir = self.cache_dir / dsid

        dsid_dir.mkdir()

        pars = {'data_dir': None,
                'n_samples': None, "n_features": n_features,
                'chunk_size': chunk_size, 'stop_words': stop_words,
                'analyzer': analyzer, 'ngram_range': ngram_range,
                'n_jobs': n_jobs, 'use_hashing': use_hashing,
                'weighting': weighting, 'norm_alpha': norm_alpha,
                'min_df': min_df, 'max_df': max_df,
                'parse_email_headers': parse_email_headers,
                'type': type(self).__name__,
                'preprocess': preprocess,
                'freediscovery_version': __version__}
        self._pars = pars
        with (dsid_dir / 'pars').open('wb') as fh:
            pickle.dump(self._pars, fh)
        return dsid

    def ingest(self, data_dir=None, file_pattern='.*', dir_pattern='.*',
               dataset_definition=None, vectorize=True,
               document_id_generator='indexed_file_path'):
        """Perform data ingestion

        Parameters
        ----------
        data_dir : str
            path to the data directory (used only if metadata not provided),
            default: None
        dataset_defintion : list of dicts
            a list of dictionaries with keys
            ['file_path', 'document_id', 'rendition_id']
            describing the data ingestion (this overwrites data_dir)
        vectorize : bool (default: True)
        """
        dsid_dir = self.cache_dir / self.dsid
        if (dsid_dir / 'db').exists():
            raise ValueError('Dataset {} already vectorized!'
                             .format(self.dsid))
        db_list = list(sorted(dsid_dir.glob('db*')))
        if len(db_list) == 0:
            internal_id_offset = -1
        elif len(db_list) >= 1:
            internal_id_offset = int(db_list[-1].name[3:])

        if dataset_definition is not None:
            db = DocumentIndex.from_list(dataset_definition, data_dir,
                                         internal_id_offset + 1, dsid_dir,
                                         document_id_generator=document_id_generator)
        elif data_dir is not None:
            db = DocumentIndex.from_folder(data_dir, file_pattern, dir_pattern,
                                           internal_id_offset + 1,
                                           document_id_generator=document_id_generator)
        else:
            db = None

        if db is not None:
            data_dir = db.data_dir

            batch_suffix = '.{:09}'.format(db.data.internal_id.iloc[-1])

            self._filenames = db.data.file_path.values.tolist()
            del db.data['file_path']

            pars = self.pars_

            if 'file_path' in db.data.columns:
                del db.data['file_path']
            db.data.to_pickle(str(dsid_dir / ('db' + batch_suffix)))
            with (dsid_dir / ('filenames' + batch_suffix)).open('wb') as fh:
                pickle.dump(self._filenames, fh)
            self._db = db

        if vectorize:
            db_list = list(sorted(dsid_dir.glob('db*')))
            filenames_list = list(sorted(dsid_dir.glob('filenames*')))
            if len(db_list) == 0:
                raise ValueError('No ingested files found!')

            if len(db_list) == 1:
                with filenames_list[0].open('rb') as fh:
                    filenames_concat = pickle.load(fh)
            elif len(db_list) >= 2:
                # accumulate different batches into a single file
                # filename file
                filenames_concat = []
                for fname in filenames_list:
                    with fname.open('rb') as fh:
                        filenames_concat += pickle.load(fh)

            if self.pars_['data_dir'] is None:
                data_dir = DocumentIndex._detect_data_dir(filenames_concat)
                self._pars['data_dir'] = data_dir
            else:
                data_dir = self._pars['data_dir']

            self._filenames = [os.path.relpath(el, data_dir)
                               for el in filenames_concat]

            with (dsid_dir / 'filenames').open('wb') as fh:
                pickle.dump(self._filenames, fh)

            for fname in filenames_list:
                fname.unlink()

            # save databases
            if len(db_list) == 1:
                db_list[0].rename(dsid_dir / 'db')
                self.db_.filenames_ = self._filenames
                self.db_.data['file_path'] = self._filenames
            elif len(db_list) >= 2:

                db_concat = []
                for fname in db_list:
                    db_concat.append(pd.read_pickle(str(fname)))
                db_new = pd.concat(db_concat, axis=0)
                db_new.filenames_ = self._filenames
                db_new.set_index('internal_id', drop=False, inplace=True)
                self._db = DocumentIndex(data_dir, db_new)
                if 'file_path' in db_new.columns:
                    del db_new['file_path']
                db_new.to_pickle(str(dsid_dir / 'db'))

            # save parameters
            self._pars['n_samples'] = len(self._filenames)
            self._pars['data_dir'] = data_dir

            with (dsid_dir / 'pars').open('wb') as fh:
                pickle.dump(self._pars, fh)

            self.transform()

            if (dsid_dir / 'raw').exists():
                shutil.rmtree(str(dsid_dir / 'raw'))

        if db is None and not vectorize:
            raise ValueError('At least one of data_dir, dataset_definition, '
                             'vectorize parameters must be provided!')
        return

    @staticmethod
    def _generate_stop_words(stop_words):
        if stop_words in [None]:
            return None
        elif stop_words == 'english':
            return stop_words
        else:
            raise ValueError

    def parse_email_headers(self):
        from email.parser import Parser
        from freediscovery.externals.jwzthreading import Message
        features = []
        for idx, fname in enumerate(self.filenames_abs_):
            txt = _read_file(fname)
            message = Parser().parsestr(txt, headersonly=True)

            msg_obj = Message(message, message_idx=idx)

            features.append(msg_obj)
        joblib.dump(features, str(self.dsid_dir / 'email_metadata'))
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
        dsid_dir = self.dsid_dir
        if not dsid_dir.exists():
            raise DatasetNotFound()

        if not (dsid_dir / 'db').exists():
            raise ValueError('Please ingest some files before running '
                             'the vectorizer!')

        pars = self.pars_
        pars['filenames_abs'] = self.filenames_abs_
        chunk_size = pars['chunk_size']
        n_samples = pars['n_samples']
        use_hashing = pars['use_hashing']

        if use_hashing:
            # make sure that we can initialize the vectorizer
            # (easier outside of the paralel loop
            vect = _vectorize_chunk(dsid_dir, 0, pars, pretend=True)

        processing_lock = (dsid_dir / 'processing')
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

                self._vect = vect
            else:
                opts_tfidf = {key: val for key, val in pars.items()
                              if key in ['stop_words',
                                         'ngram_range', 'analyzer',
                                         'min_df', 'max_df']}

                vect = CountVectorizer(input='content',
                                       max_features=pars['n_features'],
                                       **opts_tfidf)
                text_gen = (_preprocess_stream(_read_file(fname), pars['preprocess'])
                            for fname in pars['filenames_abs'])
                res = vect.fit_transform(text_gen)
                self._vect = vect
            fname = dsid_dir / 'vectorizer'
            if self._pars['use_hashing']:
                joblib.dump(self._vect, str(fname))
            else:
                # faster for pure python objects
                with fname.open('wb') as fh:
                    pickle.dump(self._vect, fh)
            fname = dsid_dir / 'tfidf_transformer'
            wt = SmartTfidfTransformer(pars['weighting'],
                                       norm_alpha=pars['norm_alpha'])
            self._idf = wt
            res = wt.fit_transform(res)
            joblib.dump(self._idf, str(fname))

            del self.pars_['filenames_abs']

            joblib.dump(res, str(dsid_dir / 'features'))
            # remove all identical files
            if use_hashing:
                for filename in dsid_dir.glob('features-*[0-9]*'):
                    filename.unlink()
        except:
            if processing_lock.exists():
                processing_lock.unlink()
            raise
        # remove processing lock if finished or if error
        if processing_lock.exists():
            processing_lock.unlink()
        _touch(dsid_dir / 'processing_finished')

    def append(self, dataset_definition, data_dir=None):
        """ Add some documents to the dataset

        This is by no mean an efficient operation, processing all the files
        at once might be more suitable in most occastions.
        """
        from freediscovery.engine.lsi import _LSIWrapper
        dsid_dir = self.dsid_dir
        db_old = self.db_.data
        internal_id_offset = db_old.internal_id.max()
        db_extra = DocumentIndex.from_list(dataset_definition, data_dir,
                                           internal_id_offset + 1, dsid_dir)
        db_new = db_extra.data
        vect = self.vect_
        tfidf = self.tfidf_

        filenames_new = list(db_new.file_path.values)

        # write down the new features file
        X_new_raw = vect.transform(filenames_new)
        X_new = tfidf.transform(X_new_raw)
        X_old = self._load_features()
        X = scipy.sparse.vstack((X_new, X_old))
        joblib.dump(X, str(dsid_dir / 'features'))

        # write down the new filenames file
        filenames_old = list(self.filenames_)
        filenames = filenames_old + filenames_new

        data_dir = DocumentIndex._detect_data_dir(filenames)
        self._pars['data_dir'] = data_dir

        self._filenames = [os.path.relpath(el, data_dir)
                           for el in filenames]

        with (dsid_dir / 'filenames').open('wb') as fh:
            pickle.dump(self._filenames, fh)
        del db_new['file_path']

        # write down the new pars file
        self._pars = self.pars_
        self._pars['n_samples'] = len(filenames)
        with (dsid_dir / 'pars').open('wb') as fh:
            pickle.dump(self._pars, fh)

        # write down the new database file
        db = pd.concat((db_old, db_new))
        if 'file_path' in db.columns:
            del db['file_path']
        db.to_pickle(str(dsid_dir / 'db'))
        self._db = DocumentIndex(self.pars_['data_dir'], db)

        # find all exisisting LSI models and update them as well
        if (dsid_dir / 'lsi').exists():
            for lsi_id in os.listdir(str(dsid_dir / 'lsi')):
                lsi_obj = _LSIWrapper(cache_dir=self.cache_dir,
                                      mid=lsi_id)
                lsi_obj.append(X_new)

        # remove all trained models for this dataset
        for model_type in ['categorizer', 'dupdet', 'cluster', 'threading']:
            if (dsid_dir / model_type).exists():
                for mid in os.listdir(str(dsid_dir / model_type)):
                    shutil.rmtree(str(dsid_dir / model_type / mid))

    def remove(self, dataset_definition):
        """ Remove some documents from the dataset

        This is by no mean an efficient operation, processing all the files
        at once might be more suitable in most occastions.
        """
        from freediscovery.engine.lsi import _LSIWrapper
        dsid_dir = self.dsid_dir
        db_old = self.db_.data
        query = pd.DataFrame(dataset_definition)
        res = self.db_.search(query, drop=False)
        del_internal_id = res.internal_id.values
        internal_id_mask = ~np.in1d(db_old.internal_id.values, del_internal_id)

        # write down the new features file
        X_old = self._load_features()
        X = X_old[internal_id_mask, :]
        joblib.dump(X, str(dsid_dir / 'features'))

        # write down the new filenames file
        filenames = list(np.array(self.filenames_)[internal_id_mask])
        with (dsid_dir / 'filenames').open('wb') as fh:
            pickle.dump(filenames, fh)
        self._filenames = filenames

        # write down the new database file
        db = db_old.iloc[internal_id_mask].copy()
        # create a new contiguous internal_id
        db['internal_id'] = np.arange(db.shape[0], dtype='int')
        self._db = DocumentIndex(self.pars_['data_dir'], db)
        if 'file_path' in db.columns:
            del db['file_path']
        db.to_pickle(str(dsid_dir / 'db'))

        # write down the new pars file
        self._pars = self.pars_
        self._pars['n_samples'] = len(filenames)
        with (dsid_dir / 'pars').open('wb') as fh:
            pickle.dump(self._pars, fh)

        # find all exisisting LSI models and update them as well
        if (dsid_dir / 'lsi').exists():
            for lsi_id in os.listdir(str(dsid_dir / 'lsi')):
                _fname = dsid_dir / 'lsi' / lsi_id / 'data'
                if _fname.exists():
                    X_lsi_old = joblib.load(str(_fname))
                    X_lsi = X_lsi_old[internal_id_mask]
                    joblib.dump(X_lsi, str(_fname))

        # remove all trained models for this dataset
        for model_type in ['categorizer', 'dupdet', 'cluster', 'threading']:
            if (dsid_dir / model_type).exists():
                for mid in os.listdir(str(dsid_dir / model_type)):
                    shutil.rmtree(str(dsid_dir / model_type / mid))

    @property
    def n_features_(self):
        """ Number of features of the vecotorizer"""
        from sklearn.feature_extraction.text import HashingVectorizer
        vect = self.vect_
        if hasattr(vect, 'vocabulary_'):
            return len(vect.vocabulary_)
        elif isinstance(vect, HashingVectorizer):
            return vect.get_params()['n_features']
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
        from freediscovery.utils import _query_features

        X = self._load_features()
        return _query_features(self.vect_, X, indices, n_top_words,
                               remove_stop_words)

    def _aggregate_features(self):
        """ Agregate features loaded as separate files features-<number>
        into a single file features"""
        from glob import glob
        out = []
        for filename in sorted(self.dsid_dir.glob('features-*[0-9]')):
            ds = joblib.load(str(filename))
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
            dsid_dir = self.cache_dir / dsid
            if not dsid_dir.exists():
                raise DatasetNotFound('dsid {} not found!'.format(dsid))
            data = pd.read_pickle(str(dsid_dir / 'db'))
            self._db = DocumentIndex(self.pars_['data_dir'], data)
        return self._db

    def delete(self):
        """ Delete the current dataset """
        import shutil
        shutil.rmtree(self.dsid_dir, ignore_errors=True)

    def __contains__(self, dsid):
        """ This is a somewhat non standard call that checks if a dsid
        exist on disk (in general)"""
        return (self.cache_dir / dsid).exists()

    def __getitem__(self, index):
        return np.asarray(self.filenames_)[index]

    def list_datasets(self):
        """ List all datasets in the working directory """
        import traceback
        out = []
        for dsid in os.listdir(str(self.cache_dir)):
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

            creation_date = os.stat(str(self.cache_dir / dsid)).st_ctime
            pars['creation_date'] = time.strftime('%c',
                                                  time.gmtime(creation_date))

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
