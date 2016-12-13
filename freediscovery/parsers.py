# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path
import re
import shutil
import numpy as np

from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .base import _BaseTextTransformer
from .utils import generate_uuid, _rename_main_thread
from .exceptions import (DatasetNotFound, InitException, NotFound, WrongParameter)


class EmailParser(_BaseTextTransformer):
    """Parse emails

    Parameters
    ----------
    cache_dir : str, default='/tmp/'
        directory where to save temporary and regression files
    dsid : str
        load an exising dataset
    verbose : bool
        pring progress messages
    """

    _PARS_SHORT = ['data_dir', 'n_samples', 'encoding']

    def transform(self, data_dir, file_pattern='.*', dir_pattern='.*',
              encoding='latin-1'):
        """Parse all emails in data_dir"""
        from email.parser import Parser
        from jwzthreading import Message

        data_dir = os.path.normpath(data_dir)

        if not os.path.exists(data_dir):
            raise NotFound('data_dir={} does not exist'.format(data_dir))
        self.data_dir = data_dir

        # parse all files in the folder
        filenames = self._list_filenames(data_dir, dir_pattern, file_pattern)

        if not filenames: # no files were found
            raise WrongParameter('No files to process were found!')

        filenames_rel = [os.path.relpath(el, data_dir) for el in filenames]
        self.dsid = dsid = generate_uuid()
        self.dsid_dir = dsid_dir = os.path.join(self.cache_dir, dsid)

        # hash collision, should not happen
        if os.path.exists(dsid_dir):
            shutil.rmtree(dsid_dir)

        os.mkdir(dsid_dir)
        pars = {'filenames': filenames_rel, 'data_dir': data_dir,
                'n_samples': len(filenames_rel), "encoding": encoding,
                'type': type(self).__name__
               }
        self._pars = pars

        features = []
        for idx, fname in enumerate(filenames):
            with open(fname, 'rt') as fh:
                txt = fh.read()
                #if sys.version_info < (3, 0) and encoding != 'utf-8':
                #    message = message.encode('utf-8')
                message = Parser().parsestr(txt, headersonly=True)

                msg_obj = Message(message, message_idx=idx)

                features.append(msg_obj)


        joblib.dump(pars, os.path.join(dsid_dir, 'pars'), compress=9)
        joblib.dump(features, os.path.join(dsid_dir, 'features'), compress=9)

        #pars['filenames_abs'] = [os.path.join(data_dir, el) for el in filenames_base]
        return dsid
