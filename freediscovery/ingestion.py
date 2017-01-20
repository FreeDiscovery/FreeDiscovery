# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path
import os
import pandas as pd
from .exceptions import (DatasetNotFound, InitException, NotFound, WrongParameter)

def _list_filenames(data_dir, dir_pattern=None, file_pattern=None):
    """ List all files in a data_dir"""
    import re
    # parse all files in the folder
    filenames = []
    for root, subdirs, files in os.walk(data_dir):
        if dir_pattern is None or re.match(dir_pattern, root):
            for fname in files:
                if file_pattern is None or re.match(file_pattern, fname):
                    filenames.append(os.path.normpath(os.path.join(root, fname)))

    # make sure that sorting order is deterministic
    return sorted(filenames)


class DocumentIndex(object):
    def __init__(self, data_dir, data, filenames):
        self.data_dir = data_dir
        self.data = data
        self.filenames = filenames

    @classmethod
    def from_list(cls, metadata):
        """ Create a DocumentIndex from a list of dictionaries, for instance
        {
            document_id: 1
            rendition_id: 4
            file_path: c:\dev\1.txt
        }

        Parmaters
        ---------
        metadata : list of dicts
            a list of dictionaries with keys ['file_path', 'document_id', 'rendition_id']
            describing the data ingestion (this overwrites data_dir)

        Returns
        -------
        result : DocumentIndex
            a DocumentIndex object
        """

        metadata = sorted(metadata, key=lambda x: x['file_path'])
        filenames = [el['file_path'] for el in metadata]

        data_dir = os.path.commonprefix(filenames)
        data_dir = os.path.normpath(data_dir)

        if not os.path.exists(data_dir) and os.path.exists(os.path.dirname(data_dir)):
            data_dir = os.path.dirname(data_dir)
        else:
            raise IOError('data_dir={} does not exist!'.format(data_dir))

        if not filenames: # no files were found
            raise WrongParameter('No files to process were found!')
        filenames_rel = [os.path.relpath(el, data_dir) for el in filenames]

        # modify the metadata list inplace
        for idx, (db_el, file_path) in enumerate(zip(metadata, filenames_rel)):
            db_el['file_path'] = file_path
            db_el['internal_id'] = idx
        db = pd.DataFrame(metadata)

        return cls(data_dir, db, filenames)

    @classmethod
    def from_folder(cls, data_dir, file_pattern=None, dir_pattern=None):
        """ Create a DocumentIndex from files in data_dir

        Parmaters
        ---------
        data_dir : str
            path to the data directory (used only if metadata not provided), default: None

        Returns
        -------
        result : DocumentIndex
            a DocumentIndex object
        """

        data_dir = os.path.normpath(data_dir)

        if not os.path.exists(data_dir):
            raise NotFound('data_dir={} does not exist'.format(data_dir))

        filenames = _list_filenames(data_dir, dir_pattern, file_pattern)
        filenames_rel = [os.path.relpath(el, data_dir) for el in filenames]
        db = [{'file_path': file_path, 'internal_id': idx} \
                            for idx, file_path in enumerate(filenames_rel)]

        db = pd.DataFrame(db)

        return cls(data_dir, db, filenames)
