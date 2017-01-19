# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path
import os
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

def _prepare_data_ingestion(data_dir, metadata, file_pattern, dir_pattern):
    """ Do some preprocessing of the input parameters
    of FeatureVectorizer for data ingestion

    Parmaters
    ---------
    data_dir : str
        path to the data directory (used only if metadata not provided), default: None
    metadata : list of dicts
        a list of dictionaries with keys ['file_path', 'document_id', 'rendition_id']
        describing the data ingestion (this overwrites data_dir)

    Returns
    -------
    filenames : list
      a list of filenames
    metadata : a pandas.DataFrame
      with at least keys 'filanames', optionally 'document_id' and 'rendition_id'
    """

    if data_dir is not None:
        data_dir = os.path.normpath(data_dir)

        if not os.path.exists(data_dir):
            raise NotFound('data_dir={} does not exist'.format(data_dir))

        filenames = _list_filenames(data_dir, dir_pattern, file_pattern)
    else:
        metadata = sorted(metadata, key=lambda x: x['file_path'])
        filenames = [el['file_path'] for el in metadata]
        data_dir = None

        if not filenames: # no files were found
            raise WrongParameter('No files to process were found!')

    return data_dir, filenames, metadata
