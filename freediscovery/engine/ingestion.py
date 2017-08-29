# -*- coding: utf-8 -*-

import os
import warnings
import re
from collections import Counter

import numpy as np
import pandas as pd

from freediscovery.exceptions import (NotFound, WrongParameter)


def _list_filenames(data_dir, dir_pattern=None, file_pattern=None):
    """ List all files in a data_dir"""
    import re
    # parse all files in the folder
    filenames = []
    for root, subdirs, files in os.walk(data_dir):
        if dir_pattern is None or re.match(dir_pattern, root):
            for fname in files:
                if file_pattern is None or re.match(file_pattern, fname):
                    filenames.append(os.path.normpath(os.path.join(root,
                                                                   fname)))

    # make sure that sorting order is deterministic
    return sorted(filenames)


def _infer_document_id_from_path(file_path):
    basename = os.path.basename
    document_id = [re.sub('\D', '', basename(el)) for el in file_path]
    non_digits = [el for el in document_id if not el.isdigit()]
    failed_msg = ('Warning: Could not infer document_id from file_path ({}), '
                  'falling back '
                  'to `document_id_generator="indexed_file_path"`.')
    if non_digits:
        print(failed_msg.format('{} file_path do not contain any digits'
                                .format(len(non_digits))))
        return None
    document_id_counts = [(item, count)
                          for item, count in Counter(document_id).items()
                          if count > 1]
    if document_id_counts:
        print(failed_msg.format('{} document_id inferred from file_path '
                                'are non uniques'
                                .format(document_id_counts)))
        return None

    return np.array(document_id, dtype=int)


def _generate_document_id(db, document_id_generator):
    if 'document_id' not in db:
        if document_id_generator == 'infer_file_path':
            document_id = _infer_document_id_from_path(db.file_path)
            if document_id is None:
                # infering failed
                document_id_generator = 'indexed_file_path'
            else:
                db['document_id'] = document_id

        if document_id_generator == 'indexed_file_path':
            db['document_id'] = db.internal_id


def _check_mutual_index(keys1, keys2):
    """ Given two datasets with columns keys1 and keys2,
    returns the columns that could be used as an index
    """
    if 'internal_id' in keys1 and 'internal_id' in keys2:
        index_cols = ['internal_id']
    elif "document_id" in keys1 and \
         "document_id" in keys2 and \
         "rendition_id" in keys1 and \
         "rendition_id" in keys2:
        index_cols = ['document_id', 'rendition_id']
    elif "document_id" in keys1 and \
         "document_id" in keys2:
        index_cols = ['document_id']
    elif "file_path" in keys1 and \
         "file_path" in keys2:
        index_cols = ['file_path']
    else:
        raise ValueError("Cannot create a mutual index from "
                         "columns\n keys1 : {}\n keys2: {}".format(
                               list(keys1), list(keys2)))

    return index_cols


class DocumentIndex(object):
    def __init__(self, data_dir, data):
        self.data_dir = data_dir
        self.data = data

    def _check_index(self, keys=None):
        """ Given a list of keys check which keys will be used for indexing
        and whether these keys could be used for an index

        Parameters
        ----------
        keys : list
          one or multiple choices among "internal_id", "document_id",
          "rendition_id", "file_path".  default=["internal_id"]
        Returns
        -------
        index_cols : list
          a subset of keys that would be used for an index
        """

        if keys is None:
            keys = ['internal_id']

        index_cols = _check_mutual_index(keys, self.data.columns)
        if index_cols == ["document_id"] \
           and not self.data.document_id.is_unique:
            raise ValueError('document_id cannot be used as an index, '
                             'since it has duplicates '
                             ' (and rendition_id has duplicates)')

        if len(index_cols) == 1:
            index_cols = index_cols[0]

        # make sure we can use the selected columns as an index
        self.data.set_index(index_cols, verify_integrity=True)
        return index_cols

    def search(self, query, strict=True, drop=True,
               return_file_path=False):
        """Search the filenames given by some user query

        Parameters
        ----------
        query : pandas.DataFrame
           a DataFrame with one of the following fields "internal_id",
           ("document_id", "rendition_id"), "document_id", "file_path"
        strict : bool
           raise an error if some documents are not found
        drop : bool
           drop columns not in the dataset

        Returns
        -------
        df : pd.DataFrame
            the response dataframe with fields
            "internal_id", "file_path" and optionally "document_id"
            and "rendition_id"
        """
        if not isinstance(query, pd.DataFrame):
            raise ValueError('The query {} must be a pandas DataFrame')
        if not query.shape[0]:
            raise ValueError('Query has zero element!')

        if 'file_path' not in self.data.columns:
            if 'file_path' in query.columns or return_file_path:
                self.data['file_path'] = self.filenames_

        index_cols = self._check_index(query.columns)

        query['sort_order'] = query.index.values

        res = self.data.merge(query, on=index_cols, how='inner',
                              suffixes=('', '_query'))
        # make sure we preserve the original order in the query
        res.sort_values(by='sort_order', inplace=True)
        del res['sort_order']

        if res.shape[0] != query.shape[0]:
            # some documents were not found
            msg = ['{} out of {} query elements not found'
                   '\n(using "{}" as index):'.format(
                          query.shape[0] - res.shape[0],
                          query.shape[0], index_cols)]
            for index, row in query.iterrows():
                if row[index_cols] not in self.data[index_cols].values:
                    msg.append('   * {}'.format(row.to_dict()))

            msg.append('Expected format: \n {}'.format(
                       self.data.iloc[0, :].to_dict()))
            msg.append('with a total of {} documents'.format(
                       self.data.shape[0]))

            if strict:
                raise NotFound('\n'.join(msg))
            else:
                print('Warning: ' + '\n'.join(msg))

        if drop:
            # ignore all additional columns
            res = res[self.data.columns]

        return res

    def _search_filenames(self, filenames):
        """ A helper function that reproduces the previous
        behaviour in FeaturesVectorizer"""
        query = pd.DataFrame(filenames, columns=['file_path'])

        res = self.search(query)
        return res.internal_id.values

    def render_dict(self, res=None, return_file_path=False):
        """Render a pandas dataframe as a list of dicts

        Parameters
        ----------
        res : {pandas.DataFrame, None}
            some dataset with additional data that must contain
            the 'internal_id' key
        return_file_path : bool
            return the file paths, default: False

        Results
        -------
        out : dict

        """
        if res is not None:
            res = res.set_index('internal_id', drop=False)
        db = self.data.set_index('internal_id', drop=False)
        if not return_file_path:
            base_keys = [key for key in self.data.columns
                         if key != 'file_path']
        else:
            base_keys = list(self.data.columns)
            if 'file_path' not in db.columns:
                db['file_path'] = self.filenames_
                base_keys += ['file_path']

        if res is not None:
            res_keys = [key for key in res
                        if key not in base_keys]
            if not return_file_path and 'file_path' in res_keys:
                res_keys.remove('file_path')

        db = db[base_keys]

        if res is not None:
            db_dict = res[res_keys].join(db, how='left')
            out = db_dict.to_dict(orient='records')
        else:
            out = db.to_dict(orient='records')

        return out

    def render_list(self, res=None, return_file_path=False):
        """Render a pandas dataframe as a dict of lists

        Parameters
        ----------
        res : {pandas.DataFrame, None}
            some dataset with additional data that must contain
            the 'internal_id' key
        return_file_path : bool
            return the file paths, default: False

        Results
        -------
        out : dict

        """
        # TODO: this can be better rewritten with df.to_dict()
        if res is not None:
            res = res.set_index('internal_id', drop=False)
        db = self.data.set_index('internal_id', drop=False)
        if not return_file_path:
            base_keys = [key for key in self.data.columns
                         if key != 'file_path']
        else:
            base_keys = list(self.data.columns)
        db = db[base_keys]

        if res is not None:
            res_keys = [key for key in res if key not in base_keys]
            if not return_file_path:
                if 'file_path' in res_keys:
                    res_keys.remove('file_path')
        else:
            res_keys = []

        out = {}
        for key in base_keys + res_keys:
            out[key] = []
        if res is not None:
            for index, row in res[res_keys].iterrows():
                db_sel_dict = db.loc[index].to_dict()
                for key, val in db_sel_dict.items():
                    out[key].append(val)
                for key, val in row.to_dict().items():
                    out[key].append(val)
        else:
            for index, row in db.iterrows():
                row_dict = row.to_dict()
                for key, val in row_dict.items():
                    out[key].append(val)
        return out

    @classmethod
    def from_list(cls, metadata, data_dir=None, internal_id_offset=0,
                  dsid_dir=None, document_id_generator='indexed_file_path'):
        """ Create a DocumentIndex from a list of dictionaries, for instance

        .. code:: javascript

            {
                document_id: 1,
                rendition_id: 4,
                file_path: "c:\dev\1.txt"
            }

        Parmaters
        ---------
        metadata : list of dicts
            a list of dictionaries with keys
            ['file_path', 'document_id', 'rendition_id']
            describing the data ingestion (this overwrites data_dir)
        data_dir : str
            path to an exisitng data dir
        internal_id_offset : int
            the offset for internal_id (used when appending files
            to an existing dataset)
        dsid_dir : str
            In case content is provided, store it to a raw folder
            in this directory

        Returns
        -------
        result : DocumentIndex
            a DocumentIndex object
        """
        has_file_path = np.array(['file_path' in row for row in metadata])
        has_content = np.array(['content' in row for row in metadata])
        if has_file_path.all():
            if has_content.any():
                warnings.warn("Both 'file_path' and 'content' fields are "
                              "provided, the latter will be ingored!")
            metadata = sorted(metadata, key=lambda x: x['file_path'])

            # modify the metadata list inplace
            for idx, db_el in enumerate(metadata):
                db_el['internal_id'] = idx + internal_id_offset
                db_el['file_path'] = os.path.normpath(db_el['file_path'])
        elif has_content.all():
            if not (dsid_dir / 'raw').exists():
                (dsid_dir / 'raw').mkdir()
            data_dir = str(dsid_dir / 'raw')
            for idx, db_el in enumerate(metadata):
                db_el['internal_id'] = idx + internal_id_offset
                if 'rendering_id' in db_el:
                    rendering_id_el = db_el['rendering_id']
                else:
                    rendering_id_el = 0
                file_path_el = '{:09}_{}.txt'.format(idx + internal_id_offset,
                                                     rendering_id_el)
                db_el['file_path'] = os.path.join(data_dir, file_path_el)
                with (dsid_dir / 'raw' / file_path_el).open('wt', encoding='utf-8') as fh:
                    fh.write(db_el.pop('content'))

        elif (~has_file_path).any():
            raise ValueError(("All the ingested elements must have a "
                              "'file_path' or 'content' field. Currently, \n"
                              " - {} / {} documents have the 'file_path' "
                              "field,\n"
                              " - {} / {} documents have the 'content' field."
                              ).format(has_file_path.sum(), len(has_file_path),
                                       has_content.sum(), len(has_content)))

        filenames = [el['file_path'] for el in metadata]
        db = pd.DataFrame(metadata)

        _generate_document_id(db, document_id_generator)

        res = cls(data_dir, db)
        res.filenames_ = filenames
        return res

    @staticmethod
    def _detect_data_dir(filenames):
        if len(filenames) > 1:
            data_dir = os.path.commonprefix(filenames)
            data_dir = os.path.normpath(data_dir)
        elif len(filenames) == 1:
            data_dir = os.path.dirname(filenames[0])
        else:
            raise ValueError('Provided list of filenames is empty!')

        if os.path.exists(data_dir):
            return data_dir
        elif os.path.exists(os.path.dirname(data_dir)):
            return os.path.dirname(data_dir)
        else:
            raise IOError('data_dir={} does not exist!'.format(data_dir))

    @classmethod
    def from_folder(cls, data_dir, file_pattern=None, dir_pattern=None,
                    internal_id_offset=0, document_id_generator='indexed_file_path'):
        """ Create a DocumentIndex from files in data_dir

        Parmaters
        ---------
        data_dir : str
            path to the data directory (used only if metadata not provided),
            default: None

        Returns
        -------
        result : DocumentIndex
            a DocumentIndex object
        """

        data_dir = os.path.abspath(os.path.normpath(data_dir))

        if not os.path.exists(data_dir):
            raise NotFound('data_dir={} does not exist'.format(data_dir))

        if document_id_generator not in ['indexed_file_path', 'infer_file_path']:
            raise WrongParameter(("document_id_generator={} not supported. It must be "
                                  "one of ['indexed_file_path', 'infer_file_path']")
                                 .format(document_id_generator))

        filenames = _list_filenames(data_dir, dir_pattern, file_pattern)

        db = [{'file_path': file_path, 'internal_id': idx + internal_id_offset}
              for idx, file_path in enumerate(filenames)]

        db = pd.DataFrame(db)

        _generate_document_id(db, document_id_generator)

        res = cls(data_dir, db)
        res.filenames_ = filenames
        return res

    def _make_relative_paths(self):
        """ By default DocumentIndex uses absolute file paths,
        this makes the file_path to be relative to dir_path"""
        if self.data_dir is None:
            self.data_dir = self._detect_data_dir(self.filenames_)

        self.filenames_ = [os.path.relpath(el, self.data_dir)
                           for el in self.filenames_]
        self.data['file_path'] = self.filenames_
