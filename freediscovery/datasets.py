# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os
import sys
import shutil
import hashlib
import platform
import random

import numpy as np
import pandas as pd
from .base import PipelineFinder
from sklearn.externals import joblib

VALID_MD5SUM = {'treclegal09_2k_subset' : '8090cc55ac18fe5c4d5d53d82fc767a2',
                'treclegal09_20k_subset': '43a711897ce724e873bdbc47a374a57e',
                'treclegal09_37k_subset': '9fb6b7505871bbaee5a438de3b0f497c',
                'legal09int': 'None',
                'fedora_ml_3k_subset': '09dbb03d13b8e341bd615ce43f2d836b',
                '20newsgroups_3categories': '7e59e10cbd824190f3f1fa82285c7865',
                '20newsgroups_micro': 'f6ec5e8669ebde1efa11148096c7cc0c'
                }

DATASET_SIZE = {'treclegal09_2k_subset' : 2.8,
                'treclegal09_20k_subset': 30,
                'treclegal09_37k_subset': 55,
                'legal09int': 1500,
                'fedora_ml_3k_subset': 3,
                '20newsgroups_3categories': 1,
                '20newsgroups_micro': 0.03
                }

def _download_dataset(cache_dir, fname, name, verify_checksum, verbose):
    """ Internal helper function to download datasets"""
    import tarfile
    import requests

    base_url = "http://r0h.eu/d/{}.tar.gz".format(name)

    if verbose:
        print('\nWarning: downloading dataset {} ({} MB) !'.format(name,
                                    DATASET_SIZE[name]))
    response = requests.get(base_url, stream=False, allow_redirects=True)
    with open(fname, "wb") as fh:
        for idx, chunk in enumerate(response.iter_content(chunk_size=1024)):
            if chunk:
                fh.write(chunk)
        if verbose:
            print('\nFile {} downloaded!'.format(fname))

    if verify_checksum:
        # compute the md5 hash by chunks
        with open(fname, 'rb') as fh:
            block_size=2**20
            md5 = hashlib.md5()
            while True:
                data = fh.read(block_size)
                if not data:
                    break
                md5.update(data)
            hash_val = md5.hexdigest()
        if hash_val != VALID_MD5SUM[name]:
            raise IOError('Checksum failed for the dataset, this may be due'
                          'to a corrupted download. Try running this function'
                          'again with the `force=True` option.')

    # extract the .tar.gz
    with tarfile.open(fname, "r:gz") as tar:
        tar.extractall(path=cache_dir)
        if verbose:
            print('Archive extracted!'.format(fname))


def _load_erdm_ground_truth(outdir):
    """A helper function to load Legal TREC 2009 data"""
    with open(os.path.join(outdir,'seed_relevant.txt'), 'rt') as fh:
        relevant_files = [el.strip() for el in fh.readlines()]

    with open(os.path.join(outdir,'seed_non_relevant.txt'), 'rt') as fh:
        non_relevant_files = [el.strip() for el in fh.readlines()]


    if platform.system() == 'Windows':
        relevant_files = [el.replace('/', '\\') for el in relevant_files]
        non_relevant_files = [el.replace('/', '\\') for el in non_relevant_files]
    return non_relevant_files, relevant_files

def _compute_document_id(internal_id, mode):
    if mode == 'squared':
        return internal_id**2
    else:
        raise NotImplementedError


def filter_dict(d, valid_keys):
    return [{key: row[key] for key in row if key in valid_keys} for row in d]


def load_dataset(name='20newsgroups_3categories', cache_dir='/tmp',
                 force=False, verbose=False, verify_checksum=False,
                 document_id_generation='squared', categories=None
                 ):
    """
    Download a benchmark dataset.

    The currently supported datasets are listed below,

    1. TREC 2009 legal collection

       - `treclegal09_2k_subset`  :   2 400 documents,   2 MB
       - `treclegal09_20k_subset` :  20 000 documents,  30 MB
       - `treclegal09_37k_subset` :  37 000 documents,  55 MB
       - `treclegal09`            : 700 000 documents, 1.2 GB

       The ground truth files for categorization are adapted from TAR Toolkit.

    2. Fedora mailing list (2009-2009)
       - `fedora_ml`

    3. The 20 newsgoups dataset
       - `20newsgroups_3categories`: only the ['comp.graphics', 'rec.sport.baseball', 'sci.space'] categories

    If you encounter any issues for downloads with this function,
    you can also manually download and extract the required dataset to `cache_dir` (the
    download url is `http://r0h.eu/d/<name>.tar.gz`), then re-run this function to get
    the required metadata.

    Parameters
    ----------
    name : str, default='20newsgroups_3categories'
       the name of the dataset file to load
    cache_dir : str, default='/tmp/'
       root directory where to save the download
    force : bool, default=False
       download again if the dataset already exists.
       Warning: this will remove previously downloaded files!
    verbose : bool, default=False
       print download progress
    verify_checksum : bool, default=False
       verify the checksum of the downloaded archive
    document_id_generation : str
       specifies how the document_id is computed from internal_id
       must be one of ['identity', 'squared']
       default="identity" (i.e. document_id = internal_id)
    categories : str
       select a subsection of the dataset, default='all'

    Returns
    -------

    metadata: dict
       a dictionary containing metadata corresponding to the dataset
    training_set : {dict, None}
       a list of dictionaries for the training set
    test_set : dict
       a list of dictionaries for the test set
    """
    from .ingestion import DocumentIndex
    from .io import parse_ground_truth_file

    if name not in VALID_MD5SUM:
        raise ValueError('Dataset name {} not known!'.format(name))

    valid_fields = ['document_id', 'internal_id', 'file_path', 'category']

    has_categories = '20newsgroups' in name or 'treclegal09' in name


    # make sure we don't have "ediscovery_cache" in the path
    cache_dir = PipelineFinder._normalize_cachedir(cache_dir)
    cache_dir = os.path.dirname(cache_dir)
    

    outdir = os.path.join(cache_dir, name)
    fname = outdir + ".tar.gz"

    if os.path.exists(outdir) and force:
        shutil.rmtree(outdir)

    if '20newsgroups' in name:
        internal_data_dir = os.path.dirname(os.path.abspath(__file__))
        internal_data_dir = os.path.join(internal_data_dir, 'data')
        if name == '20newsgroups_3categories':
            fname = name + '.pkl.xz'
        else:
            fname = name + '.pkl'

        twenty_news = joblib.load(os.path.join(internal_data_dir, fname))

    # Download the dataset if it doesn't exist
    if not os.path.exists(outdir):
        if '20newsgroups' in name:
            os.mkdir(outdir)
            for idx, doc in enumerate(twenty_news.data):
                with open(os.path.join(outdir, '{:05}.txt'.format(idx)), 'wt') as fh:
                    fh.write(doc)
        else:
            _download_dataset(cache_dir, fname, name, verify_checksum, verbose)

    if 'treclegal09' in name or 'fedora_ml' in name:
        data_dir = os.path.join(outdir, 'data')
    else:
        data_dir = outdir
    md = {'data_dir': data_dir, 'name': name}

    di = DocumentIndex.from_folder(data_dir)

    training_set = None

    if 'treclegal09' in name:
            negative_files, positive_files = _load_erdm_ground_truth(outdir)

            ground_truth_file = os.path.join(outdir, "ground_truth_file.txt")  
            gt = parse_ground_truth_file(ground_truth_file)

            res = di.search(gt, drop=False)
            di.data['category'] = res.is_relevant
            di.data['category'] = di.data['category'].apply(
                                      lambda x: 'positive' if x == 1 else 'negative')
            di.data['is_train'] = False
            res = di.search(pd.DataFrame({'file_path': positive_files + negative_files}))
            di.data.loc[res.internal_id.values, 'is_train'] = True
    elif '20newsgroups' in name:
        di.data['category'] = np.array(twenty_news.target_names)[twenty_news.target]
        di.data['is_train'] = ['-train' in el for el in twenty_news.filenames]

    if categories is not None and has_categories:
        mask = di.data.category.isin(categories)
        di.data = di.data[mask]
        di.data['internal_id'] = np.arange(len(di.data['internal_id']))

    di.data.set_index('internal_id', drop=False, inplace=True)

    di.data['document_id'] = _compute_document_id(di.data['internal_id'],
                                                  document_id_generation)
    di.data = di.data.astype('object')

    if has_categories:
        mask = di.data['is_train']
        training_set = di.render_dict(di.data[mask],
                             return_file_path=True)
        training_set = filter_dict(training_set, valid_fields)
        if name == '20newsgroups_3categories':
            # make a smaller training set
            random.seed(999998)
            training_set = random.sample(training_set,
                                         min(len(training_set), di.data.shape[0] // 5))

    dataset = di.render_dict(return_file_path=True)

    dataset = filter_dict(dataset, valid_fields)

    return md, training_set, dataset





