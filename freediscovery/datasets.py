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

import numpy as np
from .base import PipelineFinder


def load_dataset(name='treclegal09_2k_subset', cache_dir='/tmp',
                 force=False, verbose=True,
                 load_ground_truth=False, verify_checksum=False):
    """ Download a benchmark dataset.

    The currently supported datasets are listed below,

    1. TREC 2009 legal collection

       - treclegal09_2k_subset  :   2 400 documents,   2 MB
       - treclegal09_20k_subset :  20 000 documents,  30 MB
       - treclegal09_37k_subset :  37 000 documents,  55 MB
       - treclegal09            : 700 000 documents, 1.2 GB

       The ground truth files for categorization are adapted from TAR Toolkit.

    2. Fedora mailing list (2009-2009)
       - fedora_ml_

    If you encounter any issues for downloads with this function,
    you can also manually download and extract the required dataset to `cache_dir` (the
    download url is `http://r0h.eu/d/<name>.tar.gz`), then re-run this function to get
    the required metadata.

    Parameters
    ----------
    name : str, default='treclegal09_2k_subset'
       the name of the dataset file to load
    cache_dir : str, default='/tmp/'
       root directory where to save the download
    force : bool, default=False
       download again if the dataset already exists.
       Warning: this will remove previously downloaded files!
    load_ground_truth : bool, default=False
       parse the ground truth files present in the dataset
    verbose : bool, default=False
       print download progress
    verify_checksum : bool, default=False
       verify the checksum of the downloaded archive

    Returns
    -------

    response: dict
       a dictionary containing paths to the dataset and corresponding metadata
    """
    import tarfile
    import requests

    VALID_MD5SUM = {'treclegal09_2k_subset' : '8090cc55ac18fe5c4d5d53d82fc767a2',
                    'treclegal09_20k_subset': '43a711897ce724e873bdbc47a374a57e',
                    'treclegal09_37k_subset': '9fb6b7505871bbaee5a438de3b0f497c',
                    'legal09int': 'None',
                    'fedora_ml_3k_subset': '09dbb03d13b8e341bd615ce43f2d836b'
                    }

    DATASET_SIZE = {'treclegal09_2k_subset' : 2.8,
                    'treclegal09_20k_subset': 30,
                    'treclegal09_37k_subset': 55,
                    'legal09int': 1500,
                    'fedora_ml_3k_subset': 3,
                    }

    if name not in VALID_MD5SUM:
        raise ValueError('Dataset name {} not known!'.format(name))


    base_url = "http://r0h.eu/d/{}.tar.gz".format(name)

    # make sure we don't have "ediscovery_cache" in the path
    cache_dir = PipelineFinder._normalize_cachedir(cache_dir)
    cache_dir = os.path.dirname(cache_dir)
    

    outdir = os.path.join(cache_dir, name)
    fname = outdir + ".tar.gz"

    if os.path.exists(outdir) and force:
        shutil.rmtree(outdir)

    # Download the the dataset if it doesn't exist
    if not os.path.exists(outdir):

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



    results = {'base_dir': outdir, 'data_dir': os.path.join(outdir, 'data')}
    if name == 'legal09int':
        results['data_dir'] = results['base_dir']


    if load_ground_truth and 'treclegal09' in name:
        with open(os.path.join(outdir,'seed_relevant.txt'), 'rt') as fh:
            relevant_files = [el.strip() for el in fh.readlines()]

        with open(os.path.join(outdir,'seed_non_relevant.txt'), 'rt') as fh:
            non_relevant_files = [el.strip() for el in fh.readlines()]

        ground_truth_file = os.path.join(outdir, "ground_truth_file.txt")  

        if platform.system() == 'Windows':
            relevant_files = [el.replace('/', '\\') for el in relevant_files]
            non_relevant_files = [el.replace('/', '\\') for el in non_relevant_files]

        results['seed_filenames'] = relevant_files + non_relevant_files 
        results['seed_y'] = list(np.concatenate((np.ones(len(relevant_files)),
                                            np.zeros(len(non_relevant_files)))).astype('int'))
        results['ground_truth_file'] = ground_truth_file

    return results





