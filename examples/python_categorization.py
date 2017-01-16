"""
Categorization Example [Python API]
-----------------------------------

An example of categorization using the Python API
"""

from __future__ import print_function

import numpy as np

from freediscovery.datasets import load_dataset
from freediscovery.text import FeatureVectorizer
from freediscovery.categorization import _CategorizerWrapper
from freediscovery.tests.run_suite import check_cache
from freediscovery.io import parse_ground_truth_file
from freediscovery.utils import categorization_score

dataset_name = "treclegal09_2k_subset"     # see list of available datasets

cache_dir = check_cache()

if __name__ == '__main__':

    ds = load_dataset(dataset_name, load_ground_truth=True, cache_dir=cache_dir)


    # To use a custom dataset, simply specify the following variables
    data_dir = ds['data_dir']
    seed_filenames = ds['seed_filenames']
    seed_y = ds['seed_y']
    ground_truth_file = ds['ground_truth_file']  # (optional)

    fe_opts = {'data_dir': data_dir,
               'stop_words': 'english', 'chunk_size': 2000, 'n_jobs': -1,
               'use_idf': 1, 'sublinear_tf': 0, 'binary': 0, 'n_features': 50001,
               'analyzer': 'word', 'ngram_range': (1, 1), "norm": "l2"
              }

    fe = FeatureVectorizer(cache_dir=cache_dir)

    uuid = fe.preprocess(**fe_opts)
    uuid, filenames  = fe.transform()

    seed_index = fe.search(seed_filenames)

    cat = _CategorizerWrapper(cache_dir=cache_dir, parent_id=uuid)
    cat.train(seed_index, seed_y)

    predictions, md = cat.predict()

    gt = parse_ground_truth_file( ground_truth_file)
    idx_ref = cat.fe.search(gt.index.values)
    idx_res = np.arange(cat.fe.n_samples_, dtype='int')

    scores = categorization_score(idx_ref, gt.is_relevant.values,
                               idx_res, predictions)

    print('    => Test scores: MAP = {average_precision:.3f}, ROC-AUC = {roc_auc:.3f}'.format(**scores))
