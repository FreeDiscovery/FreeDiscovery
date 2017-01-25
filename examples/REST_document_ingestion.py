"""
Categorization Example [REST API]
---------------------------------

An example to illustrate binary categorizaiton with FreeDiscovery
"""

from __future__ import print_function

from time import time, sleep
from multiprocessing import Process
import requests
import pandas as pd
import sys

pd.options.display.float_format = '{:,.3f}'.format
pd.options.display.expand_frame_repr = False

dataset_name = "treclegal09_2k_subset"     # see list of available datasets

BASE_URL = "http://localhost:5001/api/v0"  # FreeDiscovery server URL

if __name__ == '__main__':

    print(" 0. Load the test dataset")
    url = BASE_URL + '/datasets/{}'.format(dataset_name)
    print(" GET", url)
    res = requests.get(url, json={'return_file_path': True}).json()

    # To use a custom dataset, simply specify the following variables
    data_dir = res['data_dir']
    seed_filenames = res['seed_filenames']
    seed_y = res['seed_y']
    ground_truth_file = res['ground_truth_file']  # (optional)
    file_path_list = res['file_path']

    dataset_definition = []
    for idx, fname in enumerate(file_path_list):
        dataset_definition.append({'document_id': idx + 10000, # define some external document id
                                  'rendering_id': 0,
                                  'file_path': fname})
    # 1. Feature extraction

    print("\n1.a Load dataset and initalize feature extraction")
    url = BASE_URL + '/feature-extraction'
    print(" POST", url)
    res = requests.post(url, json={'dataset_definition': dataset_definition,
                                   'use_hashing': True}).json()

    dsid = res['id']
    print("   => received {}".format(list(res.keys())))
    print("   => dsid = {}".format(dsid))


    print("\n1.b Start feature extraction (in the background)")

    # Run feature extraction (see REST_categorization
    # example for a background usage)
    url = BASE_URL+'/feature-extraction/{}'.format(dsid)
    print(" POST", url)
    res = requests.post(url,)


    print("\n1.d. check the parameters of the extracted features")
    url = BASE_URL + '/feature-extraction/{}'.format(dsid)
    print(' GET', url)
    res = requests.get(url).json()

    print('\n'.join(['     - {}: {}'.format(key, val) for key, val in res.items() \
                                                      if "filenames" not in key]))

    method = BASE_URL + "/feature-extraction/{}/index".format(dsid)
    res = requests.get(method, data={'filenames': seed_filenames})
    seed_index = res.json()['index']


    print("\n3.a. Train the categorization model")
    print("   {} relevant, {} non-relevant files".format(seed_y.count(1), seed_y.count(0)))

    method = 'LinearSVC'
    parent_id = dsid

    url = BASE_URL + '/categorization/'
    print(" POST", url)
    print(' Training...')

    res = requests.post(url,
                        json={'index': seed_index,
                              'y': seed_y,
                              'parent_id': parent_id,
                              'method': method,  # one of "LinearSVC", "LogisticRegression", 'xgboost'
                              }).json()

    mid = res['id']
    print("     => model id = {}".format(mid))
    print('    => Training scores: MAP = {average_precision:.3f}, ROC-AUC = {roc_auc:.3f}'.format(**res))

    print("\n3.b. Check the parameters used in the categorization model")
    url = BASE_URL + '/categorization/{}'.format(mid)
    print(" GET", url)
    res = requests.get(url).json()

    print('\n'.join(['     - {}: {}'.format(key, val) for key, val in res.items() \
                                                      if key not in ['index', 'y']]))

    print("\n3.c Categorize the complete dataset with this model")
    url = BASE_URL + '/categorization/{}/predict'.format(mid)
    print(" GET", url)
    res = requests.get(url).json()

    data = res['data']

    print(pd.DataFrame(data).set_index('internal_id'))

    print("\n3.d Test categorization accuracy")
    print("         using {}".format(ground_truth_file))  
    url = BASE_URL + '/categorization/{}/test'.format(mid)
    print("POST", url)
    res = requests.post(url, json={'ground_truth_filename': ground_truth_file}).json()

    print('    => Test scores: MAP = {average_precision:.3f}, ROC-AUC = {roc_auc:.3f}'.format(**res))

    # 4. Cleaning
    print("\n5.a Delete the extracted features (and LSI decomposition)")
    url = BASE_URL + '/feature-extraction/{}'.format(dsid)
    print(" DELETE", url)
    requests.delete(url)
