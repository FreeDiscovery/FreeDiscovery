"""
Document Clustering example
---------------------------

Find duplicates in a text collection
"""

import os
import re
import numpy as np

import pandas as pd
import sys
import shutil
from time import time
import requests

pd.options.display.float_format = '{:,.3f}'.format

def _parent_dir(path, n=0):
    path = os.path.abspath(path)
    if n == 0:
        return path
    else:
        return os.path.dirname(_parent_dir(path, n=n-1))

def _print_url(op, url):
    print(' '*1, op, url) 

use_docker = False

dataset_name = "treclegal09_2k_subset"

if use_docker:
    data_dir = "/freediscovery_shared/{}".format(dataset_name)
else:
    data_dir = "../freediscovery_shared/{}".format(dataset_name)
rel_data_dir = os.path.abspath("../../freediscovery_shared/{}".format(dataset_name)) # relative path between this file and the FreeDiscovery source folder


BASE_URL = "http://localhost:5001/api/v0"  # FreeDiscovery local server URL


# # 1. Feature extraction (non hashed)

print("\n1.a Load dataset and initalize feature extraction")
url = BASE_URL + '/feature-extraction'
_print_url("POST", url)
fe_opts = {'data_dir': os.path.join(data_dir, 'data'),
           'stop_words': 'english', 'chunk_size': 2000, 'n_jobs': -1,
           'use_idf': 1, 'sublinear_tf': 0, 'binary': 0, 'n_features': 30001,
           'analyzer': 'word', 'ngram_range': (1, 1), "norm": "l2",
           'use_hashing': False,  # hashing should be disabled for clustering
           #'min_df': 0.2, 'max_df': 0.8
          }
res = requests.post(url, json=fe_opts)

dsid = res.json()['id']
print("   => received {}".format(list(res.json().keys())))
print("   => dsid = {}".format(dsid))


print("\n1.b Run feature extraction")
# progress status is available for the hashed version only
url = BASE_URL+'/feature-extraction/{}'.format(dsid)
_print_url("POST", url)
res = requests.post(url)

print("\n1.d. check the parameters of the extracted features")
url = BASE_URL + '/feature-extraction/{}'.format(dsid)
_print_url('GET', url)
res = requests.get(url)

data = res.json()
for key, val in data.items():
    if key!='filenames':
           print('     - {}: {}'.format(key, val))


# # 2. Duplicate detection by cosine similarity (DBSCAN)

print("\n2.a. Duplicate detection by cosine similarity (DBSCAN)")

url = BASE_URL + '/clustering/dbscan/'
_print_url("POST", url)
t0 = time()
res = requests.post(url,
        json={'dataset_id': dsid,
              'lsi_components': 100,
              'eps': 0.1,            # threashold for 2 documents considered to be duplicates
              'n_max_samples': 2
              }) 

data = res.json()
mid  = data['id']
print("     => model id = {}".format(mid))

print("\n2.b. Computing cluster labels")
url = BASE_URL + '/clustering/k-mean/{}'.format(mid)
_print_url("POST", url)
res = requests.get(url,
        json={'n_top_words': 0, # don't compute cluster labels
              }) 
data = res.json()
t1 = time()

print('    .. computed in {:.1f}s'.format(t1 - t0))

labels_ = data['labels']

print('Found {} duplicates / {}'.format(len(labels_) - len(np.unique(labels_)), len(labels_)))


# # 3. Duplicate detection by Simhash

print("\n3. Duplicate detection by Simhash")

url = BASE_URL + '/duplicate-detection/'
_print_url("POST", url)
t0 = time()
res = requests.post(url,
        json={'dataset_id': dsid,
              'method': 'simhash',
              }) 

data = res.json()
mid  = data['id']
print("     => model id = {}".format(mid))

print('    .. computed in {:.1f}s'.format(time() - t0))



url = BASE_URL + '/duplicate-detection/{}'.format(mid)
_print_url("GET", url)
t0 = time()
res = requests.get(url,
        json={'distance': 1 }) 
data = res.json()
print('    .. computed in {:.1f}s'.format(time() - t0))

labels_ = data['cluster_id']

print('Found {} duplicates / {}'.format(len(labels_) - len(np.unique(labels_)), len(labels_)))


print("\n3. Duplicate detection by I-Match")

url = BASE_URL + '/duplicate-detection/'
_print_url("POST", url)
t0 = time()
res = requests.post(url,
        json={'dataset_id': dsid,
              'method': 'i-match',
              }) 

data = res.json()
mid  = data['id']
print("     => model id = {}".format(mid))

print('    .. computed in {:.1f}s'.format(time() - t0))


url = BASE_URL + '/duplicate-detection/{}'.format(mid)
_print_url("GET", url)
t0 = time()
res = requests.get(url,
        json={'n_rand_lexicons': 10,
              'rand_lexicon_ratio': 0.9}) 
data = res.json()
t1 = time()
print('    .. computed in {:.1f}s'.format(time() - t0))

labels_ = data['cluster_id']

print('Found {} duplicates / {}'.format(len(labels_) - len(np.unique(labels_)), len(labels_)))

