"""
Duplicate Detection Example [REST API]
--------------------------------------

Find near-duplicates in a text collection
"""
from __future__ import print_function

from time import time
import sys
import platform

import numpy as np
import pandas as pd
import requests

pd.options.display.float_format = '{:,.3f}'.format


dataset_name = "treclegal09_2k_subset"     # see list of available datasets

BASE_URL = "http://localhost:5001/api/v0"  # FreeDiscovery server URL

print(" 0. Load the test dataset")
url = BASE_URL + '/datasets/{}'.format(dataset_name)
print(" GET", url)
res = requests.get(url)
res = res.json()

# To use a custom dataset, simply specify the following variables
data_dir = res['data_dir']
# # 1. Feature extraction (non hashed)

print("\n1.a Load dataset and initalize feature extraction")
url = BASE_URL + '/feature-extraction'
print(" POST", url)
fe_opts = {'data_dir': data_dir,
           'stop_words': 'english', 'chunk_size': 2000, 'n_jobs': -1,
           'use_idf': 1, 'sublinear_tf': 0, 'binary': 0, 'n_features': 30001,
           'analyzer': 'word', 'ngram_range': (1, 1), "norm": "l2",
           'use_hashing': False,  # hashing should be disabled for clustering
           'min_df': 4, 'max_df': 0.75
          }
res = requests.post(url, json=fe_opts)

dsid = res.json()['id']
print("   => received {}".format(list(res.json().keys())))
print("   => dsid = {}".format(dsid))


print("\n1.b Run feature extraction")
# progress status is available for the hashed version only
url = BASE_URL+'/feature-extraction/{}'.format(dsid)
print(" POST", url)
res = requests.post(url)

print("\n1.d. check the parameters of the extracted features")
url = BASE_URL + '/feature-extraction/{}'.format(dsid)
print(' GET', url)
res = requests.get(url)

data = res.json()
print('\n'.join(['     - {}: {}'.format(key, val) for key, val in data.items() \
                                                  if "filenames" not in key]))


# # 2. Near Duplicates detection by cosine similarity (DBSCAN)
print("\n2. Near Duplicates detection by cosine similarity (DBSCAN)")

url = BASE_URL + '/clustering/dbscan/'
print(" POST", url)
t0 = time()
res = requests.post(url,
        json={'dataset_id': dsid,
              'lsi_components': 100,
              'eps': 0.1,            # 2*cosine distance for documents to be considered as duplicates
              'n_max_samples': 2
              }).json()

mid  = res['id']
print("     => model id = {}".format(mid))

url = BASE_URL + '/clustering/dbscan/{}'.format(mid)
print(" GET", url)
res = requests.get(url,
        json={'n_top_words': 0, # don't compute cluster labels
              }).json()
t1 = time()

print('    .. computed in {:.1f}s'.format(t1 - t0))

labels_ = res['labels']

print('Found {} duplicates / {}'.format(len(labels_) - len(np.unique(labels_)), len(labels_)))


# # 3. Near Duplicates detection by cosine similarity (DBSCAN)
print("\n3. Near Duplicates Detection using I-Match")

url = BASE_URL + '/duplicate-detection/'
print(" POST", url)
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
print(" GET", url)
t0 = time()
res = requests.get(url,
        json={'n_rand_lexicons': 10,
              'rand_lexicon_ratio': 0.9}).json()
t1 = time()
print('    .. computed in {:.1f}s'.format(time() - t0))

labels_ = res['cluster_id']

print('Found {} duplicates / {}'.format(len(labels_) - len(np.unique(labels_)), len(labels_)))


# 4. Cleaning
print("\n4.a Delete the extracted features")
url = BASE_URL + '/feature-extraction/{}'.format(dsid)
print(" DELETE", url)
requests.delete(url)



# Commenting out simhash for the moment
# 
#if platform.system() == 'Windows':
#    print('Simhash-py is currently not installed on Windows')
#    sys.exit()
#
#print("\n3. Duplicate detection by Simhash")
#
#url = BASE_URL + '/duplicate-detection/'
#print(" POST", url)
#t0 = time()
#res = requests.post(url,
#        json={'dataset_id': dsid,
#              'method': 'simhash',
#              }) 
#
#data = res.json()
#mid  = data['id']
#print("     => model id = {}".format(mid))
#
#print('    .. computed in {:.1f}s'.format(time() - t0))
#
#
#
#url = BASE_URL + '/duplicate-detection/{}'.format(mid)
#print(" GET", url)
#t0 = time()
#res = requests.get(url,
#        json={'distance': 1 }) 
#data = res.json()
#print('    .. computed in {:.1f}s'.format(time() - t0))
#
#labels_ = data['cluster_id']
#
#print('Found {} duplicates / {}'.format(len(labels_) - len(np.unique(labels_)), len(labels_)))



