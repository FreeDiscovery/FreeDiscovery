"""
Duplicate Detection Example [REST API]
--------------------------------------

Find near-duplicates in a text collection
"""
from __future__ import print_function

from time import time
import sys
import platform
import os

import pandas as pd
import requests

pd.options.display.float_format = '{:,.3f}'.format


dataset_name = "treclegal09_2k_subset"     # see list of available datasets

BASE_URL = "http://localhost:5001/api/v0"  # FreeDiscovery server URL

print(" 0. Load the test dataset")
url = BASE_URL + '/example-dataset/{}'.format(dataset_name)
print(" GET", url)
input_ds = requests.get(url).json()


# To use a custom dataset, simply specify the following variables
data_dir = input_ds['metadata']['data_dir']

# # 1. Feature extraction (non hashed)

print("\n1.a Load dataset and initalize feature extraction")
url = BASE_URL + '/feature-extraction'
print(" POST", url)
fe_opts = {'use_idf': 1,  # this is required to compute cluster labels (for now)
           'n_features': 30001,
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
res = requests.post(url, json={"data_dir": data_dir})

print("\n1.d. check the parameters of the extracted features")
url = BASE_URL + '/feature-extraction/{}'.format(dsid)
print(' GET', url)
res = requests.get(url)

data = res.json()
print('\n'.join(['     - {}: {}'.format(key, val)
      for key, val in data.items() if "filenames" not in key]))


print("\n2. Near Duplicates detection by cosine similarity (DBSCAN)")


# compute LSI used for DBSCAN clustering
url = BASE_URL + '/lsi/'
print("POST", url)

n_components = 100
res = requests.post(url,
                    json={'n_components': n_components,
                          'parent_id': dsid
                          }).json()

lsi_id = res['id']

url = BASE_URL + '/clustering/dbscan/'
print(" POST", url)
t0 = time()
res = requests.post(url,
        json={'parent_id': lsi_id,
              'min_similarity': 0.90,            # 2*cosine distance for documents to be considered as duplicates
              'n_max_samples': 2
              }).json()

mid = res['id']
print("     => model id = {}".format(mid))

url = BASE_URL + '/clustering/dbscan/{}'.format(mid)
print(" GET", url)
# don't compute cluster labels
res = requests.get(url, json={'n_top_words': 0}).json()
t1 = time()

print('    .. computed in {:.1f}s'.format(t1 - t0))

data = res['data']
print('Found {} duplicates / {}'
      .format(sum([len(row['documents'])
                   for row in data if len(row['documents']) > 1]),
              len(input_ds['dataset'])))


print("\n3. Near Duplicates Detection using I-Match")

url = BASE_URL + '/duplicate-detection/'
print(" POST", url)
t0 = time()
res = requests.post(url, json={'parent_id': dsid,
                               'method': 'i-match'})

data = res.json()
mid = data['id']
print("     => model id = {}".format(mid))

print('    .. computed in {:.1f}s'.format(time() - t0))


url = BASE_URL + '/duplicate-detection/{}'.format(mid)
print(" GET", url)
t0 = time()
res = requests.get(url, json={'n_rand_lexicons': 10,
                              'rand_lexicon_ratio': 0.9}).json()
t1 = time()
print('    .. computed in {:.1f}s'.format(time() - t0))

data = res['data']

print('Found {} duplicates / {}'
      .format(sum([len(row['documents']) for row in data]),
              len(input_ds['dataset'])))


# don't run the end of this example if simhash is not installed
# (e.g. on Windows)
try:
    import simhash
    skip_example = False
except:
    skip_example = True


if not skip_example:
    print("\n3. Duplicate detection by Simhash")

    url = BASE_URL + '/duplicate-detection/'
    print(" POST", url)
    t0 = time()
    res = requests.post(url, json={'parent_id': dsid,
                                   'method': 'simhash'})

    data = res.json()
    mid = data['id']
    print("     => model id = {}".format(mid))

    print('    .. computed in {:.1f}s'.format(time() - t0))

    url = BASE_URL + '/duplicate-detection/{}'.format(mid)
    print(" GET", url)
    t0 = time()
    res = requests.get(url, json={'distance': 1})
    data = res.json()
    print('    .. computed in {:.1f}s'.format(time() - t0))

    data = data['data']

    print('Found {} duplicates / {}'
          .format(sum([len(row['documents']) for row in data]),
                  len(input_ds['dataset'])))

# 4. Cleaning
print("\n4.a Delete the extracted features")
url = BASE_URL + '/feature-extraction/{}'.format(dsid)
print(" DELETE", url)
requests.delete(url)
