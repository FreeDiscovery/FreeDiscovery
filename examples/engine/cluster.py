"""
Clustering
==========

Cluster documents into clusters
"""

import os.path
import pandas as pd
from time import time
import requests

pd.options.display.float_format = '{:,.3f}'.format


dataset_name = "treclegal09_2k_subset"     # see list of available datasets

BASE_URL = "http://localhost:5001/api/v0"  # FreeDiscovery server URL

###############################################################################
#
# 0. Load the example dataset
# ---------------------------

url = BASE_URL + '/example-dataset/{}'.format(dataset_name)
print(" GET", url)
input_ds = requests.get(url).json()

# To use a custom dataset, simply specify the following variables
data_dir = input_ds['metadata']['data_dir']
dataset_definition = [{'document_id': row['document_id'],
                       'file_path': os.path.join(data_dir, row['file_path'])}
                      for row in input_ds['dataset']]

###############################################################################
#
# 1. Feature extraction (non hashed)
# ----------------------------------
# 1.a Load dataset and initalize feature extraction

url = BASE_URL + '/feature-extraction'
print(" POST", url)
res = requests.post(url).json()

dsid = res['id']
print("   => received {}".format(list(res.keys())))
print("   => dsid = {}".format(dsid))


###############################################################################
#
# 1.b Run feature extraction

url = BASE_URL+'/feature-extraction/{}'.format(dsid)
print(" POST", url)
res = requests.post(url, json={'dataset_definition': dataset_definition})


###############################################################################
#
# 2. Calculate LSI
# ----------------

url = BASE_URL + '/lsi/'
print("POST", url)

n_components = 300
res = requests.post(url,
                    json={'n_components': n_components,
                          'parent_id': dsid
                          }).json()

lsi_id = res['id']
print('  => LSI model id = {}'.format(lsi_id))
print(('  => SVD decomposition with {} dimensions '
       'explaining {:.2f} % variabilty of the data')
      .format(n_components, res['explained_variance']*100))

###############################################################################
#
# 3. Document Clustering (LSI + K-Means)
# --------------------------------------

print("\n3.a. Document clustering (LSI + K-means)")

url = BASE_URL + '/clustering/k-mean/'
print(" POST", url)
t0 = time()
res = requests.post(url,
                    json={'parent_id': lsi_id,
                          'n_clusters': 10,
                          }).json()

mid = res['id']
print("     => model id = {}".format(mid))

###############################################################################
#
# 3.b. Computing cluster labels

url = BASE_URL + '/clustering/k-mean/{}'.format(mid)
print(" GET", url)
res = requests.get(url,
                   json={'n_top_words': 3
                         }).json()
t1 = time()


data = res['data']
for row in data:
    row['n_documents'] = len(row.pop('documents'))

print('    .. computed in {:.1f}s'.format(t1 - t0))
print(pd.DataFrame(data))


###############################################################################
#
# 4. Document Clustering (LSI + Birch Clustering)
# -----------------------------------------------

print("\n4.a. Document clustering (LSI + Birch clustering)")

url = BASE_URL + '/clustering/birch/'
print(" POST", url)
t0 = time()
res = requests.post(url,
                    json={'parent_id': lsi_id,
                          'n_clusters': -1,
                          'min_similarity': 0.7,
                          'branching_factor': 20,
                          'max_tree_depth': 2,
                          }).json()

mid = res['id']
print("     => model id = {}".format(mid))

###############################################################################
#
# 4.b. Computing cluster labels

url = BASE_URL + '/clustering/birch/{}'.format(mid)
print(" GET", url)
res = requests.get(url,
                   json={'n_top_words': 3
                         }).json()
t1 = time()

print('    .. computed in {:.1f}s'.format(t1 - t0))
data = res['data']
for row in data:
    row['n_documents'] = len(row.pop('documents'))

print(pd.DataFrame(data))

###############################################################################
#
# 5. Optimal sampling (LSI + Birch Clustering)
# --------------------------------------------
t0 = time()
url = BASE_URL + '/clustering/birch/{}'.format(mid)
print(" GET", url)
res = requests.get(url,
                   json={'return_optimal_sampling': True,
                         'sampling_min_coverage': 0.9
                         }).json()
t1 = time()
print('    .. computed in {:.1f}s'.format(t1 - t0))
data = res['data']

print(pd.DataFrame(data))

###############################################################################
#
# 5. Delete the extracted features

url = BASE_URL + '/feature-extraction/{}'.format(dsid)
requests.delete(url)
