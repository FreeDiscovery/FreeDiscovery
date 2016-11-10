"""
Clustering Example [REST API]
-----------------------------

Cluster documents into clusters
"""

import numpy as np
import pandas as pd
from time import time
import requests

pd.options.display.float_format = '{:,.3f}'.format


def repr_clustering(labels, terms):
    out = []
    for ridx, row in enumerate(terms):
        out.append({'cluster_names': row, 'N_documents': (labels == ridx).sum()})
    out = pd.DataFrame(out).sort_values('N_documents', ascending=False)
    return out

dataset_name = "treclegal09_2k_subset"     # see list of available datasets

BASE_URL = "http://localhost:5001/api/v0"  # FreeDiscovery server URL

print(" 0. Load the test dataset")
url = BASE_URL + '/datasets/{}'.format(dataset_name)
print(" POST", url)
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
           'use_idf': 1, 'sublinear_tf': 1, 'binary': 0, 'n_features': 30001,
           'analyzer': 'word', 'ngram_range': (1, 1), "norm": "l2",
           'use_hashing': False  # hashing should be disabled for clustering
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


# # 2. Document Clustering (LSI + K-Means)

print("\n2.a. Document clustering (LSI + K-means)")

url = BASE_URL + '/clustering/k-mean/'
print(" POST", url)
t0 = time()
res = requests.post(url,
                    json={'dataset_id': dsid,
                          'n_clusters': 10,
                          'lsi_components': 50
                          })

data = res.json()
mid = data['id']
print("     => model id = {}".format(mid))

print("\n2.b. Computing cluster labels")
url = BASE_URL + '/clustering/k-mean/{}'.format(mid)
print(" POST", url)
res = requests.get(url,
                   json={'n_top_words': 6
                         })
data = res.json()
t1 = time()

print('    .. computed in {:.1f}s'.format(t1 - t0))
print(repr_clustering(np.array(data['labels']), data['cluster_terms']))


# # 3. Document Clustering (LSI + Ward Hierarchical Clustering)

print("\n2.a. Document clustering (LSI + Ward HC)")

url = BASE_URL + '/clustering/ward_hc/'
print(" POST", url)
t0 = time()
res = requests.post(url,
                    json={'dataset_id': dsid,
                          'n_clusters': 10,
                          'lsi_components': 50,
                          'n_neighbors': 5  # this is the connectivity constraint
                          })

data = res.json()
mid = data['id']
print("     => model id = {}".format(mid))

print("\n2.b. Computing cluster labels")
url = BASE_URL + '/clustering/ward_hc/{}'.format(mid)
print("POST", url)
res = requests.get(url,
                   json={'n_top_words': 6
                         })
data = res.json()
t1 = time()

print('    .. computed in {:.1f}s'.format(t1 - t0))
print(repr_clustering(np.array(data['labels']), data['cluster_terms']))
