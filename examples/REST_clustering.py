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
res = requests.get(url).json()

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
           'use_hashing': False,  # hashing should be disabled for clustering
           'min_df': 4, 'max_df': 0.75
           }
res = requests.post(url, json=fe_opts).json()

dsid = res['id']
print("   => received {}".format(list(res.keys())))
print("   => dsid = {}".format(dsid))


print("\n1.b Run feature extraction")
# progress status is available for the hashed version only
url = BASE_URL+'/feature-extraction/{}'.format(dsid)
print(" POST", url)
res = requests.post(url)

print("\n1.d. check the parameters of the extracted features")
url = BASE_URL + '/feature-extraction/{}'.format(dsid)
print(' GET', url)
res = requests.get(url).json()

print('\n'.join(['     - {}: {}'.format(key, val) for key, val in res.items() \
                                                  if "filenames" not in key]))

print("\n2. Calculate LSI")

url = BASE_URL + '/lsi/'
print("POST", url)

n_components = 100
res = requests.post(url,
                    json={'n_components': n_components,
                          'parent_id': dsid
                          }).json()

lsi_id = res['id']
print('  => LSI model id = {}'.format(lsi_id))
print('  => SVD decomposition with {} dimensions explaining {:.2f} % variabilty of the data'.format(
                        n_components, res['explained_variance']*100))

# # 3. Document Clustering (LSI + K-Means)

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

print("\n3.b. Computing cluster labels")
url = BASE_URL + '/clustering/k-mean/{}'.format(mid)
print(" GET", url)
res = requests.get(url,
                   json={'n_top_words': 6
                         }).json()
t1 = time()

print('    .. computed in {:.1f}s'.format(t1 - t0))
print(repr_clustering(np.array(res['labels']), res['cluster_terms']))


# # 4. Document Clustering (LSI + Ward Hierarchical Clustering)

print("\n4.a. Document clustering (LSI + Ward HC)")

url = BASE_URL + '/clustering/ward_hc/'
print(" POST", url)
t0 = time()
res = requests.post(url,
                    json={'parent_id': lsi_id,
                          'n_clusters': 10,
                          'n_neighbors': 5  # this is the connectivity constraint
                          }).json()

mid = res['id']
print("     => model id = {}".format(mid))

print("\n4.b. Computing cluster labels")
url = BASE_URL + '/clustering/ward_hc/{}'.format(mid)
print(" GET", url)
res = requests.get(url,
                   json={'n_top_words': 6
                         }).json()
t1 = time()

print('    .. computed in {:.1f}s'.format(t1 - t0))
print(repr_clustering(np.array(res['labels']), res['cluster_terms']))


# 4. Cleaning
print("\n5. Delete the extracted features")
url = BASE_URL + '/feature-extraction/{}'.format(dsid)
print(" DELETE", url)
requests.delete(url)
