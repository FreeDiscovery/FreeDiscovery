"""
Clustering Example [Python API]
-------------------------------

An example of clustering using the Python API
"""

import pandas as pd
from freediscovery.text import FeatureVectorizer
from freediscovery.cluster import _ClusteringWrapper
from freediscovery.utils import _silent
from freediscovery.datasets import load_dataset
from time import time

pd.options.display.float_format = '{:,.3f}'.format

dataset_name = "treclegal09_2k_subset"
cache_dir = '..'


print("0. Load Dataset")

ds = load_dataset(dataset_name, cache_dir=cache_dir)


print("\n1. Feature extraction (non hashed)\n")

n_features = 30000
fe = FeatureVectorizer(cache_dir=cache_dir)
uuid = fe.preprocess(ds['data_dir'],
                     n_features=n_features, use_hashing=False,
                     use_idf=True, stop_words='english')
uuid, filenames = fe.transform()


print("\n2. Document Clustering (LSI + K-Means)\n")

cat = _ClusteringWrapper(cache_dir=cache_dir, dsid=uuid)

n_clusters = 10
n_top_words = 6
lsi_components = 50


def repr_clustering(_labels, _terms):
    out = []
    for ridx, row in enumerate(_terms):
        out.append({'cluster_names': row, 'N_documents': (_labels == ridx).sum()})
    out = pd.DataFrame(out).sort_values('N_documents', ascending=False)
    return out


t0 = time()
with _silent('stderr'): # ignore some deprecation warnings
    labels, tree  = cat.k_means(n_clusters, lsi_components=lsi_components)
    terms = cat.compute_labels(n_top_words=n_top_words)
t1 = time()

print('    .. computed in {:.1f}s'.format(t1 - t0))
print(repr_clustering(labels, terms))


print('\n3. Document Clustering (LSI + Ward Hierarchical Clustering)\n')

t0 = time()
with _silent('stderr'): # ignore some deprecation warnings
    labels, tree = cat.ward_hc(n_clusters,
                               lsi_components=lsi_components,
                               n_neighbors=5   # this is the connectivity constraint
                               )
    terms = cat.compute_labels(n_top_words=n_top_words)
t1 = time()

print('    .. computed in {:.1f}s'.format(t1 - t0))
print(repr_clustering(labels, terms))
