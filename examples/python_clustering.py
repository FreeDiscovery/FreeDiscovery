"""
Clustering Example [Python API]
-------------------------------

An example of clustering using the Python API
"""

import pandas as pd
from freediscovery.text import FeatureVectorizer
from freediscovery.cluster import _ClusteringWrapper
from freediscovery.lsi import _LSIWrapper
from freediscovery.datasets import load_dataset
from freediscovery.tests.run_suite import check_cache
from time import time

pd.options.display.float_format = '{:,.3f}'.format

dataset_name = "treclegal09_2k_subset"
cache_dir = check_cache(test_env=False)


print("0. Load Dataset")

md, training_set, dataset = load_dataset(dataset_name, cache_dir=cache_dir)


print("\n1. Feature extraction (non hashed)\n")

n_features = 30000
fe = FeatureVectorizer(cache_dir=cache_dir)
uuid = fe.preprocess(md['data_dir'],
                     n_features=n_features, use_hashing=False,
                     use_idf=True, stop_words='english')
uuid, filenames = fe.transform()




n_clusters = 10
n_top_words = 6
lsi_components = 50


def repr_clustering(_labels, _terms):
    out = []
    for ridx, row in enumerate(_terms):
        out.append({'cluster_names': row, 'N_documents': (_labels == ridx).sum()})
    out = pd.DataFrame(out).sort_values('N_documents', ascending=False)
    return out

print("\n2. Computing LSI\n")
lsi = _LSIWrapper(cache_dir=cache_dir, parent_id=uuid)
lsi_res, exp_var = lsi.fit_transform(n_components=lsi_components)  # TODO unused variables



print("\n3. Document Clustering (LSI + K-Means)\n")

cat = _ClusteringWrapper(cache_dir=cache_dir, parent_id=lsi.mid)

t0 = time()
labels, tree  = cat.k_means(n_clusters)
terms = cat.compute_labels(n_top_words=n_top_words)
t1 = time()

print('    .. computed in {:.1f}s'.format(t1 - t0))
print(repr_clustering(labels, terms))


print('\n4. Document Clustering (LSI + Ward Hierarchical Clustering)\n')


t0 = time()
labels, tree = cat.ward_hc(n_clusters,
                           n_neighbors=5   # this is the connectivity constraint
                           )
terms = cat.compute_labels(n_top_words=n_top_words)
t1 = time()

print('    .. computed in {:.1f}s'.format(t1 - t0))
print(repr_clustering(labels, terms))
