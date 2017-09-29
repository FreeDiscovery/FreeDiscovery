"""
Hierarchical Clustering Example
===============================

Visualize hierarchical clusters
"""

import os.path
import pandas as pd
from time import time
import requests
from graphviz import Digraph


dataset_name = "20_newsgroups_3categories"     # see list of available datasets

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
# # 1. Feature extraction (non hashed)
# ------------------------------------
# 1.a Load dataset and initalize feature extraction

url = BASE_URL + '/feature-extraction'
print(" POST", url)
fe_opts = {'max_df': 0.6,  # filter out (too)/(un)frequent words
           'weighting': "ntc",
           }
res = requests.post(url, json=fe_opts).json()

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

n_components = 100
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
# 3. Document Clustering (LSI + Birch Clustering)
# -----------------------------------------------
# 3.a. Document clustering (LSI + Birch clustering)

url = BASE_URL + '/clustering/birch/'
print(" POST", url)
t0 = time()
res = requests.post(url,
                    json={'parent_id': lsi_id,
                          'n_clusters': -1,
                          'min_similarity': 0.55,
                          #'max_tree_depth': 3,
                          }).json()

mid = res['id']
print("     => model id = {}".format(mid))

print("\n4.b. Computing cluster labels")
url = BASE_URL + '/clustering/birch/{}'.format(mid)
print(" GET", url)
res = requests.get(url,
                   json={'n_top_words': 3
                         }).json()
t1 = time()

print('    .. computed in {:.1f}s'.format(t1 - t0))
data = res['data']

print(pd.DataFrame(data))


###############################################################################
#
# 3.b Hierarchical cluster visualization

ch = Digraph('cluster_hierarchy',
             node_attr={'shape': 'record'},
             format='png')

ch.graph_attr['rankdir'] = 'LR'
ch.graph_attr['dpi'] = "200"

for row in data:
    ch.node('cluster_{}'.format(row['cluster_id']),
            '{{<f0>{}| {{<f1> id={:03}  |<f2> N={} |<f3> sim={:.2f} }}}}'
            .format(row['cluster_label'],
                    row['cluster_id'],
                    row['cluster_size'],
                    row['cluster_similarity']))


def create_hc_links(node, ch, data):
    for child_id in node['children']:
        ch.edge('cluster_{}:f2'.format(node['cluster_id']),
                'cluster_{}:f0'.format(child_id))
        create_hc_links(data[child_id], ch, data)


create_hc_links(data[0], ch, data)

tmp_dir = os.path.join('..', '..', 'doc', 'engine', 'examples')
if os.path.exists(tmp_dir):
    ch.render('cluster_hierarchy', directory=tmp_dir, cleanup=True)
else:
    ch.view()

####################################
# .. image:: cluster_hierarchy.png

###############################################################################
#
# 4. Delete the extracted features

url = BASE_URL + '/feature-extraction/{}'.format(dsid)
requests.delete(url)
