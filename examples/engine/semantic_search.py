"""
Semantic Search
===============

An example of Semantic Search
"""

from __future__ import print_function

import os.path
import requests
import pandas as pd

pd.options.display.float_format = '{:,.3f}'.format
pd.options.display.expand_frame_repr = False

dataset_name = "treclegal09_2k_subset"     # see list of available datasets

BASE_URL = "http://localhost:5001/api/v0"  # FreeDiscovery server URL

###############################################################################
#
# 0. Load the test dataset
# ------------------------

url = BASE_URL + '/example-dataset/{}'.format(dataset_name)
print(" GET", url)
input_ds = requests.get(url).json()

# create a custom dataset definition for ingestion
data_dir = input_ds['metadata']['data_dir']
dataset_definition = [{'document_id': row['document_id'],
                       'file_path': os.path.join(data_dir, row['file_path'])}
                      for row in input_ds['dataset']]

###############################################################################
#
# 1. Feature extraction
# ---------------------
# 1.a Load dataset and initalize feature extraction

url = BASE_URL + '/feature-extraction'
print(" POST", url)
res = requests.post(url).json()

dsid = res['id']
print("   => received {}".format(list(res.keys())))
print("   => dsid = {}".format(dsid))

###############################################################################
#
# 1.b Start feature extraction

url = BASE_URL+'/feature-extraction/{}'.format(dsid)
print(" POST", url)
requests.post(url, json={'dataset_definition': dataset_definition})

###############################################################################
#
# 2. Calculate LSI
# ----------------
# (used for Nearest Neighbors method)

url = BASE_URL + '/lsi/'
print("POST", url)
n_components = 100
res = requests.post(url,
                    json={'n_components': n_components,
                          'parent_id': dsid
                          }).json()

lsi_id = res['id']
print('  => LSI model id = {}'.format(lsi_id))
print(("  => SVD decomposition with {} dimensions explaining "
       "{:.2f} % variabilty     of the data")
      .format(n_components, res['explained_variance']*100))


###############################################################################
#
# 3. Semantic search
# ------------------

print("\n3.a. Perform the semantic search")


query = ("There are some conflicts with the draft date, so we will probably "
         "need to have it on a different date.")

url = BASE_URL + '/search/'
print(" POST", url)

res = requests.post(url,
                    json={'parent_id': lsi_id,
                          'query': query
                          }).json()

data = res['data']

df = pd.DataFrame(data).set_index('document_id')
print(df)

print(df.score.max())


###############################################################################
#
# Delete the extracted features

url = BASE_URL + '/feature-extraction/{}'.format(dsid)
requests.delete(url)
